import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from src.audio_llm import AudioBatchCollator, AudioLLM, AudioTextDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = WhisperFeatureExtractor.from_pretrained(cfg["audio_encoder"])
    ds = AudioTextDataset(args.manifest)
    collator = AudioBatchCollator(extractor, cfg["max_audio_seconds"])
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collator,
    )

    model = AudioLLM(
        cfg["audio_encoder"],
        cfg["llm"],
        cfg["projector_hidden"],
        cfg["audio_pool_stride"],
        cfg["system_prompt"],
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    precision = str(cfg.get("mixed_precision", "fp32")).lower()
    use_fp16 = device.type == "cuda" and precision == "fp16"
    use_bf16 = (
        device.type == "cuda"
        and precision == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    autocast_enabled = use_fp16 or use_bf16
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(f"device={device} precision={precision} autocast={autocast_enabled}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    accum = cfg["grad_accum_steps"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(cfg["epochs"]):
        bar = tqdm(loader, desc=f"epoch {epoch + 1}")
        for step, batch in enumerate(bar, 1):
            feats = batch["input_features"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                out = model(
                    feats,
                    batch["num_samples"],
                    batch["responses"],
                    batch["prompts"],
                    cfg["max_target_tokens"],
                )
                loss = out.loss / accum

            scaler.scale(loss).backward()
            if step % accum == 0 or step == len(loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            bar.set_postfix(loss=float(loss.item() * accum))

        torch.save(
            model.projector.state_dict(),
            out_dir / f"projector-epoch-{epoch + 1}.pt",
        )


if __name__ == "__main__":
    main()
