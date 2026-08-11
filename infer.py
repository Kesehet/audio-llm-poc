import argparse

import librosa
import torch
import yaml
from transformers import WhisperFeatureExtractor

from src.audio_llm import AudioLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = WhisperFeatureExtractor.from_pretrained(cfg["audio_encoder"])
    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    audio = audio[: int(cfg["max_audio_seconds"] * 16000)]
    features = extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    model = AudioLLM(cfg["audio_encoder"], cfg["llm"], cfg["projector_hidden"], cfg["audio_pool_stride"], cfg["system_prompt"]).to(device)
    model.projector.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(model.generate_from_audio(features, torch.tensor([len(audio)]), prompt=args.prompt))


if __name__ == "__main__":
    main()
