import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


DATASETS = {
    "fleurs-en": {
        "id": "google/fleurs",
        "config": "en_us",
        "split": "train",
        "text": "transcription",
        "license": "CC BY 4.0",
        "language": "en",
    },
    "fleurs-hi": {
        "id": "google/fleurs",
        "config": "hi_in",
        "split": "train",
        "text": "transcription",
        "license": "CC BY 4.0",
        "language": "hi",
    },
    "librispeech-clean": {
        "id": "openslr/librispeech_asr",
        "config": "clean",
        "split": "train.100",
        "text": "text",
        "license": "CC BY 4.0",
        "language": "en",
    },
    "voxpopuli-en": {
        "id": "facebook/voxpopuli",
        "config": "en",
        "split": "train",
        "text": "normalized_text",
        "license": "CC0 1.0",
        "language": "en",
    },
}


def decode_audio(value):
    """Support both legacy datasets Audio dicts and newer torchcodec decoders."""
    if isinstance(value, dict):
        if value.get("array") is not None:
            return np.asarray(value["array"], dtype=np.float32), int(value["sampling_rate"])
        if value.get("path"):
            audio, sr = librosa.load(value["path"], sr=None, mono=True)
            return audio.astype(np.float32), int(sr)

    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=0)
        return data, int(samples.sample_rate)

    raise TypeError(f"Unsupported audio value: {type(value)!r}")


def main():
    parser = argparse.ArgumentParser(description="Download and normalize a public ASR dataset.")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--output", default="data/public")
    parser.add_argument("--limit", type=int, default=1000, help="0 means no limit")
    parser.add_argument("--split", help="Override the configured split")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream from Hugging Face instead of downloading the full dataset first",
    )
    args = parser.parse_args()

    spec = dict(DATASETS[args.dataset])
    split = args.split or spec["split"]
    out = Path(args.output) / args.dataset
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    ds = load_dataset(spec["id"], spec["config"], split=split, streaming=args.streaming)
    if hasattr(ds, "cast_column"):
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row_index, row in enumerate(tqdm(ds, desc=args.dataset)):
            if args.limit and count >= args.limit:
                break

            text = str(row.get(spec["text"], "")).strip()
            if not text:
                continue

            try:
                audio, sample_rate = decode_audio(row["audio"])
            except Exception as exc:
                print(f"skip row {row_index}: {exc}")
                continue

            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            filename = f"{count:08d}.wav"
            sf.write(audio_dir / filename, audio, 16000, subtype="PCM_16")

            record = {
                "audio": f"audio/{filename}",
                "prompt": f"Transcribe this {spec['language']} speech accurately.",
                "response": text,
                "task": "transcribe",
                "language": spec["language"],
                "source_dataset": spec["id"],
                "source_config": spec["config"],
                "source_split": split,
                "license": spec["license"],
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} examples to {manifest_path}")


if __name__ == "__main__":
    main()
