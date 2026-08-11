import argparse
import io
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


def _mono_float32(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # soundfile returns [samples, channels]
        audio = audio.mean(axis=1)
    return audio


def decode_raw_audio(value):
    """Decode datasets Audio(decode=False) values without TorchCodec."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected raw audio dict, got {type(value)!r}")

    raw_bytes = value.get("bytes")
    path = value.get("path")

    if raw_bytes:
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        return _mono_float32(audio), int(sr)

    if path:
        # Local/cache paths can be read by soundfile/librosa. For uncommon codecs,
        # librosa/audioread provides a fallback.
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            return _mono_float32(audio), int(sr)
        except Exception:
            audio, sr = librosa.load(path, sr=None, mono=True)
            return np.asarray(audio, dtype=np.float32), int(sr)

    raise ValueError("Audio row has neither bytes nor path")


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

    print(f"Loading {spec['id']} / {spec['config']} / {split} (streaming={args.streaming})")
    ds = load_dataset(spec["id"], spec["config"], split=split, streaming=args.streaming)

    # IMPORTANT: do not let Hugging Face decode audio here. Current datasets
    # versions use TorchCodec/FFmpeg for Audio decoding, which can hard-abort in
    # some Colab runtimes. We request raw bytes/path and decode with soundfile.
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        # Streaming IterableDataset also exposes decode(False) in newer releases.
        if hasattr(ds, "decode"):
            ds = ds.decode(False)
        else:
            raise

    count = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row_index, row in enumerate(tqdm(ds, desc=args.dataset)):
            if args.limit and count >= args.limit:
                break

            text = str(row.get(spec["text"], "")).strip()
            if not text:
                skipped += 1
                continue

            try:
                audio, sample_rate = decode_raw_audio(row["audio"])
            except Exception as exc:
                skipped += 1
                print(f"skip row {row_index}: {type(exc).__name__}: {exc}")
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

    print(f"Wrote {count} examples to {manifest_path}; skipped {skipped}")
    if count == 0:
        raise RuntimeError("No audio examples were prepared. See skip messages above.")


if __name__ == "__main__":
    main()
