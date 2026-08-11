import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge AudioLLM manifests while preserving valid audio paths.")
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output", default="data/stage1.jsonl")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output.open("w", encoding="utf-8") as dst:
        for manifest_name in args.manifests:
            manifest = Path(manifest_name).resolve()
            with manifest.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    audio = Path(row["audio"])
                    if not audio.is_absolute():
                        audio = (manifest.parent / audio).resolve()
                    row["audio"] = os.path.relpath(audio, output.parent)
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Merged {written} rows into {output}")


if __name__ == "__main__":
    main()
