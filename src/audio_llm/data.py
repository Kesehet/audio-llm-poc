import json
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset


class AudioTextDataset(Dataset):
    """JSONL rows: {"audio":"path.wav", "response":"...", "prompt":"optional"}."""

    def __init__(self, manifest: str):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        with self.manifest.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = dict(self.rows[idx])
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        row["audio_array"] = audio
        row["num_samples"] = len(audio)
        return row


class AudioBatchCollator:
    def __init__(self, feature_extractor, max_audio_seconds=30):
        self.feature_extractor = feature_extractor
        self.max_samples = int(max_audio_seconds * 16000)

    def __call__(self, batch):
        arrays = [x["audio_array"][: self.max_samples] for x in batch]
        features = self.feature_extractor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_samples,
            return_attention_mask=True,
        )
        return {
            "input_features": features.input_features,
            "wave_attention_mask": features.get("attention_mask"),
            "num_samples": torch.tensor([min(x["num_samples"], self.max_samples) for x in batch]),
            "responses": [x["response"] for x in batch],
            "prompts": [x.get("prompt", "") for x in batch],
        }
