# audio-llm-poc

A minimal proof of concept for **audio → language model → text** without running a separate Whisper transcription decoder.

## Architecture

```text
16 kHz audio
   ↓
Whisper encoder (frozen)
   ↓
audio hidden states
   ↓
temporal average pooling
   ↓
trainable projector
   ↓
Qwen embedding space
   ↓
Qwen2.5-0.5B-Instruct (frozen)
   ↓
text response
```

The first experiment intentionally trains only the projector. This is cheap, easy to debug, and tells us whether Whisper representations can be aligned to the LLM strongly enough to answer spoken requests. It is **not yet a streaming model** and it is **not meant to beat Whisper ASR**.

## Models

- Audio encoder: `openai/whisper-small`
- LLM: `Qwen/Qwen2.5-0.5B-Instruct`
- Trainable component: projector only

Both base checkpoints use Apache-2.0 model repositories. Check their model cards and licenses before production use.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Install the correct CUDA build of PyTorch for your machine if needed.

## Dataset

Create a JSONL manifest:

```json
{"audio":"audio/example.wav","response":"Sure. What is your account number?"}
```

Optional `prompt` lets one recording train under a specific instruction:

```json
{"audio":"audio/example.wav","prompt":"Answer as a bank support agent.","response":"Sure. What is your account number?"}
```

Paths are resolved relative to the manifest file. Audio is loaded mono at 16 kHz and clipped to 30 seconds by default.

See `examples/manifest.example.jsonl`.

## Train

```bash
python train.py --manifest path/to/train.jsonl
```

The default config uses batch size 1 with gradient accumulation and saves projector weights under `checkpoints/poc/`.

## Inference

```bash
python infer.py sample.wav \
  --checkpoint checkpoints/poc/projector-epoch-3.pt
```

Optional:

```bash
python infer.py sample.wav \
  --checkpoint checkpoints/poc/projector-epoch-3.pt \
  --prompt "Answer as a customer support agent."
```

## GPU expectations

This POC freezes both large pretrained components, so only the projector gets optimizer states and gradients. A 12–16 GB CUDA GPU is a realistic starting point; 24 GB gives much more room for experimentation. CPU execution is supported for debugging but will be slow.

If memory is tight, change `openai/whisper-small` to `openai/whisper-base` in `configs/poc.yaml`.

## What success looks like

Do not judge the first run by polished conversational quality. We first want to prove:

1. Training loss falls consistently.
2. The same spoken intents produce semantically related answers.
3. Unseen speakers can produce useful responses.
4. The model is using audio rather than memorizing dataset order.

A tiny dataset will overfit. That is useful as the first sanity check.

## Recommended experiment order

1. **Overfit 20–100 recordings.** Verify the architecture and loss path.
2. Train on several hours of diverse speech.
3. Add a transcription objective alongside direct-response training.
4. Add LoRA to the last LLM layers if projector-only alignment plateaus.
5. Replace fixed utterance processing with a streaming/chunked audio encoder.
6. Benchmark end-of-speech → first response token latency against the existing ASR → LLM pipeline.

## Important limitation

Qwen itself was pretrained on text tokens, not arbitrary Whisper vectors. A projector can learn an alignment, but projector-only training may plateau. That is an experimental question this repository is designed to answer cheaply. The next step would be projector + LoRA rather than training either foundation model from scratch.
