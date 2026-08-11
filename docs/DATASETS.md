# Dataset plan

The first training stage is **speech alignment**: teach the projector to make the frozen LLM recover spoken content from Whisper encoder representations. Public ASR corpora therefore use their transcript as the target response.

## Included download targets

| Key | Source | Language | Style | License | Initial role |
|---|---|---|---|---|---|
| `fleurs-en` | `google/fleurs` (`en_us`) | English | read prompts, diverse speakers | CC BY 4.0 | small clean multilingual sanity set |
| `fleurs-hi` | `google/fleurs` (`hi_in`) | Hindi | read prompts, diverse speakers | CC BY 4.0 | Hindi speech alignment |
| `librispeech-clean` | `openslr/librispeech_asr` | English | audiobook/read speech | CC BY 4.0 | scale clean English alignment |
| `voxpopuli-en` | `facebook/voxpopuli` | English | parliamentary speech | CC0 1.0 | longer, more natural speech variation |

Always retain the source metadata and comply with attribution requirements for CC BY datasets when redistributing derived datasets or artifacts.

## Why GigaSpeech is not enabled by default

GigaSpeech is useful, but its current Hugging Face access flow is gated and includes research/education access terms. We keep it out of the default corpus until the intended use and terms are reviewed explicitly.

## Download small starter subsets

```bash
python scripts/prepare_public_asr.py fleurs-en --limit 1000
python scripts/prepare_public_asr.py fleurs-hi --limit 1000
python scripts/prepare_public_asr.py librispeech-clean --limit 3000
python scripts/prepare_public_asr.py voxpopuli-en --limit 3000
```

The downloader streams by default, so these commands do not require downloading an entire source corpus first. Each command produces:

```text
data/public/<dataset>/
├── audio/
│   ├── 00000000.wav
│   └── ...
└── manifest.jsonl
```

Every manifest row keeps `source_dataset`, `source_config`, `source_split`, `language`, `task`, and `license` fields in addition to the fields used by training.

## Stage 1 target mixture

Start small. A useful first mixture is roughly:

- 1,000 FLEURS English
- 1,000 FLEURS Hindi
- 3,000 LibriSpeech clean English
- 3,000 VoxPopuli English

This is not intended to produce a good production model. It is large enough to test whether the projector learns across speakers and speech styles without spending days moving data.

## Stage 2: direct spoken-response examples

ASR corpora only teach `audio -> transcript`. Our actual target is `audio -> useful assistant response`.

After Stage 1 works, create a second manifest where the same or similar spoken utterances have customer-support style answers as targets. Keep both tasks and distinguish them using prompts such as:

```text
Transcribe this speech accurately.
Answer the caller's request helpfully and briefly.
```

We should mix both tasks instead of throwing away transcription supervision. That gives us a direct diagnostic for whether an error came from speech understanding or response generation.

## Do not commit downloaded audio

`data/` is ignored by git. Keep large corpora on local/rented training storage or object storage, not in this source repository.
