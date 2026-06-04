# Antzoki TTS — Basque Expressive Text-to-Speech

> **Antzoki** *(Basque)* — theatre, stage.  
> Expressive Basque speech synthesis with voice cloning, built on [DramaBox](https://github.com/resemble-ai/DramaBox).

**Antzoki TTS** is a [LoRA adapter](https://huggingface.co/itzune/antzoki-tts) fine-tuned on top of DramaBox (Resemble AI's cinematic TTS) that shifts the model's phonetic prior from English toward Basque, enabling natural-sounding Basque speech with dramatic delivery and voice cloning.

📖 **Blog post**: [Antzoki TTS — Euskarazko ahots sintesi adierazkorra](https://xezpeleta.github.io/blog/antzoki-tts/) (Basque)

---

## What can it do?

- **Voice cloning** — Provide a ~10-second reference clip of any speaker, and the model speaks Basque in their voice.
- **Director-style prompts** — Control emotion, pacing, breathing, and delivery through natural-language stage directions.
- **Expressive range** — From whispered intimacy to furious shouts, from joyful laughter to solemn grief.

### Audio samples

| Prompt | Audio |
|---|---|
| Shadowy villain with cold menace | [▶ shadowy_villain.mp3](output/eval/shadowy_villain.mp3) |
| Bright-eyed girl in a field of wildflowers | [▶ bright_eyed_girl.mp3](output/eval/bright_eyed_girl.mp3) |

---

## Quick start

### Prerequisites

| Component | Minimum |
|---|---|
| **GPU VRAM** | ~17 GB |
| **System RAM** | 16 GB |
| **Disk** | ~20 GB (models) |
| **CUDA** | 12.x+ |

### 1. Clone this repo with DramaBox submodule

```bash
git clone --recurse-submodules https://github.com/itzune/antzoki-tts.git
cd antzoki-tts
```

### 2. Set up Python environment with uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
cd DramaBox
uv venv --python 3.10
source .venv/bin/activate

# One command — the requirements file includes the CUDA index URL
uv pip install -r ../requirements-tts.txt
```

### 3. Download model weights (~17 GB)

```bash
# Downloads everything: DramaBox, Gemma, Antzoki LoRA
# Shows per-file progress bars. Set HF_TOKEN for faster downloads.
python ../scripts/download_models.py
```

Weight files land in `../models/`:

```
models/
├── dramabox-dit-v1.safetensors           # 6.6 GB
├── dramabox-audio-components.safetensors # 1.9 GB
├── gemma-3-12b-it-bnb-4bit/              # ~8 GB
└── antzoki-tts/
    ├── best_step_06850.safetensors       # 865 MB
    └── adapter_config.json
```

### 4. Run inference

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=ltx2 uv run python src/inference.py \
  --checkpoint      ../models/dramabox-dit-v1.safetensors \
  --full-checkpoint ../models/dramabox-audio-components.safetensors \
  --gemma-root      ../models/gemma-3-12b-it-bnb-4bit \
  --lora            ../models/antzoki-tts/best_step_06850.safetensors \
  --lora-rank       128 \
  --voice-sample    /path/to/reference.wav \
  --prompt          'A woman speaks warmly in Basque, "Kaixo, nola zaude?"' \
  --output          output.wav \
  --cfg-scale 2.5 --stg-scale 1.5 --no-watermark
```

---

## Prompt writing

Antzoki TTS inherits DramaBox's **director-script** prompt format: stage directions *outside* quotes, spoken text *inside* quotes. Directives are in English; dialogue is in Basque.

```
A seasoned radio host leans into the vintage microphone.
"Gaurkoan, bidai bat egingo dugu."
He pauses, letting the silence breathe.
"Bihotzaren taupadetatik, ametsen oihartzunetara."
```

Full prompt-writing guide and more examples: **[docs/inference-guide.md](docs/inference-guide.md)**.

---

## Model details

| | |
|---|---|
| **Base model** | DramaBox DiT v1 (3.3B params, distilled) |
| **Adapter** | LoRA (rank 128, alpha 128) |
| **Target modules** | Audio attention Q/K/V/output + FF layers across 48 transformer blocks |
| **Training data** | [OpenSLR76](https://www.openslr.org/76/) — 7,136 utterances, 52 speakers, ~14 h |
| **Training hardware** | NVIDIA GPU, 48 GB VRAM, ~6 hours |

Two checkpoints on HuggingFace:
- **`best_step_06850.safetensors`** — Best validation loss (recommended)
- **`lora_step_10000.safetensors`** — Final checkpoint

---

## Repository structure

```
antzoki-tts/
├── README.md                        # You are here
├── requirements-tts.txt               # Python dependencies (pip/uv)
├── DramaBox/                        # Git submodule (ResembleAI/DramaBox)
├── configs/
│   └── training_args_v2_openslr_only.yaml
├── docs/
│   ├── inference-guide.md           # Full inference walkthrough
│   └── training-config.md           # Training configuration reference
├── scripts/
│   ├── download_models.py            # One-shot weight downloader
│   └── eval_inference.py            # Batch evaluation across checkpoints
├── output/
│   ├── hf-upload/                   # Files uploaded to HuggingFace
│   │   ├── adapter_config.json
│   │   └── README.md
│   └── eval/                        # Example audio outputs
│       ├── shadowy_villain.mp3
│       ├── bright_eyed_girl.mp3
│       └── refs/                     # Voice reference clips used for demos
│           ├── dragoibola_demo.mp3
│           └── heidi_demo.mp3
└── .gitignore
```

---

## Performance

| Step | Time |
|---|---|
| Prompt encoding (Gemma 4-bit) | ~6 s |
| Model build + LoRA merge | ~5 s |
| Denoising (8 steps) | ~2 s |
| Audio decode (VAE + vocoder) | ~3 s |
| **Total (cold start)** | **~20–25 s** |
| **Peak GPU memory** | ~17 GB |

---

## Training

If you want to train your own LoRA on a different dataset, use DramaBox's training pipeline. The training config used for this LoRA is in `configs/training_args_v2_openslr_only.yaml` (see [docs/training-config.md](docs/training-config.md) for details).

```bash
cd DramaBox
PYTHONPATH=ltx2 uv run python src/train.py \
  --config ../configs/training_args_v2_openslr_only.yaml
```

---

## Acknowledgments

- **[DramaBox](https://github.com/resemble-ai/DramaBox)** — Resemble AI. The base TTS model.
- **[LTX-2](https://github.com/Lightricks/LTX-Video)** — Lightricks. The DiT architecture powering DramaBox.
- **[OpenSLR76](https://www.openslr.org/76/)** — Crowdsourced Basque speech dataset.
- **[Itzune](https://itzune.eus)** — Basque-language AI tools project.

---

## License

This repository and the LoRA adapter are released under **Apache 2.0**.  
DramaBox base model weights are subject to [Resemble AI's terms](https://github.com/resemble-ai/DramaBox/blob/main/LICENSE).
