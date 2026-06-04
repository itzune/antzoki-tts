# Antzoki TTS — Inference Guide

> **Antzoki** *(Basque)* — theatre, stage.  
> Expressive Basque text-to-speech with voice cloning, powered by a LoRA fine-tune of [DramaBox](https://github.com/resemble-ai/DramaBox).

This guide walks you through running Antzoki TTS from scratch: installing dependencies, downloading model weights, and generating expressive Basque speech with voice cloning and director-style prompts.

---

## How it works

Antzoki TTS is a **LoRA adapter** (rank 128) trained on top of **DramaBox** (Resemble AI's expressive TTS, itself built on the **LTX-2.3** 3.3B audio model). The LoRA was fine-tuned on the [OpenSLR76](https://www.openslr.org/76/) Basque speech corpus (~14 hours, 52 speakers) to shift the phonetic prior from English toward Basque, while preserving DramaBox's dramatic delivery capabilities.

Two inputs control the output:

1. **Voice reference** (*optional*) — A ~10-second WAV/MP3 audio clip. The model clones the speaker's timbre.
2. **Director-style prompt** — A mix of narrative direction (character, emotion, actions) and Basque spoken text inside quotes.

```
A shadowy villain speaks with cold menace, "Nire lurretan sartu zara, morroi"
He chuckles darkly, "Erruz ordainduko duzu."
His voice rises with fury, "Belaunikatu, edo suntsituko zaitut!!"
```

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | ~17 GB | 24+ GB |
| **System RAM** | 16 GB | 32 GB |
| **Disk** | ~20 GB (models) | 30 GB |
| **CUDA** | 12.x+ | 12.8+ |

Inference takes ~20–25 seconds for a 10–20 second output on a modern GPU. Peak GPU memory usage during inference is ~17 GB.

---

## Step 1: Clone DramaBox

```bash
git clone https://github.com/resemble-ai/DramaBox.git
cd DramaBox
```

---

## Step 2: Set up Python environment with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager written in Rust. It replaces pip, venv, and virtualenv with a single tool.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environment and install dependencies

```bash
# Create a virtual environment with Python 3.10+
uv venv --python 3.10
source .venv/bin/activate

# One command — the requirements file includes the CUDA index URL
uv pip install -r ../requirements-tts.txt
```

That's it. `requirements-tts.txt` bundles every package needed for inference (torch, safetensors, accelerate, peft, av, transformers, huggingface_hub, bitsandbytes, soundfile, etc.) and sets the CUDA wheel index via a `--index-url` line at the top of the file. Optional web-demo and watermarking dependencies are skipped.

> **Note**: `resemble-perth` (audio watermarking) is optional. Pass `--no-watermark` at inference time.

---

## Step 3: Download model weights

Use the bundled download script — it fetches everything with per-file progress bars:

```bash
# From the repo root (outside DramaBox/)
python scripts/download_models.py

# Or specify a custom output directory
python scripts/download_models.py --output /path/to/models

# Set HF_TOKEN for authenticated (faster) downloads
HF_TOKEN=hf_xxx python scripts/download_models.py
```

Alternatively, you can download manually from HuggingFace:

```python
from huggingface_hub import hf_hub_download, snapshot_download

# DramaBox base models (~8.5 GB)
hf_hub_download("ResembleAI/Dramabox", "dramabox-dit-v1.safetensors",
                local_dir="models")
hf_hub_download("ResembleAI/Dramabox", "dramabox-audio-components.safetensors",
                local_dir="models")

# Gemma text encoder (~8 GB)
snapshot_download("unsloth/gemma-3-12b-it-bnb-4bit",
                  local_dir="models/gemma-3-12b-it-bnb-4bit")

# Antzoki Basque LoRA (~865 MB)
hf_hub_download("itzune/antzoki-tts", "best_step_06850.safetensors",
                local_dir="models/antzoki-tts")
hf_hub_download("itzune/antzoki-tts", "adapter_config.json",
                local_dir="models/antzoki-tts")
```

Two LoRA checkpoints are available on HuggingFace:
- **`best_step_06850.safetensors`** — Best validation loss (recommended)
- **`lora_step_10000.safetensors`** — Final checkpoint

### Final model layout

```
models/
├── dramabox-dit-v1.safetensors           # 6.6 GB
├── dramabox-audio-components.safetensors # 1.9 GB
├── gemma-3-12b-it-bnb-4bit/              # ~8 GB
└── antzoki-tts/
    ├── best_step_06850.safetensors       # 865 MB
    ├── lora_step_10000.safetensors       # 865 MB
    └── adapter_config.json
```

---

## Step 4: Run inference

### Basic command

Activate the environment and run from the DramaBox directory:

```bash
cd DramaBox
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=ltx2 uv run python src/inference.py \
  --checkpoint      ../models/dramabox-dit-v1.safetensors \
  --full-checkpoint ../models/dramabox-audio-components.safetensors \
  --gemma-root      ../models/gemma-3-12b-it-bnb-4bit \
  --lora            ../models/antzoki-tts/best_step_06850.safetensors \
  --lora-rank       128 \
  --voice-sample    /path/to/reference.wav \
  --prompt          'Your prompt here' \
  --output          output.wav \
  --cfg-scale       2.5 \
  --stg-scale       1.5 \
  --no-watermark
```

### Key parameters

| Flag | Description | Default |
|---|---|---|
| `--lora` | Path to LoRA `.safetensors` | *required* |
| `--lora-rank` | LoRA rank (must match training: 128) | 128 |
| `--voice-sample` | Reference audio for voice cloning | *none* |
| `--no-ref` | Skip voice reference (unconditional TTS) | — |
| `--prompt` | Director-script prompt | *required* |
| `--cfg-scale` | Classifier-free guidance strength | 2.5 |
| `--stg-scale` | Skip-token guidance | 1.5 |
| `--gen-duration` | Target seconds (auto-detected if 0) | 0 (auto) |
| `--seed` | Random seed | 42 |
| `--no-watermark` | Skip Perth audio watermarking | — |

---

## Step 5: Prompt writing guide

DramaBox prompts follow a **director-script** format: stage directions *outside* quotes, spoken dialogue *inside* quotes. The prompt language for directions should be English (the text encoder is English-native), but the spoken text in quotes is in Basque.

### Structure

```
[character description], [emotion / action]. "[Basque dialogue]"
[more direction]. "[more Basque dialogue]"
```

### What goes *inside* quotes (the model speaks these)

- Basque dialogue text
- Laughs: `"Hahaha"`, `"Hehehe"`
- Non-verbal vocalizations: `"Mmmmm"`, `"Ugh"`, `"Ahhh"`, `"Hmm"`

### What goes *outside* quotes (stage directions — the model acts these)

- Character description: `A young woman with a soft voice…`
- Emotions: `She speaks with excitement…`, `His voice cracks with emotion…`
- Actions: `She sighs deeply.`, `He pauses for a long moment.`, `She leans into the microphone.`
- Breathing: `Her breath hitches.`, `He takes a shaky breath.`

### What to *avoid* inside quotes

The model will *speak* these literally if placed inside quotes:
`Ahem`, `Pfft`, `Sigh`, `Gasp`, `Cough`, `Chuckle` — use them outside quotes instead.

### Good prompt examples

#### Dramatic villain (with voice cloning)
```
A shadowy villain speaks with cold menace, "Nire lurretan sartu zara, morroi"
He chuckles darkly, "Erruz ordainduko duzu."
His voice rises with fury, "Belaunikatu, edo suntsituko zaitut!!"
```

#### Radio/podcast host (with voice cloning)
```
A seasoned radio host with a deep, resonant voice leans into the vintage microphone,
the warm glow of the studio lights catching his eyes.
He speaks with the gravitas of someone who has told stories for decades.
"Gaurkoan, bidai bat egingo dugu. Ez espazioan, ezta denboran ere."
He pauses, letting the silence breathe.
"Bihotzaren taupadetatik, ametsen oihartzunetara."
His voice swells with passion, filling every corner of the booth.
"Euskal Herriko ahotsek badute zerbait berezia. Entzun... eta sentitu."
He leans back in his chair, a gentle smile spreading across his face.
"Hau ez da irrati saio bat. Hau poema bat da, uhinetan barrena."
```

#### Joyful child (with voice cloning)
```
A bright-eyed girl spins in a field of wildflowers, her voice bubbling with pure, breathless wonder:
"Aizu, aitona! Entzun duzu?!"
She laughs, a sound as clear as a mountain stream.
"Makina batek hitz egiten duela dirudi, baina hain da erreala!"
She spreads her arms wide, looking up at the sky in disbelief.
"Sinestezina da… adimen artifizialak nire ahotsa sortu du!!"
```

#### Documentary narrator (with voice cloning)
```
A professional woman in her mid-thirties with a warm, rhythmic storyteller's voice
speaks with clear authority and growing excitement.
She leans into the microphone, her breath audible.
"Kaixo guztioi! Gaur denboran atzera egingo dugu, duela hirurogeita sei milioi urteko mundu harrigarri hartara."
She pauses for a moment, letting the tension build, then speaks with dramatic intensity.
"Bat-batean, zerua argitu zen. Asteroide erraldoi batek Lurra jo zuen eta dinosauroen erregealdia betiko amaitu zen!"
She chuckles softly, a smile evident in her tone.
"Nola aldatu zuen kolpe hark planetaren patua? Segituan kontatuko dizuegu!"
```

#### Simple neutral (no voice cloning)
```
A woman speaks warmly in Basque, "Kaixo, nola zaude? Gaur eguraldi ederra dago."
```

#### Emotional letter reading (with voice cloning)
```
A young girl with a soft, youthful voice reads aloud from a worn piece of paper.
Her voice is clear but heavy with a quiet, solemn sadness.
"Askotan oroitzen naiz zutaz, ama."
She pauses, swallowing hard, taking a quiet breath.
"Hemen, gerrako maniobretan ari gara jo eta su."
Her voice drops slightly, carrying heavy grief.
"Iduri du izar eta txori guztiak hilak daudela jadanik..."
```

---

## Performance

Measured on a GPU with 48 GB VRAM:

| Step | Time |
|---|---|
| Prompt encoding (Gemma 4-bit) | ~6 s |
| Model build + LoRA merge | ~5 s |
| Denoising (8 steps, distilled) | ~2 s |
| Audio decode (VAE + vocoder) | ~3 s |
| **Total (cold start)** | **~20–25 s** |

**Peak GPU memory**: ~17 GB during inference (including Gemma 7.8 GB + DiT model + LoRA + audio VAE).

For repeated generations, use DramaBox's `inference_server.py` (warm server) to skip the 5s model build time on subsequent calls.

---

## Tips

- **Match gender/age** in the prompt description to your voice reference for best results.
- **Voice references** should be ~10 seconds, clean speech without background noise. Peak-normalized to -4 dBFS automatically.
- **For longer audio** (>20s), use `--gen-duration` to set an explicit target duration instead of relying on auto-estimation.
- **If the output clips** (too loud), lower `--cfg-scale` to 1.5–2.0. Higher CFG values increase volume and can cause clipping.
- **For cleaner starts**, the `--pad-start` flag prepends silent padding that gets trimmed after decoding.
- **`best_step_06850`** is recommended over `lora_step_10000` for most use cases — it has better expressive range while retaining Basque prosody.

---

## Limitations

- Trained on read speech only (OpenSLR76). Expressive output relies on DramaBox's pretrained prior.
- Residual English prosody may appear in some phonetic contexts. Best results with voice cloning from a Basque speaker.
- Very short prompts (<3s target duration) may produce less stable output.
- The model is distilled (8 diffusion steps). Quality vs speed trade-off is acceptable for most use cases.

---

## References

| Resource | URL |
|---|---|
| Antzoki TTS (HF) | https://huggingface.co/itzune/antzoki-tts |
| DramaBox (GitHub) | https://github.com/resemble-ai/DramaBox |
| DramaBox (HF) | https://huggingface.co/ResembleAI/Dramabox |
| LTX-2 | https://github.com/Lightricks/LTX-2 |
| OpenSLR76 | https://www.openslr.org/76/ |
| Blog post (Basque) | https://xezpeleta.github.io/blog/antzoki-tts/ |
