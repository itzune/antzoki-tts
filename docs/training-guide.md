# Training Guide

This document explains how to fine-tune DramaBox for a new language or voice
style using IC-LoRA (In-Context LoRA). The approach preserves DramaBox's
expressive prior while adapting it to new linguistic patterns.

## Overview

The Basque-language Antzoki TTS model was trained in a **single phase** on
OpenSLR data, 10,000 steps, using IC-LoRA rank 128.

The resulting LoRA weights are ~865 MB and plug into the base DramaBox model at
inference time — no need to re-download or modify the 6 GB base checkpoint.

---

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 24 GB VRAM | 40+ GB VRAM |
| System RAM | 16 GB | 32+ GB |
| Storage | 50 GB free | 100+ GB free |
| CUDA | 12.4+ | 12.6+ |

Training uses a single GPU (no DDP needed). A GPU with 24 GB VRAM is sufficient
for the training loop itself, but preprocessing requires loading an unquantized
Gemma-3-12B model (~24 GB VRAM peak).

### Software

```bash
# Install dependencies
uv pip install -r requirements-tts.txt

# Clone DramaBox submodule
git submodule update --init --recursive
```

### Base models

Download the required base models (see `scripts/download_models.py`):

```
models/
├── dramabox-dit-v1.safetensors          # ~6.2 GB
├── dramabox-audio-components.safetensors # ~1.9 GB
└── gemma-3-12b-it-bnb-4bit/             # Gemma 12B Q4 quantized
```

---

## Step 1: Prepare your dataset

### Format

DramaBox training expects **preprocessed** data — pre-encoded text embeddings
and audio latents. Use `src/preprocess.py` from DramaBox to convert raw
(audio, text) pairs.

The script supports four dataset formats:

| Format | Description | Example |
|---|---|---|
| `gemini_synthetic` | `~`-separated index with id~speaker~lang~sr~samples~dur~phonemes~text | Used for Basque training |
| `manifest` | JSONL with `audio_filepath`, `text` fields | Generic |
| `tsv` | TSV with `audio_path TAB text` | Simple two-column |
| `libriheavy` | `~`-separated like gemini, duration in ms | LibriHeavy format |

### Create a speaker index

The index file maps each sample to its speaker. Format (one line per sample):

```
<sample_id>~<speaker_id>~<lang>~<sr>~<samples>~<duration_sec>~<phonemes>~<text>
```

Example:
```
0~euf_01208~eu~16000~71360~4.46~_~AMPO kooperatibak balbulak egiten ditu Indian
1~euf_07508~eu~16000~66240~4.14~_~Gerard Piqueren etxea ikusteko aukera
```

The sample ID must match the filename (without extension) in the audio directory.

### Preprocessing command

```bash
cd DramaBox

PYTHONPATH=ltx2 python3 src/preprocess.py \
    --dataset-type gemini_synthetic \
    --index data/speaker_index/basque_speaker_index.txt \
    --audio-dir data/raw/ \
    --output-dir data/preprocessed/ \
    --max-samples 10000 \
    --max-duration 20.0 \
    --min-duration 2.0 \
    --gemma-root ../models/gemma-3-12b-it-bnb-4bit
```

This produces:
```
data/preprocessed/
├── latents/sample_000000.pt        # Dummy video latents (minimal, 1 frame)
├── conditions/sample_000000.pt     # Text embeddings from Gemma
└── audio_latents/sample_000000.pt  # Audio VAE-encoded latents
```

Preprocessing is the most VRAM-intensive step. Gemma-3-12B in bf16 requires
~24 GB. If you're VRAM-constrained, run preprocessing on a separate machine
and transfer the files.

### Dataset size guidelines

- **Good starting point**: 1,000–5,000 samples, 5–50 speakers
- **Balanced**: similar number of samples per speaker
- **Duration**: 2–20 seconds per clip (shorter clips → more training steps per epoch but less context)
- **Quality**: clean, dry recordings (minimal background noise, no music, no reverb)

The Basque model was trained on ~7,000 samples from 52 speakers (~50/50 male/female split),
sourced from OpenSLR's Basque dataset.

---

## Step 2: Configure training

Create a training config YAML. See `configs/training_args_v2_openslr_only.yaml`
for a reference configuration.

### Key parameters

```yaml
# ── Data ──
data_dir:
  - data/preprocessed/

speaker_index:
  - data/speaker_index/basque_speaker_index.txt

output_dir: output/my-lora-v1/

# ── Base model ──
checkpoint: models/dramabox-dit-v1.safetensors
full_checkpoint: models/dramabox-audio-components.safetensors
base_model: dev

# ── LoRA ──
lora_rank: 128        # Higher = more capacity (64-256 typical)
lora_alpha: 128       # Usually == rank (scale factor of 1.0)
lora_dropout: 0.1     # Regularization for small datasets

# ── Voice cloning reference ──
ref_ratio: 0.3        # Fraction of training samples with reference audio
max_ref_tokens: 200   # Max tokens for the appended voice reference
text_dropout: 0.4     # Probability of zeroing text (CFG training)

# ── Schedule ──
steps: 10000
lr: 1.0e-04
lr_scheduler: cosine
warmup_steps: 500
batch_size: 1          # Single-sample per micro-step
grad_accum: 4          # Effective batch size = 4

save_every: 500
log_every: 50
seed: 42

# ── Validation (optional) ──
# val_config: configs/val_config.yaml
```

### Learning rate and steps sizing

| Dataset size | Recommended steps | LR (cosine) |
|---|---|---|
| 500–1,000 clips | 2,000–3,000 | 5e-5 |
| 1,000–5,000 clips | 5,000–8,000 | 1e-4 |
| 5,000–10,000+ clips | 8,000–15,000 | 1e-4 |

Too many steps on a small dataset → overfitting (robotic monotone).
Too few steps on a large dataset → underfitting (English accent persists).

### Validation config (optional)

Create a validation config to generate test audio at each checkpoint:

```yaml
speakers:
  - name: female_neutral
    prompt: 'A woman speaks in Basque, "Kaixo, nola zaude gaur?"'
    reference: data/val_refs/female_ref.wav

  - name: male_neutral
    prompt: 'A man speaks in Basque, "Gaur arratsaldean bilera bat dugu."'
    reference: data/val_refs/male_ref.wav

  - name: unconditional_female
    prompt: 'A woman speaks in Basque, "Euskara hizkuntza eder eta aberatsa da."'
    # No reference — tests language quality without voice cloning
```

> Hold out reference speakers from training. With 52 speakers, reserve 1 female
> and 1 male for validation only.

---

## Step 3: Launch training

```bash
cd DramaBox

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=ltx2 accelerate launch src/train.py \
    --config ../configs/training_args.yaml
```

### Training time estimate

On an NVIDIA L40S (864 GB/s memory bandwidth, 46 GB VRAM):

- **10,000 steps** with batch=1, grad_accum=4: **~8–16 hours**
- Bottleneck is memory bandwidth, not compute
- First 3 minutes establish the actual speed (check `steps/s` in logs)

### Monitoring

The training script logs to stdout:

```
Step 50/10000 | loss=0.4231 | lr=2.00e-05 | tgt_T=175 ref_T=52 total=227 | 0.3 steps/s | ETA 420min
```

Expected loss curve:
- Steps 0–500 (warmup): 1.0 → ~0.6
- Steps 500–3,000: 0.6 → ~0.3–0.4
- Steps 3,000–10,000: slow decay / plateau near 0.3

If loss plateaus above 0.5 after warmup, reduce LR.
If loss spikes → NaN, the LR is too high — halve it and restart.

---

## Step 4: Evaluate checkpoints

Checkpoints are saved as `output/<name>/lora_step_<N>.safetensors`.

### Listening test

Generated samples at checkpoints: 500, 1000, 2000, 5000, 10000.

What to listen for:
- [ ] Language phonemes correct (e.g., Basque /r/ is tapped, not English /ɹ/)
- [ ] Natural prosody (syllable-timed rhythm for Basque)
- [ ] Voice cloning still works (reference voice preserved)
- [ ] No artifacts, garbling, or robotic quality
- [ ] No English accent bleeding through

### Running inference with a checkpoint

```bash
cd DramaBox

PYTHONPATH=ltx2 python3 src/inference.py \
    --checkpoint ../models/dramabox-dit-v1.safetensors \
    --full-checkpoint ../models/dramabox-audio-components.safetensors \
    --gemma-root ../models/gemma-3-12b-it-bnb-4bit \
    --lora ../output/my-lora-v1/lora_step_05000.safetensors \
    --lora-rank 128 \
    --voice-sample ../output/eval/refs/my_voice_ref.mp3 \
    --prompt 'A speaker in my language, "Test sentence."' \
    --output test_lora.wav \
    --cfg-scale 2.5 --stg-scale 1.5
```

### Choosing the best checkpoint

Don't just pick the last one. Earlier checkpoints often sound better:

- **Lowest loss ≠ best audio**. Listen to them.
- Step 2,000–5,000 often has the best balance of adaptation and naturalness.
- If voice cloning degrades at later steps, you're overfitting — use an earlier checkpoint.

---

## Step 5: Export for inference

The LoRA weights in `lora_step_<N>.safetensors` are all you need.
Copy the best checkpoint to your inference setup:

```bash
cp output/my-lora-v1/lora_step_05000.safetensors models/my_language_lora.safetensors
```

No need to merge — DramaBox applies LoRA additively at inference time.

---

## Troubleshooting

### CUDA out of memory during training

- Reduce `lora_rank` (128 → 64)
- Enable gradient checkpointing (already on by default in DramaBox)
- Remove validation (`val_config` disabled)

### Training crashed during preprocessing

- Gemma-3-12B in bf16 needs ~24 GB. Use `--gemma-root` pointing to a 4-bit quantized version
- Split preprocessing into shards: `--shard 0 --num-shards 4` on 4 GPUs

### Loss not decreasing

- Ensure audio files are actually found (check preprocessing logs for skipped files)
- Check that audio formats are supported (.flac, .wav, .mp3)
- Try higher LR (2e-4) or add `--lr-scheduler constant` in early phase

### Generated audio sounds robotic

- Likely overfitting — use an earlier checkpoint
- Reduce `lora_rank`, increase `lora_dropout`
- Fewer training steps, lower LR

---

## Reference: Config used for Basque TTS

The Basque model was trained with a single-phase approach:

| Parameter | Value |
|---|---|
| Dataset | OpenSLR Basque (7,136 samples, 52 speakers) |
| Steps | 10,000 |
| LoRA rank | 128 |
| LoRA alpha | 128 |
| Learning rate | 1e-4 (cosine schedule) |
| Batch size | 1 (grad_accum=4) |
| Text dropout | 0.4 |
| Ref ratio | 0.3 |
| GPU | NVIDIA L40S (46 GB VRAM) |
| Training time | ~9.5 hours |
| Final checkpoint | Step 10,000 (865 MB) |

See `configs/training_args_v2_openslr_only.yaml` for the full reference config.
