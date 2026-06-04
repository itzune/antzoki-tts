# Training Configuration

## Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA L40S |
| VRAM | 46,068 MiB (~45 GB) |
| Architecture | Ada Lovelace |
| Driver | 580.142 → CUDA 13.0 |
| RAM | 125 GB |
| Mixed precision | bf16 (set by Accelerate) |

### VRAM Budget (training)

| Component | VRAM | Notes |
|---|---|---|
| DiT weights (bf16) | ~6.2 GB | `dramabox-dit-v1.safetensors` |
| Audio components (frozen, bf16) | ~1.9 GB | connector + VAE + vocoder |
| LoRA parameters (rank 128) | ~0.2 GB | 288 pairs × 2 matrices |
| Adam optimizer states | ~0.4 GB | 2× LoRA param count |
| Activations (grad checkpointing) | ~4–6 GB | re-computed per block |
| **Estimated total** | **~15–20 GB** | ~25 GB headroom on L40S |

### VRAM Budget (preprocessing — completed)

| Component | VRAM peak |
|---|---|
| Gemma-3-12B QAT unquantized (bf16) | ~24 GB |
| Audio VAE (after Gemma deleted) | ~1 GB |

---

## Training time estimate

**Expected range: 8–16 hours** for 10,000 steps on L40S.

The bottleneck is memory bandwidth (864 GB/s), not compute. At batch=1 the GPU
must stream 6.6 GB of weights per micro-step. With 4 grad-accum steps per
optimizer step and 10,000 optimizer steps = 40,000 micro-steps total.

The training script logs exact `steps/s` at step 50 — real ETA will be known
within the first 3 minutes of training.

If the estimate is too long, easy levers:
- `steps: 5000` — half the time, still effective for language-flavour adaptation
- `lora_rank: 64` — fewer params, faster + less memory
- `save_every: 250` — more checkpoints to A/B compare early

---

## `configs/training_args.yaml`

```yaml
# ── Data ───────────────────────────────────────────────────────────────────
data_dir:
  - /home/xezpeleta/dev/anttseztu/data/preprocessed/

speaker_index:
  - /home/xezpeleta/dev/anttseztu/data/speaker_index/basque_speaker_index.txt

output_dir: /home/xezpeleta/dev/anttseztu/output/basque-lora-v1/

# ── Base model ─────────────────────────────────────────────────────────────
checkpoint:      /home/xezpeleta/dev/DramaBox/dramabox-dit-v1.safetensors
full_checkpoint: /home/xezpeleta/dev/DramaBox/dramabox-audio-components.safetensors
base_model: dev    # ShiftedLogitNormal sampler — matches DramaBox (non-distilled)

# ── LoRA ───────────────────────────────────────────────────────────────────
lora_rank:    128
lora_alpha:   128
lora_dropout: 0.1  # regularisation for small dataset

# ── Voice-cloning reference ────────────────────────────────────────────────
ref_ratio:      0.3
max_ref_tokens: 200
text_dropout:   0.4

# ── Schedule ───────────────────────────────────────────────────────────────
steps:         10000
lr:            1.0e-04
lr_scheduler:  cosine
warmup_steps:  500
batch_size:    1
grad_accum:    4
max_grad_norm: 1.0
save_every:    500
log_every:      50
seed:           42

# ── Validation ─────────────────────────────────────────────────────────────
val_config: /home/xezpeleta/dev/anttseztu/configs/val_config.yaml
```

---

## Launch command

```bash
cd /home/xezpeleta/dev/DramaBox

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=ltx2 accelerate launch src/train.py \
    --config /home/xezpeleta/dev/anttseztu/configs/training_args.yaml
```

Single GPU — L40S has enough VRAM, no DDP needed.

---

## `configs/val_config.yaml`

Generates a sample WAV per entry at every `save_every` checkpoint.
Keep small (2–4 entries) so the subprocess finishes quickly.

```yaml
speakers:
  - name: female_neutral
    prompt: 'A woman speaks in Basque, "Kaixo, nola zaude gaur?"'
    reference: /home/xezpeleta/dev/anttseztu/data/val_refs/female_ref.wav

  - name: male_neutral
    prompt: 'A man speaks in Basque, "Gaur arratsaldean bilera bat dugu."'
    reference: /home/xezpeleta/dev/anttseztu/data/val_refs/male_ref.wav

  - name: female_no_ref
    prompt: 'A woman speaks in Basque, "Euskara hizkuntza eder eta aberatsa da."'
    # no reference — tests unconditional quality
```

> Pick reference WAVs from **held-out speakers** not in the training set.
> With 52 speakers, hold out 1 female + 1 male (e.g. `euf_00295`, `eum_04766`).

---

## Monitoring

### Log format (from `train.py`)
```
Step 50/10000 | loss=0.4231 | lr=2.00e-05 | tgt_T=175 ref_T=52 total=227 | 0.3 steps/s | ETA 420min
```

### Expected loss curve
- Steps 0–500 (warmup): ~1.0 → ~0.6
- Steps 500–3000: ~0.6 → ~0.3–0.4
- Steps 3000–10000: slow decay / plateau around 0.3

### Listening checkpoints
Suggested: **step 500, 1000, 2000, 5000, 10000**

### What to listen for
- [ ] English /r/, /æ/, /θ/ replaced by Basque equivalents
- [ ] Word-final consonant handling (Basque is syllable-timed)
- [ ] Voice cloning from reference still intact
- [ ] No increase in artifacts or unintelligibility
- [ ] Prosodic rhythm feels more Basque than English
