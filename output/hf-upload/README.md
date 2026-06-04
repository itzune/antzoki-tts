---
language:
  - eu
license: apache-2.0
tags:
  - text-to-speech
  - tts
  - lora
  - basque
  - euskera
  - voice-cloning
  - speech-synthesis
  - dramabox
  - ltx2
  - expressive-tts
base_model: ResembleAI/DramaBox
datasets:
  - openslr/openslr76
metrics: []
pipeline_tag: text-to-speech
library_name: peft
---

# Antzoki TTS — Basque LoRA for DramaBox

**Antzoki TTS** is a LoRA adapter for [DramaBox](https://github.com/resemble-ai/DramaBox) (Resemble AI), fine-tuned on the [OpenSLR76](https://www.openslr.org/76/) Basque speech corpus to improve Basque-language synthesis quality.

> **Antzoki** (Basque) — *theatre*, *stage*

The base DramaBox model is a highly expressive, cinematic TTS system capable of voice cloning, dramatic acting, and detailed emotional direction. This LoRA shifts its phonetic prior toward Basque, reducing the English accent while preserving dramatic and expressive capabilities.

---

## Model Details

| | |
|---|---|
| **Base model** | DramaBox DiT v1 (`dev` schedule, non-distilled) |
| **Adapter type** | LoRA (PEFT) |
| **LoRA rank** | 128 |
| **LoRA alpha** | 128 |
| **Target modules** | `audio_attn1.{to_q,to_k,to_v,to_out.0}`, `audio_ff.{net.0.proj,net.2}` — 288 weight pairs across 48 transformer blocks |
| **Training steps** | 10 000 |
| **Learning rate** | 1e-4 (cosine schedule) |
| **Dataset** | OpenSLR76 — 7 136 utterances, 52 speakers (29 F / 23 M), ~13.9 h total audio |
| **Hardware** | NVIDIA L40S (46 GB VRAM) |
| **Training time** | ~6 hours |

### Checkpoints included

| File | Description |
|---|---|
| `lora_step_10000.safetensors` | Final checkpoint (step 10 000) |
| `best_step_06850.safetensors` | Best validation loss checkpoint (step 6 850) |
| `adapter_config.json` | PEFT adapter configuration |

> **Recommended**: `best_step_06850.safetensors` for best balance of Basque prosody and expressive acting range. `lora_step_10000.safetensors` may offer better Basque phonetics at the cost of some expressiveness.

---

## Usage

Requires [DramaBox](https://github.com/resemble-ai/DramaBox) to be set up locally.

```bash
cd DramaBox

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=ltx2 python src/inference.py \
  --checkpoint      dramabox-dit-v1.safetensors \
  --full-checkpoint dramabox-audio-components.safetensors \
  --lora            /path/to/best_step_06850.safetensors \
  --voice-sample    /path/to/reference.wav \
  --prompt          "Your director-style prompt here" \
  --output          output.wav \
  --cfg-scale 2.5 \
  --stg-scale 1.5
```

> The LoRA is **never merged** — always loaded via `--lora` at inference time.

---

## Prompt Format

DramaBox uses a **director-style prompt** format: narrative context outside quotes, spoken text inside quotes.

```
A [character description], [action/emotion]. "[spoken text]"
```

### Example prompts

**Villain — dramatic menace (voice clone)**
```
A shadowy villain speaks with cold menace, "Nire lurretan sartu zara, morroi"
He chuckles darkly, "Erruz ordainduko duzu."
His voice rises with fury, "Belaunikatu, edo suntsituko zaitut!!"
```

**Documentary narrator — radio host (no voice clone)**
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

**Joyful child — wonder and excitement (voice clone)**
```
A bright-eyed girl spins in a field of wildflowers, her voice bubbling with pure, breathless wonder:
"Aizu, aitona! Entzun duzu?!"
She laughs, a sound as clear as a mountain stream.
"Makina batek hitz egiten duela dirudi, baina hain da erreala!"
She spreads her arms wide, looking up at the sky in disbelief.
"Sinestezina da... adimen artifizialak nire ahotsa sortu du!!"
```

**Neutral Basque (simple wrapper)**
```
A woman speaks in Basque, "Kaixo, nola zaude gaur?"
```

---

## Limitations

- Trained exclusively on read speech (OpenSLR76). Expressive/dramatic output relies on DramaBox's pretrained prior.
- Accent reduction is significant but not complete — residual English prosody may appear in some phonetic contexts.
- Best results with voice cloning (`--voice-sample`) from a Basque speaker.
- Very short prompts (<3 s target duration) may produce less stable output.

---

## Training Data

[OpenSLR76](https://www.openslr.org/76/) — Crowdsourced Basque speech corpus:
- 7 136 utterances across 52 speakers (29 female, 23 male)
- 3–15.5 s per clip, mean ~7 s, ~13.9 h total
- Read speech style

---

## Acknowledgements

- **[DramaBox](https://github.com/resemble-ai/DramaBox)** — Resemble AI. The base TTS model this LoRA is trained on. DramaBox is built on the LTX-2 architecture.
- **[LTX-2](https://github.com/Lightricks/LTX-Video)** — Lightricks. The underlying DiT architecture powering DramaBox.
- **[OpenSLR76](https://www.openslr.org/76/)** — Crowdsourced Basque speech dataset used for fine-tuning.

---

## License

This LoRA adapter is released under **Apache 2.0**.  
DramaBox base model weights are subject to [Resemble AI's terms](https://github.com/resemble-ai/DramaBox/blob/main/LICENSE).

---

*Part of the [Itzune](https://huggingface.co/itzune) project — Basque-language AI tools.*
