#!/usr/bin/env python3
"""
eval_inference.py — Generate evaluation WAVs for both reference voices.

Usage:
    cd DramaBox && source .venv/bin/activate
    PYTHONPATH=ltx2 python ../scripts/eval_inference.py --lora ../models/antzoki-tts/best_step_06850.safetensors

Output files:
    output/eval/dragoibola_<step>.wav
    output/eval/heidi_pozik_<step>.wav
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ── Evaluation set ─────────────────────────────────────────────────────────────

EVALS = [
    {
        "name": "dragoibola",
        "voice_sample": None,  # Provide your own reference audio
        "prompt": (
            'A shadowy villain speaks with cold menace, "Nire lurretan sartu zara, morroi" '
            'He chuckles darkly, "Erruz ordainduko duzu." '
            'His voice rises with fury, "Belaunikatu, edo suntsituko zaitut!!"'
        ),
    },
    {
        "name": "radio_dinosaur",
        "voice_sample": None,
        "prompt": (
            'A professional woman in her mid-thirties with a warm, rhythmic storyteller\'s voice speaks with clear authority and growing excitement. '
            'She leans into the microphone, her breath audible. '
            '"Kaixo guztioi! Gaur denboran atzera egingo dugu, duela hirurogeita sei milioi urteko mundu harrigarri hartara." '
            'She pauses for a moment, letting the tension build, then speaks with dramatic intensity. '
            '"Bat-batean, zerua argitu zen. Asteroide erraldoi batek Lurra jo zuen eta dinosauroen erregealdia betiko amaitu zen!" '
            'She chuckles softly, a smile evident in her tone. '
            '"Nola aldatu zuen kolpe hark planetaren patua? Segituan kontatuko dizuegu!"'
        ),
    },
    {
        "name": "radio_dinosaur_uhin",
        "voice_sample": None,  # Provide your own reference audio
        "prompt": (
            'A professional woman in her mid-thirties with a warm, rhythmic storyteller\'s voice speaks with clear authority and growing excitement. '
            'She leans into the microphone, her breath audible. '
            '"Kaixo lagun! Gaur denboran atzera egingo dugu, duela hirurogeita sei milioi urteko mundu harrigarri hartara." '
            'Mistery. Speaks with dramatic intensity. '
            '"Bat-batean, zerua argitu zen. Asteroide erraldoi batek Lurra jo zuen eta di-no-sau-ro-en erregealdia betiko amaitu zen!" '
            'She asks like rethorical question:'
            '"Nola aldatu zuen kolpe hark planetaren patua? Segituan kontatuko dizuegu!"'
        ),
    },
    {
        "name": "bizkarsoro_irakurketa",
        "voice_sample": None,  # Provide your own reference audio
        "prompt": (
            'A young girl with a soft, youthful Northern Basque accent reads aloud from a worn piece of paper. '
            'Her voice is clear but heavy with a quiet, solemn sadness. '
            'She sits on a wooden bench outside a rural house, surrounded by an anxious, silent family listening to every word. '
            '"Askotan oroitzen naiz zutaz, ama." '
            'She pauses, swallowing hard, taking a quiet breath before continuing the soldier\'s update. '
            '"Hemen, gerrako maniobretan ari gara jo eta su." '
            'Her voice drops slightly, carrying the heavy, poetic grief of the letter\'s words. '
            '"Iduri du izar eta txori guztiak hilak daudela jadanik... eta sute eta zapartek, berriz, bakarrik utziak gaituztela ilunpetan." '
            'She pauses for a long moment, letting the darkness of the imagery hang in the air, then offers a small, fragile note of hope. '
            '"Agian eguberriz joaten ahalko naiz etxera." '
            'Her tone softens into an intimate, longing plea. '
            '"Igor zaizkidazu gaztainak, otoi." '
            'She sighs softly, closing the letter with a gentle, fading whisper. '
            '"Ikus arte, Mizel."'
        ),
    },
    {
        "name": "basajaun_liburua",
        "voice_sample": None,  # Provide your own reference audio
        "prompt": (
            'A middle-aged woman with a velvety, soothing voice and a gentle melodic lilt. '
            'She speaks softly, as if the listener is just inches away, her breath warm and calm. '
            '"Behin batean, mendeak eta mendeak atzera, gailurrik altuenetan bizi zen izaki erraldoi bat..." '
            'She pauses, her breath hitching slightly in wonder. '
            '"...Basajaun zuen izena." '
            'She chuckles softly, a warm and comforting sound. '
            '"Mmmmm, imajinatu dezakezu? Haren oinatsek mendi osoa mugiarazten zuten, baina beldurrik ez izateko modukoa zen, bihotz-bihotzez gozoa baitzen." '
            'She sighs contentedly, her voice trailing off into a whisper. '
            '"Lo egin orain, txiki... bihar abentura berriak izango ditugu eta."'
        ),
    },
    {
        "name": "heidi_pozik",
        "voice_sample": None,  # Provide your own reference audio
        "prompt": (
            'A bright-eyed girl spins in a field of wildflowers, her voice bubbling with pure, breathless wonder: '
            '"Aizu, aitona! Entzun duzu?!" '
            'She laughs, a sound as clear as a mountain stream. '
            '"Makina batek hitz egiten duela dirudi, baina hain da erreala!" '
            'She spreads her arms wide, looking up at the sky in disbelief. '
            '"Sinestezina da... adimen artifizialak nire ahotsa sortu du!!"'
        ),
    },
]

# ── Paths (relative to repo root, resolved at runtime) ─────────────────────────

REPO_ROOT     = Path(__file__).resolve().parent.parent
# DramaBox is a git submodule at repo_root/DramaBox
DRAMABOX_DIR   = REPO_ROOT / "DramaBox"
MODELS_DIR     = REPO_ROOT / "models"
CHECKPOINT     = MODELS_DIR / "dramabox-dit-v1.safetensors"
FULL_CKPT      = MODELS_DIR / "dramabox-audio-components.safetensors"
GEMMA_ROOT     = MODELS_DIR / "gemma-3-12b-it-bnb-4bit"
LORA_DIR       = MODELS_DIR / "antzoki-tts"
INFERENCE_PY   = DRAMABOX_DIR / "src" / "inference.py"
OUTPUT_DIR     = REPO_ROOT / "output" / "eval"


def parse_step(lora_path: Path) -> str:
    """Extract step label from checkpoint filename.

    Supports:
      lora_step_08000.safetensors  -> step08000
      best_step_04000.safetensors  -> best04000
      adapter_model.safetensors    -> adapter
      anything_else.safetensors    -> the stem as-is
    """
    stem = lora_path.stem
    m = re.search(r"(\d+)$", stem)
    if "best" in stem and m:
        return f"best{int(m.group(1)):05d}"
    if m:
        return f"step{int(m.group(1)):05d}"
    return stem


def main():
    parser = argparse.ArgumentParser(description="Run evaluation inference for both reference voices.")
    parser.add_argument("--lora", required=True, help="Path to LoRA .safetensors checkpoint")
    parser.add_argument("--eval", nargs="+", metavar="NAME",
                        help="Run only these eval(s) by name (e.g. --eval dragoibola radio_dinosaur). "
                             f"Available: {', '.join(e['name'] for e in EVALS)}")
    parser.add_argument("--cfg-scale",  type=float, default=2.5)
    parser.add_argument("--stg-scale",  type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path,  default=OUTPUT_DIR)
    args = parser.parse_args()

    lora_path = Path(args.lora).resolve()
    if not lora_path.exists():
        sys.exit(f"ERROR: LoRA checkpoint not found: {lora_path}")

    step_label = parse_step(lora_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter evals if --eval was specified
    evals = EVALS
    if args.eval:
        names = set(args.eval)
        unknown = names - {e["name"] for e in EVALS}
        if unknown:
            sys.exit(f"ERROR: unknown eval name(s): {', '.join(sorted(unknown))}\n"
                     f"Available: {', '.join(e['name'] for e in EVALS)}")
        evals = [e for e in EVALS if e["name"] in names]
    print(f"LoRA:  {lora_path}")
    print(f"Label: {step_label}")
    print(f"Out:   {output_dir}")
    print()

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(DRAMABOX_DIR / "ltx2")}

    for eval_cfg in evals:
        voice = (REPO_ROOT / eval_cfg["voice_sample"]).resolve() if eval_cfg["voice_sample"] else None
        output = output_dir / f"{eval_cfg['name']}_{step_label}.wav"

        print(f"── {eval_cfg['name']} ──────────────────────────────────────────")
        print(f"   ref:    {voice if voice else '(none — no voice cloning)'}")
        print(f"   output: {output}")

        cmd = [
            sys.executable, str(INFERENCE_PY),
            "--checkpoint",      str(CHECKPOINT),
            "--full-checkpoint", str(FULL_CKPT),
            "--gemma-root",      str(GEMMA_ROOT),
            "--lora",            str(lora_path),
            "--lora-rank",       "128",
            "--prompt",          eval_cfg["prompt"],
            "--output",          str(output),
            "--cfg-scale",       str(args.cfg_scale),
            "--stg-scale",       str(args.stg_scale),
            "--no-watermark",
        ]
        if voice is not None:
            cmd += ["--voice-sample", str(voice)]

        result = subprocess.run(cmd, cwd=str(DRAMABOX_DIR), env=env)
        if result.returncode != 0:
            print(f"   ERROR: inference failed (exit {result.returncode})")
        else:
            size_kb = output.stat().st_size // 1024
            print(f"   OK — {size_kb} KB\n")


if __name__ == "__main__":
    main()
