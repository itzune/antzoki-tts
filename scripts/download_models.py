#!/usr/bin/env python3
"""
Download all model weights needed for Antzoki TTS inference.

Downloads (~20 GB total):
  1. DramaBox DiT transformer      — 6.6 GB  (ResembleAI/Dramabox)
  2. DramaBox audio components     — 1.9 GB  (ResembleAI/Dramabox)
  3. Gemma 3 12B IT (bnb-4bit)    — 7.8 GB  (unsloth/gemma-3-12b-it-bnb-4bit)
  4. Antzoki Basque LoRA adapter   — 0.9 GB  (itzune/antzoki-tts)

Usage:
    python scripts/download_models.py                       # download to ./models/
    python scripts/download_models.py --output /path/to/models
    HF_TOKEN=hf_xxx python scripts/download_models.py       # authenticated (faster)

The script shows per-file progress bars via huggingface_hub's built-in tqdm.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


# ── Repo / file definitions ──────────────────────────────────────────────────

DRAMABOX_REPO = "ResembleAI/Dramabox"
GEMMA_REPO    = "unsloth/gemma-3-12b-it-bnb-4bit"
ANTZOKI_REPO  = "itzune/antzoki-tts"

# Each entry: (repo_id, filename_or_None, local_subdir, label, expected_size_gb)
DOWNLOADS = [
    # DramaBox DiT transformer
    (DRAMABOX_REPO, "dramabox-dit-v1.safetensors",
     None, "DramaBox DiT (transformer)", 6.6),

    # DramaBox audio VAE + vocoder + connectors
    (DRAMABOX_REPO, "dramabox-audio-components.safetensors",
     None, "DramaBox audio components (VAE + vocoder)", 1.9),

    # Gemma text encoder (whole snapshot)
    (GEMMA_REPO, None,
     "gemma-3-12b-it-bnb-4bit", "Gemma 3 12B IT (bnb-4bit)", 7.8),

    # Antzoki LoRA weights
    (ANTZOKI_REPO, "best_step_06850.safetensors",
     "antzoki-tts", "Antzoki LoRA (best_step_06850)", 0.87),

    # Antzoki adapter config
    (ANTZOKI_REPO, "adapter_config.json",
     "antzoki-tts", "Antzoki adapter config", 0.003),
]


def print_header(text: str) -> None:
    width = 70
    print(f"\n{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}")


def download_file(repo_id: str, filename: str, output_dir: Path,
                  label: str, expected_gb: float) -> Path:
    """Download a single file from HF Hub with progress bar."""
    print(f"\n  📦 {label}  ({expected_gb:.1f} GB)")
    print(f"     {repo_id}/{filename}")

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        actual_size = Path(local).stat().st_size / 1e9
        print(f"     ✅ {actual_size:.2f} GB  →  {local}")
        return Path(local)
    except HfHubHTTPError as e:
        print(f"     ❌  Failed: {e}")
        raise


def download_snapshot(repo_id: str, output_dir: Path, subdir: str,
                      label: str, expected_gb: float) -> Path:
    """Download a full HF repo snapshot with progress bar."""
    target = output_dir / subdir
    print(f"\n  📦 {label}  (~{expected_gb:.1f} GB)")
    print(f"     {repo_id}")

    try:
        local = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            token=os.environ.get("HF_TOKEN"),
        )
        # Calculate total size
        total_bytes = sum(f.stat().st_size for f in Path(local).rglob("*") if f.is_file())
        total_gb = total_bytes / 1e9
        print(f"     ✅ {total_gb:.2f} GB  →  {local}")
        return Path(local)
    except HfHubHTTPError as e:
        print(f"     ❌  Failed: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all model weights for Antzoki TTS inference"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: ./models/ relative to repo root)",
    )
    args = parser.parse_args()

    # Resolve output dir: defaults to <repo-root>/models/
    if args.output is None:
        repo_root = Path(__file__).resolve().parent.parent
        output_dir = repo_root / "models"
    else:
        output_dir = args.output.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Welcome ──────────────────────────────────────────────────────────────

    total_gb = sum(d[4] for d in DOWNLOADS)
    print("=" * 70)
    print("  Antzoki TTS — Model Weight Downloader")
    print("=" * 70)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Total download:   ~{total_gb:.1f} GB (5 files)")
    if os.environ.get("HF_TOKEN"):
        print("  HF_TOKEN:          set ✅ (authenticated)")
    else:
        print("  HF_TOKEN:          not set ⚠️  (unauthenticated, may be rate-limited)")

    # ── Download ─────────────────────────────────────────────────────────────

    print_header("Downloading model weights")

    total_start = os.times().elapsed if hasattr(os, 'times') else 0

    success = 0
    failed = 0

    for repo_id, filename, subdir, label, expected_gb in DOWNLOADS:
        try:
            if filename is not None:
                download_file(repo_id, filename, output_dir, label, expected_gb)
            else:
                download_snapshot(repo_id, output_dir, subdir, label, expected_gb)
            success += 1
        except Exception:
            failed += 1
            continue

    # ── Summary ──────────────────────────────────────────────────────────────

    print_header("Download complete")

    # Calculate total size on disk
    total_bytes = 0
    for f in output_dir.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size

    print(f"  Downloaded:  {success}/{len(DOWNLOADS)}  ({failed} failed)")
    print(f"  Total size:  {total_bytes / 1e9:.2f} GB")
    print(f"  Location:    {output_dir.resolve()}")
    print()

    # ── Final directory layout ───────────────────────────────────────────────

    print("  Final layout:")
    print(f"  {output_dir}/")
    tree_lines = []
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(output_dir)
            size_mb = f.stat().st_size / 1e6
            tree_lines.append(f"    ├─ {rel}  ({size_mb:.0f} MB)" if size_mb >= 1
                              else f"    ├─ {rel}  ({size_mb*1000:.0f} KB)")

    # Show top-level dirs
    shown = set()
    for f in sorted(output_dir.iterdir()):
        if f.is_dir():
            count = sum(1 for _ in f.rglob("*"))
            size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file()) / 1e6
            print(f"    ├─ {f.name}/  ({count} files, {size:.0f} MB)")
            shown.add(f.name)
        elif f.is_file():
            size_mb = f.stat().st_size / 1e6
            if size_mb >= 1:
                print(f"    ├─ {f.name}  ({size_mb:.0f} MB)")
            else:
                print(f"    ├─ {f.name}  ({size_mb*1000:.0f} KB)")
            shown.add(f.name)

    if failed > 0:
        print(f"\n  ⚠️  {failed} download(s) failed. Re-run to retry.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
