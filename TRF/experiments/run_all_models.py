"""
run_all_models.py — run all 4 TRF experiment scripts, then compare results.

Runs TRF_sklearn.py / TRF_mne.py / TRF_boosting.py / TRF_conv.py as
subprocesses (each already loops over every subject x condition and saves its
own pickles incrementally via results.py) so there's no shared-interpreter
state between them (each sets up its own torch/mne/eelbrain globals). If a
script fails, its failure is logged and the remaining scripts still run —
partial results are still useful, and each script's pickles are already
durable on disk before any failure downstream.

All 4 subprocesses are pinned to the same save_dir (resolved once, up front)
so a run spanning midnight doesn't split across two date-stamped folders.
Once every selected script has been attempted, compare_models.main() builds
comparison plots from whatever pickles ended up in that save_dir.

Usage
-----
    python run_all_models.py                       # all 4 models
    python run_all_models.py --models sklearn,mne   # subset
    python run_all_models.py --config config_lofi.yaml --sfreq 128
        # (any flag not claimed below is forwarded verbatim to every script)
"""
import argparse
import subprocess
import sys
from pathlib import Path

from config import load_config
import compare_models

SCRIPT_DIR = Path(__file__).resolve().parent

# Fastest/cheapest first, most expensive (GPU-bound SGD) last.
MODEL_SCRIPTS = {
    'sklearn': 'TRF_sklearn.py',
    'mne': 'TRF_mne.py',
    'boosting': 'TRF_boosting.py',
    'conv': 'TRF_conv.py',
}


def _build_parser():
    p = argparse.ArgumentParser(
        description="Run all 4 TRF experiment scripts, then compare results.")
    p.add_argument("--models", default=None,
                   help=f"Comma-separated subset of {list(MODEL_SCRIPTS)} (default: all).")
    return p


def main():
    args, forwarded_args = _build_parser().parse_known_args(sys.argv[1:])

    selected = list(MODEL_SCRIPTS) if args.models is None else args.models.split(',')
    unknown = set(selected) - set(MODEL_SCRIPTS)
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {list(MODEL_SCRIPTS)}")

    # Resolve one canonical save_dir up front so every subprocess (even ones
    # started after midnight relative to the first) lands in the same place.
    config = load_config(cli_args=sys.argv[1:])
    save_dir = config.save_dir
    if '--save-dir' not in forwarded_args:
        forwarded_args = forwarded_args + ['--save-dir', str(save_dir)]

    print(f"Running models {selected} -> save_dir = {save_dir}\n")

    results = {}
    for name in selected:
        script_path = SCRIPT_DIR / MODEL_SCRIPTS[name]
        cmd = [sys.executable, str(script_path), *forwarded_args]
        print(f"\n{'=' * 70}\nRunning {name} ({script_path.name})\n{'=' * 70}")
        proc = subprocess.run(cmd, cwd=SCRIPT_DIR)
        results[name] = proc.returncode == 0
        if proc.returncode != 0:
            print(f"[warn] {name} exited with code {proc.returncode} — "
                  f"continuing with the remaining scripts.")

    print(f"\n{'=' * 70}\nRun summary\n{'=' * 70}")
    for name in selected:
        print(f"  {'OK ' if results[name] else 'FAIL'} — {name}")

    if not any(results.values()):
        print("\nNo model script succeeded — skipping comparison plots.")
        return

    print(f"\n{'=' * 70}\nComparing results\n{'=' * 70}")
    compare_models.main(save_dir=save_dir)


if __name__ == '__main__':
    main()
