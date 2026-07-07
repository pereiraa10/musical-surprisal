"""
config.py — load all experiment constants from config.yaml, with CLI overrides.

Every dataset/preprocessing/TRF constant that used to live at the top of
dataset.py now lives in config.yaml. This module reads that YAML, resolves
relative paths against the TRF/ directory (the parent of experiments/), applies
any command-line overrides, and returns a single nested-dataclass `Config`
object used consistently across utils.py, dataset.py, and every TRF_*.py script.

Usage
-----
    from config import load_config
    config = load_config()                       # config.yaml next to this file
    config = load_config(cli_args=sys.argv[1:])  # honor --sfreq/--tmin/... overrides

    config.conditions            # {'acoustic': [...], 'acoustic_and_surprisal': [...]}
    config.tmin, config.tmax, config.sfreq
    config.subjects              # list[str]
    config.paths.eeg_dir, config.paths.save_dir  # resolved absolute Paths
    config.subject_type['Sub2']  # 'Non-musician'

Run directly to inspect the resolved config:
    python config.py                  # prints resolved config (YAML + CLI)
    python config.py --sfreq 128 ...  # with overrides
"""

import argparse
import sys
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

# experiments/ -> TRF/. Relative paths in the YAML are resolved against this,
# matching the old BASE_DIR = Path(__file__).resolve().parent.parent.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class Paths:
    base_dir: Path
    data_root: Path
    wav_dir: Path
    eeg_dir: Path
    save_dir: Path
    # Only needed for source_type='mat' (IDyOM/MIDI surprisal, liberi_dataset);
    # None for datasets with no symbolic score (see config.stimulus_source_type).
    midi_dir: Optional[Path] = None
    pitch_surprisal_file: Optional[Path] = None
    onset_surprisal_file: Optional[Path] = None


@dataclass
class Config:
    paths: Paths
    eeg_filename_pattern: str
    subjects: list
    subject_type: dict
    trial_to_song_id: dict
    low_frequency: int
    high_frequency: int
    sfreq: int
    tmin: float
    tmax: float
    ic_clip: float
    feature_keys_acoustic: list
    feature_keys_surprisal: list
    conditions: dict
    window_samples: Optional[int] = None
    hop_samples: Optional[int] = None
    # 'mat' (default, liberi_dataset's precomputed dataStim.mat) or
    # 'audio_files' (compute envelope on demand from raw stimulus audio) —
    # selects the _StimulusLibrary source mode in utils.get_stimulus_library.
    stimulus_source_type: str = 'mat'
    # {subject: [stimulus_path_or_None, ...]}, one entry per trial — only used
    # by non-'mat' EEG loaders that can't determine stimulus identity from the
    # EEG file itself (see utils._load_eeg_from_edf).
    trial_to_stimulus: dict = field(default_factory=dict)

    # Convenience alias so `config.save_dir` works like the old module constant.
    @property
    def save_dir(self):
        return self.paths.save_dir


def _resolve_paths(raw_paths, save_dir_override=None):
    """Join the YAML's relative paths against BASE_DIR (=TRF/) into absolute
    Paths, exactly as the old dataset.py computed BASE_DIR/DATA_ROOT/etc."""
    data_root = BASE_DIR / raw_paths["data_root"]
    if save_dir_override is not None:
        save_dir = Path(save_dir_override)
        if not save_dir.is_absolute():
            save_dir = BASE_DIR / save_dir
    else:
        save_dir = BASE_DIR / raw_paths["save_dir_template"].format(date=date.today())
    return Paths(
        base_dir=BASE_DIR,
        data_root=data_root,
        wav_dir=data_root / raw_paths["wav_subdir"],
        eeg_dir=data_root / raw_paths.get("eeg_subdir", ""),
        save_dir=save_dir,
        midi_dir=(data_root / raw_paths["midi_subdir"]
                  if "midi_subdir" in raw_paths else None),
        pitch_surprisal_file=(BASE_DIR / raw_paths["pitch_surprisal_file"]
                               if "pitch_surprisal_file" in raw_paths else None),
        onset_surprisal_file=(BASE_DIR / raw_paths["onset_surprisal_file"]
                               if "onset_surprisal_file" in raw_paths else None),
    )


def _build_parser():
    """argparse for CLI overrides. Every flag defaults to None so we can tell
    'not provided' from an explicit value and only override when given."""
    p = argparse.ArgumentParser(
        description="Load/override TRF experiment config (config.yaml).")
    p.add_argument("--config", default=None,
                   help="Path to a config.yaml (default: next to config.py).")
    p.add_argument("--eeg-filename-pattern", default=None,
                   help="Per-subject EEG filename, {subject}-templated.")
    p.add_argument("--sfreq", type=int, default=None, help="Target sampling rate (Hz).")
    p.add_argument("--tmin", type=float, default=None, help="TRF window start (s).")
    p.add_argument("--tmax", type=float, default=None, help="TRF window end (s).")
    p.add_argument("--low-frequency", type=int, default=None, help="HPF cutoff (Hz).")
    p.add_argument("--high-frequency", type=int, default=None, help="LPF cutoff (Hz).")
    p.add_argument("--ic-clip", type=float, default=None, help="IDyOM surprisal clip (bits).")
    p.add_argument("--save-dir", default=None, help="Override the output pickle dir.")
    p.add_argument("--window-samples", type=int, default=None,
                   help="TRFDataset fixed window length (samples); omit for full-trial.")
    p.add_argument("--hop-samples", type=int, default=None,
                   help="TRFDataset window hop (samples); defaults to window-samples.")
    return p


def load_config(path=None, cli_args=None):
    """Load config from YAML, apply CLI overrides, resolve paths.

    path     : explicit config.yaml path (else --config, else the default).
    cli_args : list of CLI tokens (e.g. sys.argv[1:]) to parse for overrides;
               None means "no CLI overrides" (safe for plain imports). Unknown
               args are ignored so a script's own flags don't break loading.
    """
    parser = _build_parser()
    args, _ = parser.parse_known_args([] if cli_args is None else cli_args)

    config_path = Path(path or args.config or DEFAULT_CONFIG_PATH)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # ── scalar overrides: CLI beats YAML ──
    eeg_filename_pattern = args.eeg_filename_pattern or raw["eeg_filename_pattern"]
    low_frequency = args.low_frequency if args.low_frequency is not None else raw["preprocessing"]["low_frequency"]
    high_frequency = args.high_frequency if args.high_frequency is not None else raw["preprocessing"]["high_frequency"]
    sfreq = args.sfreq if args.sfreq is not None else raw["preprocessing"]["sfreq"]
    tmin = args.tmin if args.tmin is not None else raw["trf"]["tmin"]
    tmax = args.tmax if args.tmax is not None else raw["trf"]["tmax"]
    ic_clip = args.ic_clip if args.ic_clip is not None else raw["trf"]["ic_clip"]

    win = raw.get("windowing", {}) or {}
    window_samples = args.window_samples if args.window_samples is not None else win.get("window_samples")
    hop_samples = args.hop_samples if args.hop_samples is not None else win.get("hop_samples")

    paths = _resolve_paths(raw["paths"], save_dir_override=args.save_dir)

    feature_keys_acoustic = list(raw["feature_keys"]["acoustic"])
    feature_keys_surprisal = list(raw["feature_keys"]["surprisal"])
    conditions = {
        "acoustic": feature_keys_acoustic,
        "acoustic_and_surprisal": feature_keys_acoustic + feature_keys_surprisal,
    }

    # trial_to_song_id keys must be ints (YAML usually parses them as ints
    # already, but coerce defensively so lookups by int(marker) never miss).
    # Absent entirely for datasets with no per-trial song table (e.g. no
    # symbolic score / surprisal features).
    trial_to_song_id = {int(k): int(v) for k, v in raw.get("trial_to_song_id", {}).items()}

    stimulus_source_type = raw.get("stimulus_source_type", "mat")
    # {subject: [stimulus_path_or_None, ...]}: resolve each filename against
    # wav_dir; a null YAML entry (unfilled placeholder) stays None.
    trial_to_stimulus = {
        subject: [None if fn is None else str(paths.wav_dir / fn) for fn in trials]
        for subject, trials in raw.get("trial_to_stimulus", {}).items()
    }

    return Config(
        paths=paths,
        eeg_filename_pattern=eeg_filename_pattern,
        subjects=list(raw["subjects"]),
        subject_type=dict(raw["subject_type"]),
        trial_to_song_id=trial_to_song_id,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        sfreq=sfreq,
        tmin=tmin,
        tmax=tmax,
        ic_clip=ic_clip,
        stimulus_source_type=stimulus_source_type,
        trial_to_stimulus=trial_to_stimulus,
        feature_keys_acoustic=feature_keys_acoustic,
        feature_keys_surprisal=feature_keys_surprisal,
        conditions=conditions,
        window_samples=window_samples,
        hop_samples=hop_samples,
    )


if __name__ == "__main__":
    cfg = load_config(cli_args=sys.argv[1:])
    print("Resolved config:")
    print(f"  config.yaml       : {DEFAULT_CONFIG_PATH}")
    print(f"  base_dir          : {cfg.paths.base_dir}")
    print(f"  data_root         : {cfg.paths.data_root}")
    print(f"  eeg_dir           : {cfg.paths.eeg_dir}")
    print(f"  save_dir          : {cfg.paths.save_dir}")
    print(f"  pitch_surprisal   : {cfg.paths.pitch_surprisal_file}")
    print(f"  onset_surprisal   : {cfg.paths.onset_surprisal_file}")
    print(f"  eeg_filename      : {cfg.eeg_filename_pattern}")
    print(f"  sfreq/tmin/tmax   : {cfg.sfreq} / {cfg.tmin} / {cfg.tmax}")
    print(f"  lpf/hpf/ic_clip   : {cfg.high_frequency} / {cfg.low_frequency} / {cfg.ic_clip}")
    print(f"  window/hop        : {cfg.window_samples} / {cfg.hop_samples}")
    print(f"  n subjects        : {len(cfg.subjects)}")
    print(f"  conditions        : { {k: v for k, v in cfg.conditions.items()} }")
    print(f"  subject_type[Sub2]: {cfg.subject_type.get('Sub2')}")
    print(f"  trial_to_song[1,11,20]: "
          f"{cfg.trial_to_song_id.get(1)}, {cfg.trial_to_song_id.get(11)}, {cfg.trial_to_song_id.get(20)}")
