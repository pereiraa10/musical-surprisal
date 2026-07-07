"""
run_daly_dataset.py — extract EEG + audio features (envelope/onsets) for the
ds002725 (Daly et al. 2019) classicalMusic task.

Mirrors just the setup preamble every TRF_*.py script shares (load config ->
loop subjects -> build PreparedSubject -> to_dataset(...)), stopping at
feature extraction rather than fitting a TRF model. This is NOT a
replacement for run_all_models.py, which orchestrates the 4 full
TRF-fitting scripts (TRF_sklearn/TRF_mne/TRF_boosting/TRF_conv) as
subprocesses and compares their results — pointing those scripts at
config_daly.yaml to actually fit models is a later step, once
trial_to_stimulus is filled in for real.

config_daly.yaml's trial_to_stimulus is currently an unfilled placeholder
(see its comments, and the calibration notes in utils._load_eeg_from_edf):
the EEG channels documented to carry stimulus identity are flat at the ADC
floor in this BIDS release. So per-subject full feature extraction
(to_dataset('acoustic')) is expected to raise a clear, actionable error
until that mapping is supplied. This script reports on what already works
independent of that mapping:
  1. EDF loading + trial segmentation (utils.load_subject_raw_eeg) --
     works today, no stimulus mapping required.
  2. Envelope computation directly from a stimulus audio file
     (utils.compute_envelope_from_audio) -- works today, independent of
     trial alignment.
  3. The full PreparedSubject -> TRFDataset('acoustic') pipeline -- will
     succeed once trial_to_stimulus is filled in for a subject; until then
     it reports the documented error instead of crashing the whole run.

Usage
-----
    python run_daly_dataset.py                       # all subjects in config_daly.yaml
    python run_daly_dataset.py --subjects sub-01,sub-02
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import utils
from config import load_config
from dataset import PreparedSubject

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_daly.yaml"


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None,
                   help="Path to config_daly.yaml (default: next to this script).")
    p.add_argument("--subjects", default=None,
                   help="Comma-separated subset of config's subjects (default: all).")
    return p


def report_envelope_from_audio(config):
    """Sanity-check compute_envelope_from_audio directly against one real
    stimulus file, independent of any trial alignment."""
    sample_mp3 = sorted(config.paths.wav_dir.glob('*.mp3'))[0]
    env = utils.compute_envelope_from_audio(sample_mp3, config.sfreq)
    print(f"compute_envelope_from_audio({sample_mp3.name!r}, sfreq={config.sfreq}) -> "
          f"shape={env.shape}, dtype={env.dtype}, "
          f"range=[{env.min():.4g}, {env.max():.4g}], "
          f"duration={len(env) / config.sfreq:.1f}s")


def load_and_report_eeg(config, subject):
    """Load one subject's EDF + trial segmentation and report basic shape
    info. Works today regardless of trial_to_stimulus."""
    eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=subject)
    eeg_data = utils.load_subject_raw_eeg(
        eeg_path, subject, trial_to_stimulus=config.trial_to_stimulus.get(subject))
    durations_s = [round(t.shape[0] / eeg_data['fs'], 1) for t in eeg_data['trials']]
    print(f"  trials={len(eeg_data['trials'])} "
          f"channels={eeg_data['trials'][0].shape[1]} fs={eeg_data['fs']}Hz "
          f"durations(s)={durations_s}")
    return eeg_data


def try_extract_features(config, subject, eeg_data):
    """Attempt the full PreparedSubject -> to_dataset('acoustic') pipeline.
    Expected to raise (a documented, actionable ValueError) while
    trial_to_stimulus is still an unfilled placeholder."""
    try:
        prepared = PreparedSubject(subject, eeg_data, config)
        ds = prepared.to_dataset('acoustic')
        for t in ds.trials:
            assert set(t) == {'eeg', 'envelope', 'onsets'}, set(t)
            for k, arr in t.items():
                assert np.isfinite(arr).all(), f"non-finite values in {k!r}"
        print(f"  [OK] to_dataset('acoustic'): {ds.n_trials} trials, "
              f"{ds.n_channels} channels, lengths={ds.trial_lengths}")
    except ValueError as e:
        print(f"  [blocked] {e}")


def main():
    args, _ = _build_parser().parse_known_args(sys.argv[1:])
    config = load_config(path=args.config or DEFAULT_CONFIG_PATH)
    subjects = args.subjects.split(',') if args.subjects else config.subjects

    print("== compute_envelope_from_audio sanity check ==")
    report_envelope_from_audio(config)

    for subject in subjects:
        print(f"\n== {subject} ==")
        eeg_data = load_and_report_eeg(config, subject)
        try_extract_features(config, subject, eeg_data)


if __name__ == '__main__':
    main()
