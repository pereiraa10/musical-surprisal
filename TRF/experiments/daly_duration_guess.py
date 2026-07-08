"""
daly_duration_guess.py — best-effort, UNVERIFIED duration-clustering guess at
the ds002725 (Daly et al. 2019) classicalMusic task's trial_to_stimulus
mapping.

Standalone analysis tool, not imported by the pipeline. See the plan notes /
config_daly.yaml's trial_to_stimulus comment for why an automatic decode
isn't otherwise possible: the `music`/`trialtype`/`ft_valance`/`ft_arousal`/
`ft_ghostvalence`/`ft_ghostarousal` channels documented to carry stimulus
identity or trial type are all flat at the ADC floor in this BIDS release.

Per the paper's Methods (Daly et al. 2019), each subject's classicalMusic
session plays 4 of the 7 available classical pieces, each in 3 trial types
(music-only, music-and-reporting, reporting-only) = 12 trials — matching the
12 trial-start markers found in every subject's events.tsv. music-only and
music-and-reporting trials play a piece's audio start-to-finish, so their
duration should closely match that piece's real length (mod a few seconds of
trigger jitter); reporting-only trials replay a FEELTRACE trace of
presumably similar duration but have NO real audio, regardless of which
piece this heuristic assigns them.

Method: for each subject, brute-force all C(7,4)=35 four-piece subsets; for
each subset, assign every one of the 12 measured trial durations to its
nearest-duration piece among those 4; keep the subset + assignment with the
lowest total absolute duration error. This enforces the paper's "only 4 of 7
pieces used" constraint, which naive per-trial nearest-neighbor matching
against all 7 would ignore.

This is NOT channel-verified and does NOT distinguish trial type — treat the
output as a starting hypothesis to manually verify, not a ready-to-use
mapping. It is deliberately NOT wired into config_daly.yaml's
trial_to_stimulus (which stays null, the safe default); this script writes
to a separate trial_to_stimulus_guess_daly.yaml instead.

Usage
-----
    python daly_duration_guess.py
"""
import itertools
from pathlib import Path

import mne
import pandas as pd
import soundfile as sf
import yaml

from config import load_config

OUTPUT_PATH = Path(__file__).resolve().parent / "trial_to_stimulus_guess_daly.yaml"

_HEADER = """\
# trial_to_stimulus_guess_daly.yaml — UNVERIFIED, experimental duration-based
# guess at the ds002725 classicalMusic trial_to_stimulus mapping. Generated
# by daly_duration_guess.py -- see that file's docstring for the method and
# its limitations.
#
# NOT channel-verified. Does NOT distinguish trial type (music-only vs.
# music-and-reporting vs. reporting-only) -- the reporting-only third of
# trials has no real audio regardless of which piece is listed here. Do NOT
# use this for a real TRF fit without independent verification. It is not
# read by config.py / the pipeline; config_daly.yaml's own trial_to_stimulus
# stays null.
#
# Each subject's entry gives, per trial: [guessed_filename, abs_error_seconds].
# total_error_seconds (per subject) is the sum of abs_error across its 12
# trials -- a large value is a signal this subject's guess is unreliable.
#
# Two further known limitations, visible in the numbers below:
#  - Assignment is nearest-duration per trial within the chosen 4-piece
#    subset, NOT balanced to exactly 3 trials/piece (the paper's actual
#    design) -- in practice most subjects skew heavily toward one piece
#    (often p5_mendelssohn, whose ~160s duration sits centrally among the
#    observed trial durations), which is a real assignment artifact, not
#    evidence that piece was replayed that often.
#  - Every subject's LAST trial shows an outsized error: there is no
#    trial-end marker in this dataset, so the last trial's boundary falls
#    back to end-of-recording, which can include dead time after the final
#    piece actually finished playing -- treat every subject's last entry as
#    especially unreliable.
# 'missing_edf' subjects have no classicalMusic recording in the dataset at all.

"""


def classical_piece_durations(wav_dir):
    """{filename: duration_seconds} for every mp3 in wav_dir, via a cheap
    header read (soundfile.info), no full decode."""
    return {p.name: sf.info(str(p)).frames / sf.info(str(p)).samplerate
            for p in sorted(wav_dir.glob('*.mp3'))}


def trial_durations_for_subject(config, subject):
    """The 12 classicalMusic trial durations for `subject`, computed the
    same way _load_eeg_from_edf does (trial i: onset_i -> onset_{i+1}, last
    -> end of recording), without loading full EEG signal data."""
    eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=subject)
    events_path = eeg_path.parent / eeg_path.name.replace('_eeg.edf', '_events.tsv')
    events_df = pd.read_csv(events_path, sep='\t')
    onsets = events_df.loc[events_df['trial_type'] == 768, 'onset'].to_numpy()

    raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose='ERROR')
    total_duration = raw.n_times / raw.info['sfreq']

    ends = list(onsets[1:]) + [total_duration]
    return [e - s for s, e in zip(onsets, ends)]


def best_subset_assignment(trial_durs, piece_durs, subset_size=4):
    """Try every `subset_size`-piece subset of piece_durs; for each, assign
    every trial duration to its nearest-duration piece in the subset. Return
    (best_subset_names, per_trial_assignment, per_trial_abs_error) for the
    subset+assignment with the lowest total absolute error."""
    names = list(piece_durs)
    best = None
    for subset in itertools.combinations(names, subset_size):
        assignment, errors = [], []
        for td in trial_durs:
            name = min(subset, key=lambda n: abs(piece_durs[n] - td))
            assignment.append(name)
            errors.append(abs(piece_durs[name] - td))
        total_error = sum(errors)
        if best is None or total_error < best[0]:
            best = (total_error, subset, assignment, errors)
    _, subset, assignment, errors = best
    return list(subset), assignment, errors


def main():
    config = load_config(path=Path(__file__).resolve().parent / "config_daly.yaml")
    piece_durs = classical_piece_durations(config.paths.wav_dir)
    print(f"Classical piece durations: "
          f"{ {k: round(v, 1) for k, v in piece_durs.items()} }\n")

    results = {}
    for subject in config.subjects:
        eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=subject)
        if not eeg_path.exists():
            print(f"{subject}: SKIPPED (no classicalMusic EDF for this subject in the dataset)")
            results[subject] = 'missing_edf'
            continue
        trial_durs = trial_durations_for_subject(config, subject)
        subset, assignment, errors = best_subset_assignment(trial_durs, piece_durs)
        total_error = sum(errors)
        distinct = len(set(assignment))
        print(f"{subject}: total_error={total_error:.1f}s distinct_pieces={distinct} "
              f"subset={sorted(subset)}")
        results[subject] = {
            'total_error_seconds': round(float(total_error), 1),
            'trials': [[name, round(float(err), 1)] for name, err in zip(assignment, errors)],
        }

    with open(OUTPUT_PATH, 'w') as f:
        f.write(_HEADER)
        yaml.safe_dump(results, f, sort_keys=False)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
