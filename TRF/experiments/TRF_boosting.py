"""
TRF_boosting.py (experiments/ version) — eelbrain boosting TRF pipeline
=================================================================================

Modernized from ../TRF_pickle_A_and_AM.py: that script predated the per-trial
LPF -> downsample -> HPF -> strip-padding pipeline (it filtered the
concatenated signal) and sourced its envelope on the fly from the raw .wav
files at 100 Hz rather than from dataStim.mat's precomputed envelope at
64 Hz. This version goes through dataset.PreparedSubject + dataset.TRFDataset
like the other three experiment scripts, so all four are directly comparable
— see the "modernize" decision recorded in EVALUATION_NOTES.md.

Two things do NOT carry over unchanged, both intentional:

  1. TMIN/TMAX are now -0.1/0.6 (dataset.py's shared convention), not the
     original -0.05/0.55. Results are NOT numerically comparable to pickles
     produced by the old top-level script.
  2. eelbrain.boosting's cross-validated `r` is computed from its own
     internal train/validate/test partitioning (partitions=10), which is
     algorithmically different from the manual trial-based LOOCV the other
     three scripts implement by hand — see EVALUATION_NOTES.md. It is not
     "boosting's version of LOOCV," it's a different (and roughly comparable,
     but not identical) validation scheme.

Y_pred/Y_true reconstruction
-----------------------------
BoostingResult does not expose per-sample held-out predictions as a plain
array (verified against the installed eelbrain source: no y_pred attribute).
Its own `r` is computed, per partition, using the TRF fit on that partition's
*training* folds — a different h per partition, which isn't retained unless
`partition_results=True` is passed to boosting() (not done here, to keep
runtime down across 19 subjects x 2 conditions).

Instead, Y_pred/Y_true here are reconstructed post-hoc with eelbrain.convolve()
using the single final averaged TRF (`trf_cv.h`), applied per-trial. Per the
BoostingResult docstring, this is exactly the h * x operation boosting itself
uses to generate predictions, just with the across-all-partitions average h
rather than each partition's own training h. This makes `r_per_channel` in the
pickle authoritative (it's `trf_cv.r`, straight from eelbrain) while
`per_trial_r`/`Y_pred`/`Y_true` are a close, clearly-labeled approximation
useful for alignment plots and per-trial breakdowns — not a re-derivation of
`r_per_channel`.
"""

import os
import sys

import numpy as np
import eelbrain

from config import load_config
import utils
from dataset import PreparedSubject
import results as res

DEBUG = True
BASIS = 0.050
PARTITIONS = 10
ERROR = 'l1'


def reconstruct_predictions(ds, trf_cv, feature_keys):
    """Per-trial predicted EEG via eelbrain.convolve(trf_cv.h, x), using the
    final averaged TRF (not the per-partition training TRF boosting used
    internally — see module docstring)."""
    Y_pred_trials, Y_true_trials = [], []
    for i in range(len(ds.events['event'])):
        x = [ds.events[k][i] for k in feature_keys]
        y_pred = eelbrain.convolve(trf_cv.h, x)
        Y_pred_trials.append(y_pred.get_data(('sensor', 'time')).T)
        Y_true_trials.append(ds.events['eeg'][i].get_data(('sensor', 'time')).T)

    trial_boundaries = []
    offset = 0
    for arr in Y_true_trials:
        trial_boundaries.append((offset, offset + len(arr)))
        offset += len(arr)

    return np.concatenate(Y_pred_trials), np.concatenate(Y_true_trials), trial_boundaries


def extract_weights(trf_cv, feature_keys):
    """Stash trf_cv.h's raw data per predictor. Not forced into the ridge/conv
    (n_channels, n_lags, n_features) ndarray shape — boosting's h is already a
    named, dimensioned NDVar per predictor and reshaping would throw that
    array away for no benefit."""
    h_list = trf_cv.h if isinstance(trf_cv.h, (list, tuple)) else [trf_cv.h]
    return {
        'predictor_order': list(feature_keys),
        'h_by_predictor': {
            k: h.get_data(('time', 'sensor')) for k, h in zip(feature_keys, h_list)
        },
        'h_time': h_list[0].get_dim('time').times,
    }


def main():
    config = load_config(cli_args=sys.argv[1:])
    save_dir = config.paths.save_dir
    os.makedirs(save_dir, exist_ok=True)

    for SUBJECT in config.subjects:
        # Load raw EEG + run the condition-independent pipeline once per
        # subject (PreparedSubject); each condition below only reruns the
        # cheap per-condition z-scoring step, not the full preprocessing.
        eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=SUBJECT)
        eeg_data = utils.load_subject_raw_eeg(
            eeg_path, SUBJECT, config.trial_to_stimulus.get(SUBJECT))
        prepared = PreparedSubject(SUBJECT, eeg_data, config, debug=DEBUG)

        for condition, feature_keys in config.conditions.items():
            ds = prepared.to_dataset(condition, window_samples=None)
            trf_cv = eelbrain.boosting(
                'eeg', feature_keys, config.tmin, config.tmax, data=ds.events,
                basis=BASIS, partitions=PARTITIONS, test=True, error=ERROR)

            r_per_channel = trf_cv.r.get_data('sensor')
            r_rank_per_channel = trf_cv.r_rank.get_data('sensor')

            try:
                Y_pred, Y_true, trial_boundaries = reconstruct_predictions(
                    ds, trf_cv, feature_keys)
            except Exception as exc:
                print(f"  [warn] {SUBJECT} | {condition}: could not reconstruct "
                      f"per-trial predictions ({exc}); saving without Y_pred/Y_true.")
                Y_pred = Y_true = trial_boundaries = None

            try:
                weights = extract_weights(trf_cv, feature_keys)
            except Exception as exc:
                print(f"  [warn] {SUBJECT} | {condition}: could not extract "
                      f"TRF weights ({exc}); saving weights=None.")
                weights = None

            result = res.build_result(
                subject=SUBJECT, subject_type=ds.subject_type, condition=condition,
                feature_keys=feature_keys, model_family='boosting',
                channel_names=ds.channel_names, Y_true=Y_true, Y_pred=Y_pred,
                trial_boundaries=trial_boundaries, r_per_channel=r_per_channel,
                weights=weights,
                extra_meta={
                    'r_rank_per_channel': r_rank_per_channel,
                    'basis': BASIS, 'partitions': PARTITIONS, 'error': ERROR,
                    'note': (
                        'Y_pred/Y_true reconstructed post-hoc via eelbrain.convolve '
                        'with the final averaged TRF; NOT the same per-partition '
                        'prediction boosting used internally for r_per_channel. '
                        'See EVALUATION_NOTES.md.'
                    ),
                },
            )
            path = res.result_filename(save_dir, SUBJECT, 'boosting', condition)
            res.save_result(path, result)

            print(f"  {SUBJECT} | {condition}: boosting mean r = {r_per_channel.mean():.4f}"
                  f"  (rank r = {r_rank_per_channel.mean():.4f})")


if __name__ == '__main__':
    main()
