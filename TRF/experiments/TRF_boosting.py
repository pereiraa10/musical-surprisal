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
runtime down across 19 subjects x 2 feature_sets).

Instead, Y_pred/Y_true here are reconstructed post-hoc with eelbrain.convolve()
using the single final averaged TRF, applied per-trial. Per the BoostingResult
docstring, this is exactly the h * x operation boosting itself uses to
generate predictions, just with the across-all-partitions average h rather
than each partition's own training h. This makes `r_per_channel` in the
pickle authoritative (it's `trf_cv.r`, straight from eelbrain) while
`per_trial_r`/`Y_pred`/`Y_true` are a close, clearly-labeled approximation
useful for alignment plots and per-trial breakdowns — not a re-derivation of
`r_per_channel`.

boosting() is called with its default `scale_data=True`, so `trf_cv.h` is fit
in *internally normalized* units — it expects normalized x and produces
normalized y (this is a *global* normalization computed by eelbrain across
all cases, not the per-trial z-scoring used elsewhere in this codebase, but
close enough in scale for this purpose). `eelbrain.convolve()` is a plain
kernel convolution with no awareness of scaling, so `trf_cv.h` must be paired
with x in roughly matching (normalized) units: `ds.trials[i][k]` (the
per-trial z-scored features/EEG already used by TRF_ridge_3.py/TRF_conv_1.py,
via utils.zscore_trials) puts both Y_pred and Y_true in per-trial z-scored
units, matching the ridge/conv convention and the 'Actual EEG (z-scored)'
label in results.plot_alignment. Note `trf_cv.h_scaled` (eelbrain's
`h * y_scale/x_scale` rescaling) is deliberately NOT used here — it rescales
`h` to match *raw*, unnormalized x, which is the wrong pairing for the
z-scored x used below (verified empirically: convolving h_scaled against
z-scored x collapses Y_pred to ~1e-7, while h against z-scored x gives a
sensibly-scaled, weakly-correlated signal consistent with boosting's own
(low) r).
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
    internally — see module docstring). Both x and Y_true are drawn from
    ds.trials, which is already per-trial z-scored exactly like ridge/conv
    (utils.zscore_trials), so Y_pred and Y_true end up in the same units as
    the plot's 'Actual EEG (z-scored)' label assumes. trf_cv.h (not
    h_scaled) is used because it is fit against boosting's own internally
    normalized x/y — see module docstring for why h_scaled is the wrong
    pairing here."""
    h = trf_cv.h
    h_list = h if isinstance(h, (list, tuple)) else [h]

    Y_pred_trials, Y_true_trials = [], []
    for t in ds.trials:
        n_time = t['eeg'].shape[0]
        time_axis = eelbrain.UTS(0, 1 / ds.config.sfreq, n_time)
        x = [eelbrain.NDVar(t[k], (time_axis,), name=k) for k in feature_keys]
        y_pred = eelbrain.convolve(h_list, x)
        Y_pred_trials.append(y_pred.get_data(('sensor', 'time')).T)
        Y_true_trials.append(t['eeg'])

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
        # Load raw EEG + run the feature_set-independent pipeline once per
        # subject (PreparedSubject); each feature_set below only reruns the
        # cheap per-feature_set z-scoring step, not the full preprocessing.
        eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=SUBJECT)
        eeg_data = utils.load_subject_raw_eeg(
            eeg_path, SUBJECT, config.trial_to_stimulus.get(SUBJECT))
        prepared = PreparedSubject(SUBJECT, eeg_data, config, debug=DEBUG)

        for feature_set, feature_keys in config.feature_sets.items():
            ds = prepared.to_dataset(feature_set, window_samples=None)
            trf_cv = eelbrain.boosting(
                'eeg', feature_keys, config.tmin, config.tmax, data=ds.events,
                basis=BASIS, partitions=PARTITIONS, test=True, error=ERROR)

            r_per_channel = trf_cv.r.get_data('sensor')
            r_rank_per_channel = trf_cv.r_rank.get_data('sensor')

            try:
                Y_pred, Y_true, trial_boundaries = reconstruct_predictions(
                    ds, trf_cv, feature_keys)
            except Exception as exc:
                print(f"  [warn] {SUBJECT} | {feature_set}: could not reconstruct "
                      f"per-trial predictions ({exc}); saving without Y_pred/Y_true.")
                Y_pred = Y_true = trial_boundaries = None

            try:
                weights = extract_weights(trf_cv, feature_keys)
            except Exception as exc:
                print(f"  [warn] {SUBJECT} | {feature_set}: could not extract "
                      f"TRF weights ({exc}); saving weights=None.")
                weights = None

            result = res.build_result(
                subject=SUBJECT, subject_type=ds.subject_type, feature_set=feature_set,
                feature_keys=feature_keys, model_family='boosting',
                channel_names=ds.channel_names, Y_true=Y_true, Y_pred=Y_pred,
                trial_boundaries=trial_boundaries, r_per_channel=r_per_channel,
                weights=weights,
                extra_meta={
                    'r_rank_per_channel': r_rank_per_channel,
                    'basis': BASIS, 'partitions': PARTITIONS, 'error': ERROR,
                    'note': (
                        'Y_pred/Y_true reconstructed post-hoc via eelbrain.convolve '
                        'with the final averaged TRF; NOT the same '
                        'per-partition prediction boosting used internally for '
                        'r_per_channel. Both are in per-trial z-scored units, '
                        'matching TRF_ridge_3.py/TRF_conv_1.py. See EVALUATION_NOTES.md.'
                    ),
                },
            )
            path = res.result_filename(save_dir, SUBJECT, 'boosting', feature_set)
            res.save_result(path, result)

            print(f"  {SUBJECT} | {feature_set}: boosting mean r = {r_per_channel.mean():.4f}"
                  f"  (rank r = {r_rank_per_channel.mean():.4f})")


if __name__ == '__main__':
    main()
