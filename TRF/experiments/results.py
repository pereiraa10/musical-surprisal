"""
results.py — shared pickle output schema and filename convention for the TRF
experiment scripts in experiments/.

This intentionally contains no model-fitting or evaluation logic (see
EVALUATION_NOTES.md for that — LOOCV harnesses, alpha selection, and
lag-matrix construction stay in each script since they differ per model
family). It only standardizes what gets written to disk once a script has
already produced its own Y_pred/Y_true/weights, so every pickle in
dataset.SAVE_DIR has the same shape regardless of which script produced it.
That's what makes cross-model notebook analysis (avg_trf.ipynb,
correlation_plot.ipynb) possible without per-script loading code.

Filename convention
--------------------
    {subject}__{model_family}[_{variant}]__{condition}.pkl
e.g. Sub10__sklearn_ridge__acoustic.pkl
     Sub10__conv__nonlinear__acoustic_and_surprisal.pkl   (variant kept distinct)

Replaces the old `Sub10_['envelope', 'onsets']_sklearn_ridge_acoustic_data.pkl`
scheme, which embedded the raw Python list repr of feature_keys in the
filename. feature_keys now live in meta['feature_keys'] instead.

Pickle schema
-------------
    {
      'meta': {
          'subject', 'subject_type', 'condition', 'feature_keys',
          'model_family', 'model_variant', 'channel_names',
          'trial_boundaries',   # list[(start, end)] into Y_pred/Y_true per trial
          'best_alpha', 'date_run', ...any extra_meta passed in
      },
      'r_per_channel':    (n_channels,) ndarray — whole-session concatenated-holdout r
      'per_trial_r':      (n_trials, n_channels) ndarray
      'Y_pred', 'Y_true': (T_total, n_channels) float32 ndarray — always saved,
                          every model. float32 (not float64): a full-precision
                          pair runs ~300MB/file, which multiplies to 50-100+ GB
                          across 20 subjects x 2 conditions x ~6 model/variants;
                          float32 halves that and is still far finer than the
                          64 Hz EEG signal's actual precision. Correlations in
                          this file are computed before downcasting.
      'weights':          (n_channels, n_lags, n_features) ndarray or None
      'alpha_selection':  dict {alpha: mean_cv_r} or None   (ridge only)
      'training_history': dict of per-epoch arrays or None (conv only)
    }
"""
import pickle
from datetime import date

import numpy as np
from scipy.stats import pearsonr


def result_filename(save_dir, subject, model_family, condition, variant=None):
    model_tag = model_family if variant is None else f'{model_family}_{variant}'
    return save_dir / f'{subject}__{model_tag}__{condition}.pkl'


def per_trial_r(Y_true, Y_pred, trial_boundaries):
    """Per-trial, per-channel Pearson r. Returns (n_trials, n_channels)."""
    n_channels = Y_true.shape[1]
    out = np.zeros((len(trial_boundaries), n_channels))
    for ti, (start, end) in enumerate(trial_boundaries):
        for ch in range(n_channels):
            out[ti, ch] = pearsonr(Y_true[start:end, ch], Y_pred[start:end, ch])[0]
    return out


def build_result(*, subject, subject_type, condition, feature_keys, model_family,
                  channel_names, Y_true=None, Y_pred=None, trial_boundaries=None,
                  r_per_channel=None, model_variant=None, best_alpha=None,
                  alpha_selection=None, weights=None, training_history=None,
                  extra_meta=None):
    """Assemble the standardized result dict every script pickles.

    Usual case (sklearn/mne/conv)
    ------------------------------
    Y_true, Y_pred : (T_total, n_channels) ndarray
        Concatenated held-out predictions/actual, in trial order.
    trial_boundaries : list[(start, end)]
        Index ranges into Y_true/Y_pred per trial, so per-trial slices (for
        alignment plots, per-trial r, etc.) don't require re-running the model.
    r_per_channel is computed from Y_true/Y_pred automatically when both are given.

    Exception (eelbrain.boosting)
    ------------------------------
    eelbrain.boosting's cross-validated predictions aren't exposed as a plain,
    trial-aligned array (see EVALUATION_NOTES.md), so TRF_pickle_A_and_AM.py
    passes Y_true=Y_pred=None and supplies `r_per_channel` directly from
    `trf_cv.r`. In that case `per_trial_r` is left as None.

    Optional
    --------
    best_alpha, alpha_selection : ridge only.
        alpha_selection is a dict {alpha: mean_cv_r} from the LOOCV alpha
        search — previously computed then discarded without being saved.
    weights : (n_channels, n_lags, n_features) ndarray or None.
        The TRF kernel, where the model family has one in that shape (all
        ridge variants; conv 'linear'/'separable' variants, since their first
        conv layer *is* the TRF kernel).
    training_history : dict of per-epoch arrays, conv models only.
    """
    pt_r = None
    if Y_true is not None and Y_pred is not None:
        n_channels = Y_true.shape[1]
        computed_r = np.array([
            pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(n_channels)
        ])
        r_per_channel = computed_r if r_per_channel is None else r_per_channel
        if trial_boundaries is not None:
            pt_r = per_trial_r(Y_true, Y_pred, trial_boundaries)
    elif r_per_channel is None:
        raise ValueError(
            "build_result needs either (Y_true and Y_pred) or an explicit "
            "r_per_channel override.")

    meta = {
        'subject': subject,
        'subject_type': subject_type,
        'condition': condition,
        'feature_keys': list(feature_keys),
        'model_family': model_family,
        'model_variant': model_variant,
        'channel_names': list(channel_names),
        'trial_boundaries': trial_boundaries,
        'best_alpha': best_alpha,
        'date_run': str(date.today()),
    }
    if extra_meta:
        meta.update(extra_meta)

    # Correlations above are computed from the original (float64) arrays for
    # numerical accuracy; only the stored copies are downcast. A ~300MB/file
    # float64 Y_pred+Y_true pair (20 subjects x 2 conditions x ~6 models would
    # be 50-100+ GB) halves to ~150MB in float32, which is still far finer
    # than the 64 Hz EEG signal's actual precision.
    if Y_pred is not None:
        Y_pred = Y_pred.astype(np.float32)
    if Y_true is not None:
        Y_true = Y_true.astype(np.float32)

    return {
        'meta': meta,
        'r_per_channel': r_per_channel,
        'per_trial_r': pt_r,
        'Y_pred': Y_pred,
        'Y_true': Y_true,
        'weights': weights,
        'alpha_selection': alpha_selection,
        'training_history': training_history,
    }


def save_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(result, f)
