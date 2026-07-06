"""
TRF_mne.py (experiments/ version) — MNE ReceptiveField ridge TRF pipeline
============================================================================

Same algorithm as ../TRF_mne.py: alpha is selected via trial-based LOOCV using
MNE ReceptiveField's own internal lag-matrix construction (not the explicit
Toeplitz matrix TRF_sklearn.py builds). Data loading/preprocessing/
alignment/z-scoring now come from dataset.TRFDataset instead of being duplicated
inline.

Alpha values selected here are NOT comparable to TRF_sklearn.py's — the two
implementations construct/scale their lag matrices differently. See
EVALUATION_NOTES.md.
"""

import os
import sys

import numpy as np
from scipy.stats import pearsonr
from mne.decoding import ReceptiveField
from sklearn.linear_model import Ridge

from config import load_config
import utils
from dataset import TRFDataset
import results as res

RIDGE_ALPHAS = np.logspace(1, 7, 25)
DEBUG = True


def select_alpha_loocv_mne(X_all, Y_all, feature_keys, alphas, tmin, tmax, sfreq):
    """Trial-based LOOCV alpha search, refitting ReceptiveField per fold per
    alpha (no rank-1-update shortcut available for MNE's internal lag
    matrix). Returns (best_alpha, best_r, alpha_r)."""
    n_trials = len(X_all)
    alpha_r = {}
    best_alpha, best_r = alphas[0], -np.inf

    for alpha in alphas:
        r_folds = []
        for i in range(n_trials):
            X_train = np.concatenate([X_all[j] for j in range(n_trials) if j != i])
            Y_train = np.concatenate([Y_all[j] for j in range(n_trials) if j != i])

            rf = ReceptiveField(
                tmin, tmax, sfreq,
                feature_names=feature_keys,
                estimator=Ridge(alpha=alpha, fit_intercept=False),
            )
            rf.fit(X_train, Y_train)
            Y_pred_i = rf.predict(X_all[i])
            Y_true_i = Y_all[i]

            r_fold = np.mean([
                pearsonr(Y_true_i[:, ch], Y_pred_i[:, ch])[0]
                for ch in range(Y_true_i.shape[1])
            ])
            r_folds.append(r_fold)

        avg_r = float(np.mean(r_folds))
        alpha_r[float(alpha)] = avg_r
        if avg_r > best_r:
            best_r = avg_r
            best_alpha = alpha

    return best_alpha, best_r, alpha_r


def loocv_mne(X_all, Y_all, feature_keys, alpha, tmin, tmax, sfreq):
    """Leave-one-trial-out CV using MNE ReceptiveField (Ridge, alpha fixed).

    Returns (Y_pred_concat, Y_true_concat, coefs_per_fold, trial_boundaries).
    coefs_per_fold[i] : rf.coef_, shape (n_channels, n_features, n_lags).
    """
    Y_pred_all, Y_true_all, coefs, trial_boundaries = [], [], [], []
    offset = 0

    for i in range(len(X_all)):
        X_train = np.concatenate([X_all[j] for j in range(len(X_all)) if j != i])
        Y_train = np.concatenate([Y_all[j] for j in range(len(Y_all)) if j != i])

        rf = ReceptiveField(
            tmin, tmax, sfreq,
            feature_names=feature_keys,
            estimator=Ridge(alpha=alpha, fit_intercept=False),
        )
        rf.fit(X_train, Y_train)
        pred_i = rf.predict(X_all[i])

        Y_pred_all.append(pred_i)
        Y_true_all.append(Y_all[i])
        coefs.append(rf.coef_)   # (n_channels, n_features, n_lags)
        trial_boundaries.append((offset, offset + len(pred_i)))
        offset += len(pred_i)

    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all), coefs, trial_boundaries


def average_weights(coefs):
    """Average per-fold (n_channels, n_features, n_lags) coefficients across
    LOOCV folds and reshape to (n_channels, n_lags, n_features)."""
    mean_coef = np.mean(coefs, axis=0)   # (n_channels, n_features, n_lags)
    return np.ascontiguousarray(np.transpose(mean_coef, (0, 2, 1)))


def main():
    config = load_config(cli_args=sys.argv[1:])
    save_dir = config.paths.save_dir
    os.makedirs(save_dir, exist_ok=True)

    for SUBJECT in config.subjects:
        eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=SUBJECT)
        eeg_data = utils.load_subject_raw_eeg(eeg_path, SUBJECT)

        for condition, feature_keys in config.conditions.items():
            ds = TRFDataset(SUBJECT, eeg_data, condition, config,
                            window_samples=None, debug=DEBUG)
            trials = ds.trials
            X_all = [np.column_stack([t[k] for k in feature_keys]) for t in trials]
            Y_all = [t['eeg'] for t in trials]

            best_alpha, best_r, alpha_r = select_alpha_loocv_mne(
                X_all, Y_all, feature_keys, RIDGE_ALPHAS,
                config.tmin, config.tmax, config.sfreq)
            print(f"  {SUBJECT} | {condition} [MNE]: "
                  f"selected alpha = {best_alpha:.2e}  (mean CV r = {best_r:.4f})")

            Y_pred, Y_true, coefs, trial_boundaries = loocv_mne(
                X_all, Y_all, feature_keys, best_alpha,
                config.tmin, config.tmax, config.sfreq)
            weights = average_weights(coefs)

            result = res.build_result(
                subject=SUBJECT, subject_type=ds.subject_type, condition=condition,
                feature_keys=feature_keys, model_family='mne_ridge',
                channel_names=ds.channel_names, Y_true=Y_true, Y_pred=Y_pred,
                trial_boundaries=trial_boundaries, best_alpha=float(best_alpha),
                alpha_selection=alpha_r, weights=weights,
            )
            path = res.result_filename(save_dir, SUBJECT, 'mne_ridge', condition)
            res.save_result(path, result)

            print(f"  {SUBJECT} | {condition}: MNE Ridge mean r = "
                  f"{result['r_per_channel'].mean():.4f}")


if __name__ == '__main__':
    main()
