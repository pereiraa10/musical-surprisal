"""
TRF_sklearn.py (experiments/ version) — explicit Toeplitz ridge TRF pipeline
==============================================================================

Same algorithm as ../TRF_sklearn.py: alpha is selected via trial-based LOOCV
on an explicit Toeplitz lag matrix (built here, not by a library). Data
loading/preprocessing/alignment/z-scoring now come from dataset.Dataset
instead of being duplicated inline.

Alpha values selected here are NOT comparable to TRF_mne.py's — the two
implementations construct/scale their lag matrices differently. See
EVALUATION_NOTES.md.
"""

import os

import numpy as np
from scipy.stats import pearsonr

from dataset import Dataset, CONDITIONS, TMIN, TMAX, SFREQ, SUBJECTS, SAVE_DIR
import results as res

RIDGE_ALPHAS = np.logspace(1, 7, 25)
DEBUG = True


def build_lag_matrix(x, tmin, tmax, sfreq):
    """Vectorized Toeplitz lag matrix using stride tricks.

    X[t, j] = x_padded[t + lag_max - j], so each row is a reversed sliding
    window of x_padded, giving every lag [lag_min .. lag_max] at once. Column
    0 -> lag_min, column n_lags-1 -> lag_max.
    """
    n_lags = int(round((tmax - tmin) * sfreq)) + 1
    lag_min = int(round(tmin * sfreq))
    lag_max = lag_min + n_lags - 1
    n = len(x)

    x_pad = np.concatenate([np.zeros(lag_max), x, np.zeros(max(0, -lag_min))])
    wins = np.lib.stride_tricks.sliding_window_view(x_pad, n_lags)
    return np.ascontiguousarray(wins[:n, ::-1])


def build_design_matrix(features, tmin, tmax, sfreq):
    """Concatenate lag matrices for all features. features: dict {name: 1-D array}.
    Column blocks are feature-major: [feat0's n_lags cols, feat1's n_lags cols, ...]."""
    return np.hstack([build_lag_matrix(v, tmin, tmax, sfreq) for v in features.values()])


def select_alpha_loocv(Phi_all, Y_all, alphas):
    """Trial-based LOOCV alpha search via XTX/XTY rank-1 updates (O(T_i*p^2)
    per fold instead of restacking n-1 trials). Returns (best_alpha, best_r,
    alpha_r) where alpha_r is {alpha: mean_cv_r} for every candidate."""
    Phi_full = np.concatenate(Phi_all)
    Y_full = np.concatenate(Y_all)
    p = Phi_full.shape[1]
    XTX = Phi_full.T @ Phi_full
    XTY = Phi_full.T @ Y_full

    alpha_r = {}
    best_alpha, best_r = alphas[0], -np.inf
    for alpha in alphas:
        alpha_I = alpha * np.eye(p)
        r_folds = []
        for Phi_i, Y_i in zip(Phi_all, Y_all):
            XTX_tr = XTX - Phi_i.T @ Phi_i
            XTY_tr = XTY - Phi_i.T @ Y_i
            W = np.linalg.solve(XTX_tr + alpha_I, XTY_tr)
            r_fold = np.mean([
                pearsonr(Y_i[:, ch], (Phi_i @ W)[:, ch])[0]
                for ch in range(Y_i.shape[1])
            ])
            r_folds.append(r_fold)
        avg_r = float(np.mean(r_folds))
        alpha_r[float(alpha)] = avg_r
        if avg_r > best_r:
            best_r = avg_r
            best_alpha = alpha

    return best_alpha, best_r, alpha_r


def loocv_ridge(Phi_all, Y_all, alpha):
    """Leave-one-trial-out CV with a fixed alpha using XTX rank-1 updates.

    Returns (Y_pred_concat, Y_true_concat, coefs_per_fold, trial_boundaries).
    coefs_per_fold[i] : (n_channels, n_features*n_lags) ridge weights for fold i.
    """
    Phi_full = np.concatenate(Phi_all)
    Y_full = np.concatenate(Y_all)

    p = Phi_full.shape[1]
    XTX = Phi_full.T @ Phi_full
    XTY = Phi_full.T @ Y_full
    alpha_I = alpha * np.eye(p)

    Y_pred = np.zeros_like(Y_full)
    coefs = []
    trial_boundaries = []
    offset = 0

    for Phi_i, Y_i in zip(Phi_all, Y_all):
        n_i = len(Phi_i)
        XTX_train = XTX - Phi_i.T @ Phi_i
        XTY_train = XTY - Phi_i.T @ Y_i
        W = np.linalg.solve(XTX_train + alpha_I, XTY_train)
        Y_pred[offset:offset + n_i] = Phi_i @ W
        coefs.append(W.T)   # (n_channels, n_features*n_lags)
        trial_boundaries.append((offset, offset + n_i))
        offset += n_i

    return Y_pred, Y_full, coefs, trial_boundaries


def average_weights(coefs, n_features, n_lags):
    """Average per-fold (n_channels, n_features*n_lags) coefficients across
    LOOCV folds and reshape to (n_channels, n_lags, n_features).

    build_design_matrix concatenates feature-major blocks (feat0's n_lags
    columns, then feat1's, ...), so the flat coefficient axis is
    (n_features, n_lags) before the final transpose.
    """
    mean_coef = np.mean(coefs, axis=0)                    # (n_channels, n_features*n_lags)
    n_channels = mean_coef.shape[0]
    reshaped = mean_coef.reshape(n_channels, n_features, n_lags)
    return np.ascontiguousarray(np.transpose(reshaped, (0, 2, 1)))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    n_lags = int(round((TMAX - TMIN) * SFREQ)) + 1

    for SUBJECT in SUBJECTS:
        ds = Dataset(SUBJECT, debug=DEBUG)

        for condition, feature_keys in CONDITIONS.items():
            trials = ds.get_trials(condition)
            Phi_all = [
                build_design_matrix({k: t[k] for k in feature_keys}, TMIN, TMAX, SFREQ)
                for t in trials
            ]
            Y_all = [t['eeg'] for t in trials]

            best_alpha, best_r, alpha_r = select_alpha_loocv(Phi_all, Y_all, RIDGE_ALPHAS)
            print(f"  {SUBJECT} | {condition} [sklearn]: "
                  f"selected alpha = {best_alpha:.2e}  (mean CV r = {best_r:.4f})")

            Y_pred, Y_true, coefs, trial_boundaries = loocv_ridge(Phi_all, Y_all, best_alpha)
            weights = average_weights(coefs, len(feature_keys), n_lags)

            result = res.build_result(
                subject=SUBJECT, subject_type=ds.subject_type, condition=condition,
                feature_keys=feature_keys, model_family='sklearn_ridge',
                channel_names=ds.channel_names, Y_true=Y_true, Y_pred=Y_pred,
                trial_boundaries=trial_boundaries, best_alpha=float(best_alpha),
                alpha_selection=alpha_r, weights=weights,
            )
            path = res.result_filename(SAVE_DIR, SUBJECT, 'sklearn_ridge', condition)
            res.save_result(path, result)

            print(f"  {SUBJECT} | {condition}: sklearn Ridge mean r = "
                  f"{result['r_per_channel'].mean():.4f}")


if __name__ == '__main__':
    main()
