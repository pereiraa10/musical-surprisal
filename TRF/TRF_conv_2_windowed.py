"""
TRF_conv_2_windowed.py
────────────────────────────────────────────────────────────────────────────────
Windowed mini-batch training variant of TRF_conv_1.py.

The scientific evaluation protocol — subject-level training, leave-one-trial-out
(LOOCV) cross-validation, Pearson r computed on the concatenated held-out
predictions per channel — is identical to TRF_conv_1.py and TRF_ridge_3.py.
Held-out r values from this script are directly comparable to both baselines.

What changed vs TRF_conv_1.py
────────────────────────────────────────────────────────────────────────────────
[1] TRAINING STRATEGY  (train_one_fold)
    TRF_conv_1  — each full trial is a single forward pass (effective batch=1).
    TRF_conv_2  — each trial is sliced into overlapping WINDOW_SAMPLES-length
                  windows using a sliding window with stride HOP_SAMPLES.
                  Gradient updates use mini-batches of BATCH_SIZE windows drawn
                  from all training trials.  Better gradient variance and more
                  effective use of the optimiser for nonlinear models.

[2] NORMALIZATION  (StimToEEG — nonlinear variant only)
    nn.BatchNorm1d(hidden)  →  nn.GroupNorm(num_groups=4, num_channels=hidden)
    GroupNorm normalises within each instance independently of batch size.  EEG
    sequences are highly autocorrelated so BatchNorm statistics fluctuate across
    trials; GroupNorm is more stable.

[3] EARLY STOPPING CRITERION  (train_one_fold)
    Validation MSE is still computed on validation windows (diagnostic only).
    The checkpoint-selection and early-stopping metric is validation Pearson r
    measured on the FULL validation trial, preserving consistency with TRF_conv_1.

[4] DEBUG DIAGNOSTICS  (loocv_conv)
    Each fold prints: number of training trials, training windows, validation
    windows, and held-out trial length.

[5] LEAKAGE GUARD  (loocv_conv)
    A runtime assertion verifies that the held-out test trial object is not
    present in the training set for every fold.

[6] ALIGNMENT PLOT  (plot_alignment, main loop)
    After each subject/condition LOOCV, saves a two-panel figure comparing
    the predicted and actual EEG for the channel at CHANNEL_IDX.

All stimulus loading, preprocessing, z-scoring, the LOOCV outer loop, and the
held-out evaluation and result-saving logic are unchanged from TRF_conv_1.py.
────────────────────────────────────────────────────────────────────────────────

Run from musical-surprisal/TRF/:
    python TRF_conv_2_windowed.py

NOTE: requires PyTorch (pip install torch).  No GPU required; runs on CPU.
"""

import csv
import os
import warnings
from math import gcd
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly as sp_resample_poly
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader   # [CHANGE 1] windowed mini-batch support

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt

import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func
import eelbrain


# ─── Config (mirrors TRF_conv_1.py / TRF_ridge_3.py where shared) ──────────────

TMIN  = -0.1    # seconds  — receptive field start (pre-stimulus)
TMAX  = 0.600   # seconds  — receptive field end   (post-stimulus)
SFREQ = 64      # Hz after resampling
IC_CLIP = 15.0

# Receptive field in taps, derived exactly like build_lag_matrix in the ridge code
N_LAGS  = int(round((TMAX - TMIN) * SFREQ)) + 1   # e.g. 0.7s * 64 + 1 = 45 taps
LAG_MIN = int(round(TMIN * SFREQ))                # e.g. -6  (pre-stimulus taps)
LAG_MAX = LAG_MIN + N_LAGS - 1                    # e.g.  38 (post-stimulus taps)

# ── Deep-learning knobs ──
MODEL_VARIANT = 'nonlinear'   # 'linear' | 'separable' | 'nonlinear'
HIDDEN        = 32            # width of the shared temporal feature bank
N_BLOCKS      = 2             # nonlinear conv blocks (only used by 'nonlinear')
EPOCHS        = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-3          # the SGD-era analog of ridge alpha
EARLY_STOP_PATIENCE = 25      # epochs without validation-r improvement before stopping

# ── [CHANGE 1] Windowing constants ──────────────────────────────────────────────
# Replace full-trial training with overlapping-window mini-batches.
# Windows are extracted independently from each trial; no window crosses trial
# boundaries, so the LOOCV held-out guarantee is preserved.
WINDOW_SEC     = 7.0          # window length in seconds
HOP_SEC        = 6.0          # hop (stride) in seconds
BATCH_SIZE     = 64           # windows per gradient step

WINDOW_SAMPLES = int(WINDOW_SEC * SFREQ)   # 128 samples
HOP_SAMPLES    = int(HOP_SEC   * SFREQ)   # 6 samples

# ── Alignment plot channel ───────────────────────────────────────────────────────
CHANNEL_IDX = 0               # EEG channel used in the alignment figure

DEBUG = True
SEED  = 0


def _select_device():
    """Pick the best available backend: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _select_device()
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)


def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


# ════════════════════════════════════════════════════════════════════════════════
# [CHANGE 1] Windowing utilities
# ════════════════════════════════════════════════════════════════════════════════

def _make_windows(X, Y, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES):
    """Slice one trial into overlapping fixed-length windows.

    X : (T, n_features)  numpy array (z-scored, as produced by the ridge pipeline)
    Y : (T, n_channels)  numpy array

    Returns
    -------
    X_wins : (n_windows, n_features, window_samples)  — channel-first for Conv1d
    Y_wins : (n_windows, n_channels, window_samples)

    Sliding window: start positions 0, hop, 2*hop, … up to the last position
    where a full window fits.  No window crosses the trial boundary.
    """
    T = X.shape[0]
    starts = list(range(0, T - window_samples + 1, hop_samples))
    if len(starts) == 0:
        raise ValueError(
            f"Trial length {T} < WINDOW_SAMPLES {window_samples}; "
            "cannot extract any windows.  Shorten WINDOW_SEC or check the data.")
    # Transpose each window slice to channel-first layout.
    X_wins = np.stack([X[s:s + window_samples].T for s in starts], axis=0)
    Y_wins = np.stack([Y[s:s + window_samples].T for s in starts], axis=0)
    return X_wins, Y_wins


def _count_windows(trial_list, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES):
    """Return the total number of windows that would be extracted from a list of trials."""
    return sum(
        max(0, (x.shape[0] - window_samples) // hop_samples + 1)
        for x in trial_list
    )


# ════════════════════════════════════════════════════════════════════════════════
# [CHANGE 1] PyTorch Dataset for pre-computed windows
# ════════════════════════════════════════════════════════════════════════════════

class WindowDataset(Dataset):
    """Dataset wrapping precomputed sliding-window excerpts from one or more trials.

    Windows are stored channel-first:
        X : (n_windows, n_features, WINDOW_SAMPLES)
        Y : (n_windows, n_channels, WINDOW_SAMPLES)

    This layout matches Conv1d's expected (batch, channels, length) convention so
    no transposition is needed inside the training loop.
    """

    def __init__(self, X_wins: np.ndarray, Y_wins: np.ndarray):
        self.X = torch.from_numpy(X_wins.astype(np.float32))
        self.Y = torch.from_numpy(Y_wins.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ════════════════════════════════════════════════════════════════════════════════
# Model  (same architecture as TRF_conv_1.py; only the nonlinear variant changes)
# ════════════════════════════════════════════════════════════════════════════════
#
# Tensor convention: (batch, channels=features, time).  The CausalPad module
# ensures each output sample sees the same [LAG_MIN, LAG_MAX] context the ridge
# Toeplitz matrix provides, independent of batch size or window length.

class CausalPad(nn.Module):
    """Left/right pad so output length == input length AND the kernel spans the
    same [LAG_MIN, LAG_MAX] window the ridge lag matrix used."""
    def __init__(self, left, right):
        super().__init__()
        self.left, self.right = left, right

    def forward(self, x):
        return nn.functional.pad(x, (self.left, self.right))


class StimToEEG(nn.Module):
    def __init__(self, n_features, n_channels, variant='nonlinear',
                 hidden=HIDDEN, n_blocks=N_BLOCKS):
        super().__init__()
        self.variant = variant
        # Pad so a length-N_LAGS kernel covers lags LAG_MIN..LAG_MAX with
        # output_len == input_len.  LAG_MAX taps of past, |LAG_MIN| taps of future.
        pad_left, pad_right = LAG_MAX, max(0, -LAG_MIN)

        if variant == 'linear':
            # One conv, no nonlinearity — equivalent to the TRF lag matrix.
            self.net = nn.Sequential(
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, n_channels, kernel_size=N_LAGS, bias=True),
            )

        elif variant == 'separable':
            # Shared temporal bank (linear) → 1x1 spatial readout (linear).
            self.net = nn.Sequential(
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, hidden, kernel_size=N_LAGS, bias=True),
                nn.Conv1d(hidden, n_channels, kernel_size=1, bias=True),
            )

        elif variant == 'nonlinear':
            # [CHANGE 2] Replace BatchNorm1d with GroupNorm.
            # GroupNorm normalises independently per instance and is stable at
            # batch size 1 (full-trial eval) and for autocorrelated EEG sequences
            # where BatchNorm statistics vary across trials.
            assert hidden % 4 == 0, \
                f"HIDDEN={hidden} must be divisible by GroupNorm num_groups=4"
            layers = [
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, hidden, kernel_size=N_LAGS, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=hidden),   # was BatchNorm1d
                nn.GELU(),
            ]
            for _ in range(n_blocks - 1):
                layers += [
                    nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
                    nn.GroupNorm(num_groups=4, num_channels=hidden),   # was BatchNorm1d
                    nn.GELU(),
                ]
            layers += [nn.Conv1d(hidden, n_channels, kernel_size=1, bias=True)]
            self.net = nn.Sequential(*layers)

        else:
            raise ValueError(f'Unknown MODEL_VARIANT: {variant}')

    def forward(self, x):
        # x: (batch, n_features, T)  →  (batch, n_channels, T)
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════════
# Training / evaluation — leave-one-trial-out, matching the ridge protocol
# ════════════════════════════════════════════════════════════════════════════════

def _to_tensor(arr_2d):
    """(T, C) numpy → (1, C, T) float32 tensor on DEVICE (for full-trial inference)."""
    return torch.from_numpy(
        np.ascontiguousarray(arr_2d.T[None].astype(np.float32))).to(DEVICE)


def _pearsonr_channels(pred, target):
    """Mean Pearson r across EEG channels. pred/target: (1, n_ch, T) torch tensors."""
    p = pred.detach().cpu().numpy()[0].T    # (T, n_ch)
    t = target.detach().cpu().numpy()[0].T
    rs = [pearsonr(p[:, c], t[:, c])[0] for c in range(p.shape[1])]
    return float(np.nanmean(rs))


def train_one_fold(X_tr, Y_tr, n_features, n_channels):
    """Train on a list of training trials using windowed mini-batch updates.

    Differs from TRF_conv_1.train_one_fold only in the training loop [CHANGE 1]:
        • Training windows from all non-validation training trials are pooled into
          a single WindowDataset and fed through a DataLoader (batch_size=BATCH_SIZE,
          shuffle=True, drop_last=False).
        • Each forward+backward pass sees a mini-batch of windows, not a full trial.
        • Validation MSE is monitored on the windowed validation trial (fast proxy).
        • Validation Pearson r is computed on the FULL validation trial and drives
          both best-model checkpointing and early stopping [CHANGE 3].

    X_tr, Y_tr : lists of (T_i, n_features) / (T_i, n_channels) numpy arrays,
    each z-scored per-trial (identical to TRF_conv_1).

    Returns: model, train_mse_history, val_mse_history, val_r_history
    """
    model = StimToEEG(n_features, n_channels, MODEL_VARIANT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # Inner validation split: last training trial held out (unchanged from TRF_conv_1).
    inner_val_idx = len(X_tr) - 1
    tr_idx = list(range(inner_val_idx))

    # ── Full validation trial tensors — used for Pearson r and early stopping ──
    # [CHANGE 3] We keep the full-trial tensor for r computation; validation
    # windows are only used for the MSE monitoring metric.
    Xv_full = _to_tensor(X_tr[inner_val_idx])   # (1, n_features, T_val)
    Yv_full = _to_tensor(Y_tr[inner_val_idx])   # (1, n_channels, T_val)

    # ── [CHANGE 1] Build windowed training dataset ───────────────────────────────
    all_X_wins, all_Y_wins = [], []
    for i in tr_idx:
        xw, yw = _make_windows(X_tr[i], Y_tr[i])
        all_X_wins.append(xw)
        all_Y_wins.append(yw)
    X_train_wins = np.concatenate(all_X_wins, axis=0)   # (N_train, n_feat, W)
    Y_train_wins = np.concatenate(all_Y_wins, axis=0)   # (N_train, n_ch,   W)

    train_loader = DataLoader(
        WindowDataset(X_train_wins, Y_train_wins),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    # ── Windowed validation dataset — for MSE monitoring only ───────────────────
    Xv_wins, Yv_wins = _make_windows(X_tr[inner_val_idx], Y_tr[inner_val_idx])
    val_loader = DataLoader(
        WindowDataset(Xv_wins, Yv_wins),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    train_mse_history, val_mse_history, val_r_history = [], [], []
    best_val_r, best_state, since = -np.inf, None, 0

    for epoch in range(EPOCHS):
        # ── [CHANGE 1] Mini-batch training loop over windowed excerpts ───────────
        model.train()
        epoch_train_mse = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)   # (B, n_features, W)
            Y_batch = Y_batch.to(DEVICE)   # (B, n_channels, W)
            opt.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)
            loss.backward()
            opt.step()
            epoch_train_mse.append(loss.item())

        model.eval()
        with torch.no_grad():
            # Validation MSE on windowed val excerpts (monitoring only).
            val_mse_batches = []
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                val_mse_batches.append(loss_fn(model(X_batch), Y_batch).item())
            v_mse = float(np.mean(val_mse_batches))

            # [CHANGE 3] Pearson r on the FULL validation trial — checkpoint metric.
            pred_full = model(Xv_full)
            v_r = _pearsonr_channels(pred_full, Yv_full)

        train_mse_history.append(float(np.mean(epoch_train_mse)))
        val_mse_history.append(v_mse)
        val_r_history.append(v_r)

        # Best-model selection and early stopping by validation Pearson r.
        if v_r > best_val_r + 1e-6:
            best_val_r = v_r
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_mse_history, val_mse_history, val_r_history


def loocv_conv(X_all, Y_all, n_features, n_channels):
    """Leave-one-trial-out CV.  Returns (Y_pred_concat, Y_true_concat, lc_stats).

    The outer LOOCV structure is identical to TRF_conv_1.loocv_conv.
    Additions vs TRF_conv_1:
        [CHANGE 4] Per-fold debug output (window counts, held-out length).
        [CHANGE 5] Leakage guard: runtime assertion that no training trial is the
                   same Python object as the held-out test trial.
    """
    Y_pred_all, Y_true_all = [], []
    fold_train_mse, fold_val_mse, fold_val_r = [], [], []

    for i in range(len(X_all)):
        X_tr = [X_all[j] for j in range(len(X_all)) if j != i]
        Y_tr = [Y_all[j] for j in range(len(Y_all)) if j != i]

        # ── [CHANGE 5] Leakage guard ─────────────────────────────────────────────
        # X_all[i] must not appear in the training set (would indicate a bug in
        # list comprehension above or in shared array references).
        X_test_ref = X_all[i]
        for j, x in enumerate(X_tr):
            assert x is not X_test_ref, (
                f"Fold {i}: training trial {j} is the same object as the test trial "
                "— potential data leakage detected.")

        # ── [CHANGE 4] Debug diagnostics ─────────────────────────────────────────
        if DEBUG:
            n_pure_tr = len(X_tr) - 1          # inner-val trial excluded
            n_tr_wins = _count_windows([X_tr[k] for k in range(n_pure_tr)])
            n_val_wins = _count_windows([X_tr[-1]])
            heldout_n  = X_all[i].shape[0]
            print(f"  fold {i}:  train_trials={n_pure_tr}  "
                  f"train_windows={n_tr_wins}  "
                  f"val_windows={n_val_wins}  "
                  f"heldout_samples={heldout_n}")

        model, tr_mse, v_mse, v_r = train_one_fold(X_tr, Y_tr, n_features, n_channels)
        fold_train_mse.append(tr_mse)
        fold_val_mse.append(v_mse)
        fold_val_r.append(v_r)

        # Held-out evaluation on the full trial — unchanged from TRF_conv_1.
        model.eval()
        with torch.no_grad():
            pred = model(_to_tensor(X_all[i])).cpu().numpy()[0].T   # (T, n_ch)
        Y_pred_all.append(pred)
        Y_true_all.append(Y_all[i])

        if DEBUG:
            print(f"    → epochs={len(tr_mse)}, best_val_r={max(v_r):.4f}")

    # Aggregate per-fold learning curves (NaN-pad shorter folds).
    max_epochs = max(len(h) for h in fold_train_mse)

    def _pad(histories):
        mat = np.full((len(histories), max_epochs), np.nan)
        for k, h in enumerate(histories):
            mat[k, :len(h)] = h
        return mat

    lc_stats = SimpleNamespace(
        mean_train_mse = np.nanmean(_pad(fold_train_mse), axis=0),
        std_train_mse  = np.nanstd( _pad(fold_train_mse), axis=0),
        mean_val_mse   = np.nanmean(_pad(fold_val_mse),   axis=0),
        std_val_mse    = np.nanstd( _pad(fold_val_mse),   axis=0),
        mean_val_r     = np.nanmean(_pad(fold_val_r),     axis=0),
        std_val_r      = np.nanstd( _pad(fold_val_r),     axis=0),
        n_epochs       = max_epochs,
    )
    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all), lc_stats


def make_trf_result(Y_true, Y_pred, sensor_dim):
    n_ch   = Y_true.shape[1]
    r_vals = np.array([pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(n_ch)])
    return SimpleNamespace(r=eelbrain.NDVar(r_vals, dims=(sensor_dim,), name='r'))


# ════════════════════════════════════════════════════════════════════════════════
# Learning curve plot — unchanged from TRF_conv_1.py
# ════════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(lc_stats, subject, condition, variant, save_dir):
    """Two-panel learning-curve figure: MSE and validation Pearson r vs epoch."""
    epochs = np.arange(1, lc_stats.n_epochs + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"Learning curves — {subject} | {condition} | {variant} (windowed)",
        fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, lc_stats.mean_train_mse, label='Train MSE', color='steelblue')
    ax.fill_between(epochs,
                    lc_stats.mean_train_mse - lc_stats.std_train_mse,
                    lc_stats.mean_train_mse + lc_stats.std_train_mse,
                    alpha=0.3, color='steelblue')
    ax.plot(epochs, lc_stats.mean_val_mse, label='Val MSE', color='darkorange')
    ax.fill_between(epochs,
                    lc_stats.mean_val_mse - lc_stats.std_val_mse,
                    lc_stats.mean_val_mse + lc_stats.std_val_mse,
                    alpha=0.3, color='darkorange')
    ax.set_ylabel('MSE', fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Optimization loss (MSE)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.plot(epochs, lc_stats.mean_val_r, label='Val Pearson r', color='seagreen')
    ax.fill_between(epochs,
                    lc_stats.mean_val_r - lc_stats.std_val_r,
                    lc_stats.mean_val_r + lc_stats.std_val_r,
                    alpha=0.3, color='seagreen')
    ax.axhline(0, color='grey', linewidth=0.75, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Pearson r', fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Validation encoding quality — Pearson r (early-stopping criterion)',
                 fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fname = save_dir / f"{subject}_{condition}_{variant}_windowed_learning_curves.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if DEBUG:
        print(f"  Saved learning curves → {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# [CHANGE 6] Alignment plot — predicted vs actual EEG for one channel
# ════════════════════════════════════════════════════════════════════════════════

def plot_alignment(Y_true, Y_pred, subject, condition, variant, save_dir,
                   channel_idx=CHANNEL_IDX, sfreq=SFREQ):
    """Two-panel alignment figure for the concatenated held-out predictions.

    Panel 1: actual (black) and predicted (green) EEG for `channel_idx`.
    Panel 2: residual (actual − predicted).

    First 10 seconds of the concatenated held-out signal are displayed.
    """
    r_vals = np.array([pearsonr(Y_true[:, c], Y_pred[:, c])[0]
                       for c in range(Y_true.shape[1])])
    ch = channel_idx if channel_idx < Y_true.shape[1] else 0
    n_plot = min(len(Y_true), int(10 * sfreq))
    t_plot = np.arange(n_plot) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f'Predicted vs Actual EEG  |  {subject}, channel {ch}\n'
        f'CNN TRF (windowed)  ·  {condition}  ·  r = {r_vals[ch]:.3f}',
        fontsize=12, fontweight='bold')

    axes[0].plot(t_plot, Y_true[:n_plot, ch],
                 color='black', lw=0.7, label='Actual EEG (z-scored)')
    axes[0].plot(t_plot, Y_pred[:n_plot, ch],
                 color='seagreen', lw=0.9, alpha=0.85,
                 label=f'Predicted EEG  (r = {r_vals[ch]:.3f})')
    axes[0].set_ylabel('z-score')
    axes[0].set_title('Actual vs Predicted EEG  (CNN TRF windowed)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_plot, Y_true[:n_plot, ch] - Y_pred[:n_plot, ch],
                 color='darkorange', lw=0.7, label='Residual (actual − predicted)')
    axes[1].axhline(0, color='black', lw=0.6, linestyle='--')
    axes[1].set_ylabel('z-score')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Residual: Actual − Predicted')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = save_dir / f"{subject}_{condition}_{variant}_windowed_alignment_ch{ch}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if DEBUG:
        print(f"  Saved alignment plot → {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# Stimulus / IDyOM loading  (verbatim from TRF_ridge_3.py / TRF_conv_1.py)
# ════════════════════════════════════════════════════════════════════════════════

stim_mat = loadmat(constants.EEG_DIR / "dataStim.mat",
                   struct_as_record=False, squeeze_me=True)
stim        = stim_mat["stim"]
stim_fs     = int(stim.fs)
stimFeature = stim.data[0, :]

_g        = gcd(stim_fs, SFREQ)
stim_up   = SFREQ   // _g
stim_down = stim_fs // _g
unique_song_ids = np.unique(stim.stimIdxs)

idyom_pitch_mat = loadmat(constants.PITCH_SURPRISAL_FILE, squeeze_me=True)
idyom_onset_mat = loadmat(constants.ONSET_SURPRISAL_FILE, squeeze_me=True)

pitch_surprisal_data, pitch_entropy_data = {}, {}
onset_surprisal_data, onset_entropy_data = {}, {}
for song_id in unique_song_ids:
    song_name = f"audio{song_id}"
    if song_name not in idyom_pitch_mat:
        raise KeyError(f"{song_name} not found in PITCH_SURPRISAL_FILE")
    if song_name not in idyom_onset_mat:
        raise KeyError(f"{song_name} not found in ONSET_SURPRISAL_FILE")
    raw_pitch = np.asarray(idyom_pitch_mat[song_name])
    raw_onset = np.asarray(idyom_onset_mat[song_name])
    pitch_surprisal_data[song_id] = np.clip(raw_pitch[0], 0, IC_CLIP)
    pitch_entropy_data[song_id]   = raw_pitch[1]
    onset_surprisal_data[song_id] = np.clip(raw_onset[0], 0, IC_CLIP)
    onset_entropy_data[song_id]   = raw_onset[1]

os.makedirs(constants.SAVE_DIR, exist_ok=True)

_config = {
    'TMIN':                 TMIN,
    'TMAX':                 TMAX,
    'SFREQ':                SFREQ,
    'IC_CLIP':              IC_CLIP,
    'N_LAGS':               N_LAGS,
    'LAG_MIN':              LAG_MIN,
    'LAG_MAX':              LAG_MAX,
    'MODEL_VARIANT':        MODEL_VARIANT,
    'HIDDEN':               HIDDEN,
    'N_BLOCKS':             N_BLOCKS,
    'EPOCHS':               EPOCHS,
    'LR':                   LR,
    'WEIGHT_DECAY':         WEIGHT_DECAY,
    'EARLY_STOP_PATIENCE':  EARLY_STOP_PATIENCE,
    'WINDOW_SEC':           WINDOW_SEC,
    'HOP_SEC':              HOP_SEC,
    'BATCH_SIZE':           BATCH_SIZE,
    'WINDOW_SAMPLES':       WINDOW_SAMPLES,
    'HOP_SAMPLES':          HOP_SAMPLES,
    'CHANNEL_IDX':          CHANNEL_IDX,
    'DEBUG':                DEBUG,
    'SEED':                 SEED,
    'DEVICE':               str(DEVICE),
}
_config_path = constants.SAVE_DIR / 'config.csv'
with open(_config_path, 'w', newline='') as _f:
    _w = csv.writer(_f)
    _w.writerow(['parameter', 'value'])
    _w.writerows(_config.items())
print(f"Config saved → {_config_path}")

stim_pitch_surprisal_ndvars = {}
stim_pitch_entropy_ndvars   = {}
stim_onset_surprisal_ndvars = {}
stim_onset_entropy_ndvars   = {}


# ════════════════════════════════════════════════════════════════════════════════
# Main loop over subjects  (outer structure unchanged from TRF_conv_1.py)
# ════════════════════════════════════════════════════════════════════════════════

for SUBJECT in constants.SUBJECTS:

    eeg_data = eeg_func.load_subject_raw_eeg(
        constants.EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)

    preprocessed_trials = eeg_func.preprocess_eeg_trials(
        eeg_data,
        target_fs=SFREQ,
        lpf_hz=constants.HIGH_FREQUENCY,
        hpf_hz=constants.LOW_FREQUENCY,
        debug=DEBUG,
    )
    eeg_trial_lengths = [t.shape[0] for t in preprocessed_trials]

    raw    = eeg_func.create_mne_raw_from_preprocessed(
        preprocessed_trials, SFREQ, eeg_data['chanlocs'])
    events = eeg_func.create_eelbrain_events(raw)

    # ── Stimulus resampling + alignment (verbatim from ridge script) ──
    envelopes = []
    for i in range(len(events['event'])):
        env_raw = np.asarray(stimFeature[i], dtype=np.float64)
        n_eeg   = eeg_trial_lengths[i]
        env_resampled = sp_resample_poly(env_raw, stim_up, stim_down)
        n_min = min(len(env_resampled), n_eeg)
        diff  = len(env_resampled) - n_eeg
        if abs(diff) > 4 * SFREQ:
            warnings.warn(
                f"Trial {i}: unusually large stim/EEG mismatch "
                f"(stim={len(env_resampled)}, EEG={n_eeg}, diff={diff} smp).")
        env_resampled = env_resampled[:n_min]
        time_axis = eelbrain.UTS(0, 1 / SFREQ, n_min)
        envelopes.append(eelbrain.NDVar(env_resampled, (time_axis,)))

    events['envelope'] = envelopes
    events['onsets']   = [env.diff('time').clip(0) for env in envelopes]
    events['duration'] = eelbrain.Var([env.time.tstop for env in envelopes])
    events['eeg']      = eelbrain.load.mne.variable_length_epochs(
        events, 0, tstop='duration', decim=1, adjacency='auto')

    for i, stimulus_id in enumerate(events['event']):
        song_id = int(stimulus_id % 10) or 10
        if song_id in stim_pitch_surprisal_ndvars:
            continue
        midi_path = constants.MIDI_DIR / f"audio{song_id}.mid"
        time      = events['envelope'][i].time
        n_times   = time.nsamples
        stim_pitch_surprisal_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, pitch_surprisal_data[song_id], SFREQ, n_times),
            dims=(time,), name="pitch_surprisal")
        stim_pitch_entropy_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, pitch_entropy_data[song_id], SFREQ, n_times),
            dims=(time,), name="pitch_entropy")
        stim_onset_surprisal_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, onset_surprisal_data[song_id], SFREQ, n_times),
            dims=(time,), name="onset_surprisal")
        stim_onset_entropy_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, onset_entropy_data[song_id], SFREQ, n_times),
            dims=(time,), name="onset_entropy")

    events['pitch_surprisal'] = [stim_pitch_surprisal_ndvars[int(sid % 10) or 10] for sid in events['event']]
    events['pitch_entropy']   = [stim_pitch_entropy_ndvars[int(sid % 10) or 10]   for sid in events['event']]
    events['onset_surprisal'] = [stim_onset_surprisal_ndvars[int(sid % 10) or 10] for sid in events['event']]
    events['onset_entropy']   = [stim_onset_entropy_ndvars[int(sid % 10) or 10]   for sid in events['event']]

    # Convert every trial to numpy with EEG/stimulus alignment (as in ridge)
    trials = []
    for i in range(len(events['event'])):
        eeg_arr  = events['eeg'][i].get_data(('sensor', 'time')).T
        n = min(eeg_arr.shape[0], events['envelope'][i].x.shape[0])
        trials.append({
            'eeg':             eeg_arr[:n],
            'envelope':        events['envelope'][i].x[:n],
            'onsets':          events['onsets'][i].x[:n],
            'pitch_surprisal': events['pitch_surprisal'][i].x[:n],
            'pitch_entropy':   events['pitch_entropy'][i].x[:n],
            'onset_surprisal': events['onset_surprisal'][i].x[:n],
            'onset_entropy':   events['onset_entropy'][i].x[:n],
        })

    sensor_dim = events['eeg'][0].sensor
    n_channels = trials[0]['eeg'].shape[1]

    for condition, feature_keys in [
        ('acoustic',               ['envelope', 'onsets']),
        ('acoustic_and_surprisal', ['envelope', 'onsets', 'pitch_surprisal',
                                    'pitch_entropy', 'onset_surprisal', 'onset_entropy']),
    ]:
        X_all = [np.column_stack([zscore(t[k]) for k in feature_keys]) for t in trials]
        Y_all = [zscore(t['eeg']) for t in trials]
        n_features = len(feature_keys)

        print(f"\n  {SUBJECT} | {condition} | variant={MODEL_VARIANT} (windowed) "
              f"| features={n_features} channels={n_channels} "
              f"| window={WINDOW_SAMPLES}smp hop={HOP_SAMPLES}smp batch={BATCH_SIZE}")

        Y_pred, Y_true, lc_stats = loocv_conv(X_all, Y_all, n_features, n_channels)
        trf_conv = make_trf_result(Y_true, Y_pred, sensor_dim)

        # Result saving — same structure as TRF_conv_1; filename updated to avoid
        # overwriting conv_1 outputs.
        suffix  = 'acoustic_data' if condition == 'acoustic' else 'acoustic_and_surprisal_data'
        eelbrain.save.pickle(
            {'trf_cv': trf_conv, 'Y_pred': Y_pred, 'Y_true': Y_true},
            constants.SAVE_DIR / f'{SUBJECT}_{feature_keys}_conv2_windowed_{MODEL_VARIANT}_{suffix}.pkl')

        plot_learning_curves(lc_stats, SUBJECT, condition, MODEL_VARIANT, constants.SAVE_DIR)

        # [CHANGE 6] Alignment plot — new in TRF_conv_2_windowed.
        plot_alignment(Y_true, Y_pred, SUBJECT, condition, MODEL_VARIANT, constants.SAVE_DIR)

        print(f"  {SUBJECT} | {condition}: conv2_windowed ({MODEL_VARIANT}) "
              f"mean r = {trf_conv.r.mean():.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# MODIFICATION SUMMARY  (every section that differs from TRF_conv_1.py)
# ════════════════════════════════════════════════════════════════════════════════
#
# [CHANGE 1] — WINDOWING STRATEGY  (new: _make_windows, _count_windows,
#              WindowDataset, DataLoader in train_one_fold)
#   WHY: full-trial SGD has effective batch size 1, giving poor gradient
#        variance estimates and underutilising mini-batch optimisers for
#        nonlinear models.  Overlapping windows pool ~11k gradient steps per
#        epoch (vs ~18 trials), enabling stable optimisation without altering
#        the receptive field or evaluation protocol.
#
# [CHANGE 2] — GroupNorm replaces BatchNorm1d  (StimToEEG, nonlinear variant)
#   WHY: BatchNorm statistics computed over a batch of windows can be
#        inconsistent with statistics computed over a full trial at inference
#        time, and EEG autocorrelation makes inter-batch variance high.
#        GroupNorm normalises within each (instance, group) pair independently
#        of batch size and is unaffected by sequence autocorrelation.
#
# [CHANGE 3] — EARLY STOPPING on full-trial Pearson r  (train_one_fold)
#   WHY: window-MSE and full-trial Pearson r need not be monotone in each
#        other; using r on the full trial for checkpointing preserves exact
#        consistency with TRF_conv_1 and TRF_ridge_3's evaluation protocol.
#
# [CHANGE 4] — PER-FOLD DEBUG DIAGNOSTICS  (loocv_conv)
#   WHY: window counts are not obvious from trial lengths; printing them makes
#        it easy to catch trials that produce 0 windows and to verify that
#        training and validation data volumes are as expected.
#
# [CHANGE 5] — LEAKAGE GUARD ASSERTION  (loocv_conv)
#   WHY: explicit runtime check that the held-out test trial is not included in
#        the training list, guarding against accidental aliasing from future
#        refactors that might modify the list-comprehension split.
#
# [CHANGE 6] — ALIGNMENT PLOT  (new: plot_alignment, called in main loop)
#   WHY: visual sanity check.  The plot shows the predicted vs actual EEG for
#        CHANNEL_IDX on the first 10 s of the concatenated held-out signal,
#        making it easy to assess whether the model is learning a plausible
#        temporal structure or producing a flat/noisy prediction.
#
# UNCHANGED: stimulus loading, IDyOM preprocessing, per-trial z-scoring,
#            outer LOOCV loop structure, held-out inference (full-trial),
#            Pearson r computation, pickle keys, learning-curve plot format.
