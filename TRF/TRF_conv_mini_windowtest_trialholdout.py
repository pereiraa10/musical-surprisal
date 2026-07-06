"""
TRF_conv_mini_windowtest_trialholdout.py
────────────────────────────────────────────────────────────────────────────────
Fork of TRF_conv_mini_windowtest_traincurves.py that fixes a leakage confound in
the window/overlap comparison and adds a genuine held-out-trial generalization
check (Pearson r + prediction-vs-actual plots), so the "large window / low
overlap" vs "small window / high overlap" question can be answered on data the
model never touched during training, not just on a randomly-interleaved slice
of the same window pool it trained on.

WHY THIS SCRIPT EXISTS — the leakage problem in the two earlier mini-test scripts
────────────────────────────────────────────────────────────────────────────────
TRF_conv_mini_windowtest.py / _traincurves.py build ONE pool of windows from ALL
trials, shuffle it, and split off VAL_FRACTION as "validation." At high overlap
(e.g. window=5s, overlap=4.3s -> hop=0.7s), consecutive windows from the same
trial differ by only 0.7s / 5s = 14% of their content. Because the split happens
AFTER pooling and shuffling, a "validation" window can end up just one hop away
in time from a "training" window from the very same trial — i.e. the model may
have already seen ~86%+ of that "held-out" window's content, just shifted by a
fraction of a second. This means configs with more overlap don't just have more
data — they also have progressively leakier validation, which would make them
look like they're generalizing better even if they aren't. This is very likely
part of what's driving the "small window + big overlap wins" side of the
U-shaped pattern observed across the sweeps so far, and it needs to be ruled
out before trusting that result.

THE FIX — hold out whole trials, not shuffled windows
────────────────────────────────────────────────────────────────────────────────
This script partitions trials themselves into TRAIN_TRIAL_INDICES and
HELD_OUT_TRIAL_INDICES (all trials belonging to the last HELD_OUT_N_SONGS
unique songs — see the SONG-level fix below, not just "last N trials by
index") BEFORE any windowing happens. Every training window, at every
window/overlap config, is
built ONLY from TRAIN_TRIAL_INDICES. Held-out trials never contribute a single
window to training, regardless of overlap — the leakage above is structurally
impossible here. Held-out trials are used two ways:
  1. Windowed (same window_sec/hop_sec as the training config) to get a
     per-epoch validation MSE curve for the STABLE/OVERFITTING verdict, exactly
     as in _traincurves.py, but now genuinely leak-free.
  2. Run through the trained model WHOLE (one forward pass on the entire
     continuous trial, like the production LOOCV scripts do) at the end of
     training, to compute a real per-channel Pearson r and to plot predicted
     vs actual EEG directly. This is the same metric used everywhere else in
     the project (ridge, conv_1, conv_2_windowed), so heldout_r here is at
     least roughly comparable in kind (not magnitude — single-subject,
     single-holdout-split, not LOOCV) to those headline numbers.

This script is the suggested INTERIM step between the ~100-window "does it
learn anything" mini-tests and a full 20-subject LOOCV run: single subject,
but with a genuine unseen-trial generalization check and real Pearson r,
which the earlier window-level-split scripts could not honestly provide.

WHAT CHANGED IN THE SWEEP GRID
────────────────────────────────────────────────────────────────────────────────
Overlap is now specified as OVERLAP_FRAC_TO_TEST (fraction of window_sec) rather
than absolute seconds. A fixed overlap in seconds means something very
different at window=2s (overlap=1.4s is 70% of the window) vs window=8s
(overlap=1.4s is 18% of the window) — comparing "low vs high overlap" across
window sizes is not apples-to-apples unless overlap is expressed relative to
the window it's cutting into. hop_sec = window_sec * (1 - overlap_frac).

WHAT IS COPIED VERBATIM FROM TRF_conv_mini_windowtest_traincurves.py
────────────────────────────────────────────────────────────────────────────────
- Receptive-field constants, CausalPad/StimToEEG, windowing utilities,
  WindowDataset, _smooth, stimulus/IDyOM loading, single-subject data prep.
- The per-epoch training loop and STABLE/OVERFITTING/UNDERFIT-OR-FLAT verdict
  logic (now computed on genuinely held-out-trial windows).

WHAT IS NEW
────────────────────────────────────────────────────────────────────────────────
- [FIX] Trial-level holdout is now BY SONG IDENTITY, not raw trial index.
  song_id is computed everywhere in this codebase as `stimulus_id % 10 (or 10)`
  — a strong hint there are only 10 unique songs, so a subject with e.g. 30
  trials has ~3 repetitions per song. The first version of this script held out
  the last HELD_OUT_N_TRIALS trials BY INDEX, which does NOT guarantee the
  held-out trials feature songs absent from training — a repeated song could
  appear in both sets under a different trial index. That leak produced
  implausible heldout_r values (~0.5-0.6, vs ~0.02-0.08 everywhere else in this
  project) — see check_trial_song_repeats.py, which confirmed the overlap.
  HELD_OUT_N_SONGS now selects whole songs to exclude; every trial replaying a
  held-out song, regardless of index, goes to the held-out set.
- Trial-level train/held-out split, computed after trials load.
- OVERLAP_FRAC_TO_TEST instead of absolute-second overlaps.
- Full-trial inference on held-out trial(s) at the end of each config's training:
  heldout_r (mean Pearson r across channels), heldout_mse, and a
  pred_std_ratio = std(prediction) / std(actual) on the plotted channel — a
  direct, cheap answer to "is it still just predicting the mean" (near 0 ==
  yes, near 1 == predicted variance matches real EEG variance).
- plot_alignment_per_config: per-config actual-vs-predicted + residual figure
  on the held-out trial (same style as production's plot_alignment).
- plot_alignment_comparison: ONE figure overlaying the actual held-out EEG
  trace with every config's prediction, so the different window/overlap
  configs can be visually compared against the same ground truth directly.
- Extended summary CSV with heldout_r, heldout_mse, pred_std_ratio.
- [NEW, 2026-07-02] Null checks on heldout_r itself, run automatically for every
  config: heldout_r_shift_null (circular-shift each held-out trial's true EEG by
  half its length — breaks correct timing, keeps each trial's own autocorrelation)
  and heldout_r_xshuffle_null (pair a trial's prediction with a DIFFERENT held-out
  trial's true EEG — breaks stimulus-response correspondence entirely; needs >=2
  held-out trials). If either null r exceeds NULL_R_FLAG_THRESH, the config is
  flagged "[NULL SURVIVES]" — evidence that heldout_r is not (only) measuring
  genuine stimulus-locked encoding. Added because the song-identity fix barely
  moved heldout_r for window=5.0s/overlap_frac=0.15 (0.5182 pre-fix -> 0.530
  post-fix), suggesting song repetition wasn't the only leak for that config.
- [NEW, 2026-07-02] Rescaled-overlay line in plot_alignment_per_config: the
  predicted trace rescaled to match the actual trace's std, plotted alongside
  the raw-scale prediction. Pearson r is scale-invariant, so a visually "flat"
  (low-amplitude) prediction can still post a high r if its shape is
  proportional to the real signal — this line isolates shape-matching from the
  amplitude mismatch instead of relying on eyeballing two differently-scaled
  traces.

Run from musical-surprisal/TRF/:
    python TRF_conv_mini_windowtest_trialholdout.py

NOTE: requires PyTorch. No GPU required. Requires the same dataset dependencies
as TRF_conv_2_windowed.py (constants.EEG_DIR .mat files, IDyOM surprisal .mat
files, eelbrain, mne, pretty_midi).
────────────────────────────────────────────────────────────────────────────────
"""

import os
import csv
import warnings
from math import gcd
from types import SimpleNamespace
from itertools import product

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly as sp_resample_poly
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func
import eelbrain


# ════════════════════════════════════════════════════════════════════════════════
# Config — receptive field (verbatim from TRF_conv_mini_windowtest_traincurves.py)
# ════════════════════════════════════════════════════════════════════════════════

TMIN  = -0.1
TMAX  = 0.600
SFREQ = 64
IC_CLIP = 15.0

N_LAGS  = int(round((TMAX - TMIN) * SFREQ)) + 1   # 46 taps
LAG_MIN = int(round(TMIN * SFREQ))                # -6
LAG_MAX = LAG_MIN + N_LAGS - 1                     # 39

MODEL_VARIANT = 'nonlinear'   # 'linear' | 'separable' | 'nonlinear'
HIDDEN        = 32
N_BLOCKS      = 2
LR            = 1e-3
WEIGHT_DECAY  = 1e-3

# ── Mini-test scope ──────────────────────────────────────────────────────────────
SUBJECT   = 'Sub2'
CONDITION = 'acoustic'                  # 'acoustic' | 'acoustic_and_surprisal'

# ── [FIX] Trial-level holdout, BY SONG not by raw trial index ───────────────────
# See docstring / check_trial_song_repeats.py: song_id repeats across trials in
# this dataset, so holding out by trial index does not guarantee held-out songs
# are absent from training. HELD_OUT_N_SONGS excludes whole songs (all their
# repeated trials) from training, regardless of trial index.
HELD_OUT_N_SONGS = 2

# ── [CHANGED] Windowing sweep — overlap as a FRACTION of window_sec ─────────────
WINDOW_SECS_TO_TEST   = [2.0, 5.0, 8.0]        # low / mid / high context length
OVERLAP_FRAC_TO_TEST  = [0.15, 0.5, 0.85]      # low / mid / high relative overlap
                                                 # hop_sec = window_sec * (1 - overlap_frac)

N_WINDOWS_SUBSET = 300     # cap per config, drawn ONLY from training trials
VAL_FRACTION_UNUSED = None  # (kept for reference; held-out trials replace this)
EPOCHS           = 300
BATCH_SIZE       = 16

SMOOTH_WINDOW = 15
OVERFIT_UPTICK_FRAC = 0.05
UNDERFIT_RATIO = 0.9

# ── [NEW] Held-out-trial evaluation / plotting ───────────────────────────────────
CHANNEL_IDX = 0            # channel shown in alignment plots
PLOT_SECONDS = 20          # seconds of the held-out trial to plot
PRED_STD_FLAG_THRESH = 0.15  # pred_std_ratio below this -> "looks like it's still
                              # predicting ~the mean" flag in the printed summary

# ── [NEW] Null checks on heldout_r itself ────────────────────────────────────────
# Same idea as diagnostic_d2_shuffle.py, scoped to THIS script's held-out trials:
# deliberately break the correct correspondence between prediction and ground
# truth and see if r survives. If it does, heldout_r is not (only) measuring
# genuine stimulus-locked encoding, regardless of what the song-level holdout
# fixed. See the response this was added for: heldout_r=0.53 at window=5.0s/
# overlap_frac=0.15 was ~unchanged by the song fix (was 0.5182 pre-fix), which
# means song repetition probably wasn't the only thing going on for this config.
NULL_R_FLAG_THRESH = 0.10   # |null r| above this -> flag as a likely residual leak

DEBUG = True
SEED  = 0


def _select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _select_device()
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)
_rng = np.random.RandomState(SEED)


def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _smooth(series, window):
    if window <= 1 or len(series) < window:
        return np.asarray(series, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(series, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


# ════════════════════════════════════════════════════════════════════════════════
# Windowing utilities (verbatim)
# ════════════════════════════════════════════════════════════════════════════════

def _make_windows(X, Y, window_samples, hop_samples):
    T = X.shape[0]
    starts = list(range(0, T - window_samples + 1, hop_samples))
    if len(starts) == 0:
        return None, None
    X_wins = np.stack([X[s:s + window_samples].T for s in starts], axis=0)
    Y_wins = np.stack([Y[s:s + window_samples].T for s in starts], axis=0)
    return X_wins, Y_wins


class WindowDataset(Dataset):
    def __init__(self, X_wins, Y_wins):
        self.X = torch.from_numpy(X_wins.astype(np.float32))
        self.Y = torch.from_numpy(Y_wins.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _to_tensor(arr_2d):
    """(T, C) numpy -> (1, C, T) float32 tensor on DEVICE, for whole-trial inference."""
    return torch.from_numpy(
        np.ascontiguousarray(arr_2d.T[None].astype(np.float32))).to(DEVICE)


# ════════════════════════════════════════════════════════════════════════════════
# Model (verbatim)
# ════════════════════════════════════════════════════════════════════════════════

class CausalPad(nn.Module):
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
        pad_left, pad_right = LAG_MAX, max(0, -LAG_MIN)

        if variant == 'linear':
            self.net = nn.Sequential(
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, n_channels, kernel_size=N_LAGS, bias=True),
            )
        elif variant == 'separable':
            self.net = nn.Sequential(
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, hidden, kernel_size=N_LAGS, bias=True),
                nn.Conv1d(hidden, n_channels, kernel_size=1, bias=True),
            )
        elif variant == 'nonlinear':
            assert hidden % 4 == 0, f"HIDDEN={hidden} must be divisible by GroupNorm num_groups=4"
            layers = [
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, hidden, kernel_size=N_LAGS, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=hidden),
                nn.GELU(),
            ]
            for _ in range(n_blocks - 1):
                layers += [
                    nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
                    nn.GroupNorm(num_groups=4, num_channels=hidden),
                    nn.GELU(),
                ]
            layers += [nn.Conv1d(hidden, n_channels, kernel_size=1, bias=True)]
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f'Unknown MODEL_VARIANT: {variant}')

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════════
# [NEW] Per-config run: train on TRAIN trials only, evaluate on HELD-OUT trials
# ════════════════════════════════════════════════════════════════════════════════

def run_window_config(window_sec, overlap_frac, X_train_trials, Y_train_trials,
                       X_heldout_trials, Y_heldout_trials, n_features, n_channels):
    hop_sec = window_sec * (1 - overlap_frac)
    if hop_sec <= 0:
        raise ValueError(f"overlap_frac={overlap_frac} gives hop_sec<=0 for "
                          f"window_sec={window_sec}")

    window_samples = int(round(window_sec * SFREQ))
    hop_samples = max(1, int(round(hop_sec * SFREQ)))

    # ── Training windows: TRAIN trials only ──────────────────────────────────────
    all_X_wins, all_Y_wins = [], []
    for X, Y in zip(X_train_trials, Y_train_trials):
        xw, yw = _make_windows(X, Y, window_samples, hop_samples)
        if xw is not None:
            all_X_wins.append(xw)
            all_Y_wins.append(yw)
    if not all_X_wins:
        print(f"  [SKIP] window={window_sec}s overlap_frac={overlap_frac} "
              f"— no training trial is long enough.")
        return None
    X_wins = np.concatenate(all_X_wins, axis=0)
    Y_wins = np.concatenate(all_Y_wins, axis=0)
    n_available = X_wins.shape[0]

    idx = _rng.permutation(n_available)
    n_used = min(N_WINDOWS_SUBSET, n_available)
    if n_used < N_WINDOWS_SUBSET:
        print(f"  [WARN] window={window_sec}s overlap_frac={overlap_frac}: only "
              f"{n_available} training windows available (< requested "
              f"{N_WINDOWS_SUBSET}); using all {n_available}.")
    idx = idx[:n_used]
    X_tr, Y_tr = X_wins[idx], Y_wins[idx]
    baseline_mse = float(np.mean(Y_tr ** 2))

    # ── Validation windows: HELD-OUT trials only (leak-free by construction) ────
    hv_X_wins, hv_Y_wins = [], []
    for X, Y in zip(X_heldout_trials, Y_heldout_trials):
        xw, yw = _make_windows(X, Y, window_samples, hop_samples)
        if xw is not None:
            hv_X_wins.append(xw)
            hv_Y_wins.append(yw)
    X_val = np.concatenate(hv_X_wins, axis=0) if hv_X_wins else None
    Y_val = np.concatenate(hv_Y_wins, axis=0) if hv_Y_wins else None

    train_loader = DataLoader(
        WindowDataset(X_tr, Y_tr),
        batch_size=min(BATCH_SIZE, len(X_tr)),
        shuffle=True, drop_last=False)
    val_loader = None
    if X_val is not None and len(X_val) > 0:
        val_loader = DataLoader(
            WindowDataset(X_val, Y_val),
            batch_size=min(BATCH_SIZE, len(X_val)),
            shuffle=False, drop_last=False)

    model = StimToEEG(n_features, n_channels, MODEL_VARIANT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_mse_hist, val_mse_hist = [], []
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, Yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        train_mse_hist.append(float(np.mean(batch_losses)))

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                v_losses = [loss_fn(model(Xb.to(DEVICE)), Yb.to(DEVICE)).item()
                            for Xb, Yb in val_loader]
            val_mse_hist.append(float(np.mean(v_losses)))

    final_train_mse = train_mse_hist[-1]
    final_val_mse = val_mse_hist[-1] if val_mse_hist else float('nan')

    if val_mse_hist:
        val_smoothed = _smooth(val_mse_hist, SMOOTH_WINDOW)
        best_val_epoch = int(np.argmin(val_smoothed))
        best_val_mse = float(val_smoothed[best_val_epoch])
        val_uptick_from_best = float(val_smoothed[-1]) - best_val_mse
        final_gap = final_val_mse - final_train_mse
        if val_uptick_from_best > OVERFIT_UPTICK_FRAC * baseline_mse:
            verdict = 'OVERFITTING'
        elif final_train_mse / baseline_mse > UNDERFIT_RATIO:
            verdict = 'UNDERFIT-OR-FLAT'
        else:
            verdict = 'STABLE'
    else:
        best_val_epoch, best_val_mse = -1, float('nan')
        val_uptick_from_best, final_gap = float('nan'), float('nan')
        verdict = 'NO-VAL-SPLIT'

    # ── [NEW] Whole-trial inference on held-out trial(s): real r + prediction ───
    model.eval()
    heldout_pred_full, heldout_true_full = [], []
    with torch.no_grad():
        for X, Y in zip(X_heldout_trials, Y_heldout_trials):
            pred = model(_to_tensor(X)).cpu().numpy()[0].T  # (T, n_channels)
            heldout_pred_full.append(pred)
            heldout_true_full.append(Y)
    Y_pred_concat = np.concatenate(heldout_pred_full)
    Y_true_concat = np.concatenate(heldout_true_full)
    heldout_r = float(np.nanmean([
        pearsonr(Y_true_concat[:, c], Y_pred_concat[:, c])[0]
        for c in range(Y_true_concat.shape[1])]))
    heldout_mse = float(np.mean((Y_true_concat - Y_pred_concat) ** 2))

    pred_ch = Y_pred_concat[:, CHANNEL_IDX]
    true_ch = Y_true_concat[:, CHANNEL_IDX]
    pred_std_ratio = float(np.std(pred_ch) / (np.std(true_ch) + 1e-12))
    flat_flag = pred_std_ratio < PRED_STD_FLAG_THRESH

    # ── [NEW] Null checks: does heldout_r survive deliberately broken correspondence? ──
    def _r_from_trials(pred_trials, true_trials):
        n_mins = [min(p.shape[0], t.shape[0]) for p, t in zip(pred_trials, true_trials)]
        Yp = np.concatenate([p[:n] for p, n in zip(pred_trials, n_mins)])
        Yt = np.concatenate([t[:n] for t, n in zip(true_trials, n_mins)])
        return float(np.nanmean([
            pearsonr(Yt[:, c], Yp[:, c])[0] for c in range(Yt.shape[1])]))

    # Null 1 — circular shift: shift each held-out trial's TRUE EEG by half its
    # own length before scoring. Preserves each signal's own autocorrelation
    # structure but destroys correct stimulus-locked timing.
    shifted_true = [np.roll(Y, Y.shape[0] // 2, axis=0) for Y in heldout_true_full]
    heldout_r_shift_null = _r_from_trials(heldout_pred_full, shifted_true)

    # Null 2 — cross-trial pairing: score trial i's prediction against a
    # DIFFERENT held-out trial's true EEG. Destroys stimulus-response
    # correspondence entirely. Needs >=2 held-out trials.
    if len(heldout_pred_full) >= 2:
        n_ho = len(heldout_pred_full)
        perm = _rng.permutation(n_ho)
        for k in range(n_ho):
            if perm[k] == k:
                swap = (k + 1) % n_ho
                perm[k], perm[swap] = perm[swap], perm[k]
        xshuffled_true = [heldout_true_full[perm[i]] for i in range(n_ho)]
        heldout_r_xshuffle_null = _r_from_trials(heldout_pred_full, xshuffled_true)
    else:
        heldout_r_xshuffle_null = float('nan')

    null_survives = (abs(heldout_r_shift_null) > NULL_R_FLAG_THRESH or
                      (not np.isnan(heldout_r_xshuffle_null) and
                       abs(heldout_r_xshuffle_null) > NULL_R_FLAG_THRESH))

    print(f"  window={window_sec:>4.1f}s overlap_frac={overlap_frac:>4.2f} "
          f"hop={hop_sec:.2f}s  n_avail={n_available:>5d} n_used={n_used:>4d}  "
          f"train_mse={final_train_mse:.4f}  gap={final_gap:.4f}  [{verdict}]  "
          f"heldout_r={heldout_r:+.4f}  pred_std_ratio={pred_std_ratio:.3f}"
          f"{'  [FLAT/MEAN-LIKE]' if flat_flag else ''}\n"
          f"      null checks -> shift_null_r={heldout_r_shift_null:+.4f}  "
          f"xshuffle_null_r={heldout_r_xshuffle_null:+.4f}"
          f"{'  [NULL SURVIVES -> LIKELY RESIDUAL LEAK]' if null_survives else '  [nulls collapse as expected]'}")

    return SimpleNamespace(
        window_sec=window_sec, overlap_frac=overlap_frac, hop_sec=hop_sec,
        window_samples=window_samples, hop_samples=hop_samples,
        n_available=n_available, n_used=n_used,
        baseline_mse=baseline_mse,
        train_mse_hist=train_mse_hist, val_mse_hist=val_mse_hist,
        final_train_mse=final_train_mse, final_val_mse=final_val_mse,
        mse_ratio=final_train_mse / baseline_mse,
        best_val_mse=best_val_mse, best_val_epoch=best_val_epoch,
        val_uptick_from_best=val_uptick_from_best, final_gap=final_gap,
        verdict=verdict,
        heldout_r=heldout_r, heldout_mse=heldout_mse,
        heldout_r_shift_null=heldout_r_shift_null,
        heldout_r_xshuffle_null=heldout_r_xshuffle_null,
        null_survives=null_survives,
        pred_std_ratio=pred_std_ratio, flat_flag=flat_flag,
        heldout_pred_ch=pred_ch, heldout_true_ch=true_ch,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════════

def save_summary_csv(results, save_dir):
    fname = save_dir / f"mini_windowtest_trialholdout_summary_{SUBJECT}_{CONDITION}.csv"
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['window_sec', 'overlap_frac', 'hop_sec', 'n_available', 'n_used',
                    'baseline_mse', 'final_train_mse', 'mse_ratio', 'final_gap',
                    'verdict', 'heldout_r', 'heldout_mse', 'pred_std_ratio', 'flat_flag',
                    'heldout_r_shift_null', 'heldout_r_xshuffle_null', 'null_survives'])
        for r in results:
            w.writerow([r.window_sec, r.overlap_frac, r.hop_sec, r.n_available,
                        r.n_used, r.baseline_mse, r.final_train_mse, r.mse_ratio,
                        r.final_gap, r.verdict, r.heldout_r, r.heldout_mse,
                        r.pred_std_ratio, r.flat_flag,
                        r.heldout_r_shift_null, r.heldout_r_xshuffle_null, r.null_survives])
    print(f"Saved summary CSV -> {fname}")


def plot_alignment_per_config(r, save_dir, sfreq=SFREQ):
    n_plot = min(len(r.heldout_true_ch), int(PLOT_SECONDS * sfreq))
    t_plot = np.arange(n_plot) / sfreq
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"{SUBJECT} ch{CHANNEL_IDX} — window={r.window_sec}s "
                 f"overlap_frac={r.overlap_frac} — heldout r={r.heldout_r:.3f}, "
                 f"verdict={r.verdict}, null(shift/xshuffle)="
                 f"{r.heldout_r_shift_null:+.3f}/{r.heldout_r_xshuffle_null:+.3f}",
                 fontsize=10, fontweight='bold')
    axes[0].plot(t_plot, r.heldout_true_ch[:n_plot], color='black', lw=0.8, label='Actual')
    axes[0].plot(t_plot, r.heldout_pred_ch[:n_plot], color='seagreen', lw=1.0,
                 alpha=0.85, label=f'Predicted, raw scale (r={r.heldout_r:.3f})')
    # [NEW] Rescaled overlay: predicted trace rescaled to match the actual
    # trace's std. Pearson r is scale-invariant, so a "flat-looking" prediction
    # can still post a high r if its SHAPE is proportional to the real signal —
    # this line isolates shape-matching from the amplitude mismatch so you can
    # visually judge which one you're looking at.
    scale = np.std(r.heldout_true_ch) / (np.std(r.heldout_pred_ch) + 1e-12)
    pred_rescaled = r.heldout_pred_ch[:n_plot] * scale
    axes[0].plot(t_plot, pred_rescaled, color='crimson', lw=0.9, linestyle='--',
                 alpha=0.7, label=f'Predicted, rescaled ×{scale:.2f} '
                                   f'(pred_std_ratio={r.pred_std_ratio:.3f})')
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('z-score')
    axes[0].grid(alpha=0.3)
    axes[1].plot(t_plot, r.heldout_true_ch[:n_plot] - r.heldout_pred_ch[:n_plot],
                 color='darkorange', lw=0.7)
    axes[1].axhline(0, color='black', lw=0.6, linestyle='--')
    axes[1].set_ylabel('residual')
    axes[1].set_xlabel('time (s)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fname = save_dir / (f"trialholdout_alignment_{SUBJECT}_{CONDITION}_"
                         f"w{r.window_sec}_of{r.overlap_frac}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved alignment plot -> {fname}")


def plot_alignment_comparison(results, save_dir, sfreq=SFREQ):
    """One figure: actual held-out EEG plus every config's prediction overlaid,
    so the different window/overlap configs can be compared directly against
    the same ground truth — the plot requested to check whether any config is
    doing more than predicting ~the mean."""
    n_plot = min(len(results[0].heldout_true_ch), int(PLOT_SECONDS * sfreq))
    t_plot = np.arange(n_plot) / sfreq
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_plot, results[0].heldout_true_ch[:n_plot], color='black', lw=1.1,
            label='Actual EEG', zorder=10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for c, r in enumerate(results):
        ax.plot(t_plot, r.heldout_pred_ch[:n_plot], color=colors[c % len(colors)],
                lw=1.0, alpha=0.85,
                label=f"win={r.window_sec}s ovl={r.overlap_frac} (r={r.heldout_r:.3f})")
    ax.axhline(0, color='grey', lw=0.6, linestyle=':')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('z-score')
    ax.set_title(f"{SUBJECT} ch{CHANNEL_IDX} — prediction comparison across "
                 f"window/overlap configs (held-out trial)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = save_dir / f"trialholdout_alignment_comparison_{SUBJECT}_{CONDITION}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison alignment plot -> {fname}")


def plot_train_val_curves(results, save_dir):
    windows = sorted(set(r.window_sec for r in results))
    fig, axes = plt.subplots(1, len(windows), figsize=(4.5 * len(windows), 4), sharey=True)
    if len(windows) == 1:
        axes = [axes]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, w in zip(axes, windows):
        for c, r in enumerate([r for r in results if r.window_sec == w]):
            color = colors[c % len(colors)]
            ax.plot(r.train_mse_hist, color=color, linestyle='-',
                    label=f"ovl={r.overlap_frac} train")
            if r.val_mse_hist:
                ax.plot(_smooth(r.val_mse_hist, SMOOTH_WINDOW), color=color,
                        linestyle='--', label=f"ovl={r.overlap_frac} val (held-out trial)")
            ax.axhline(r.baseline_mse, linestyle=':', color='grey', linewidth=0.7)
        ax.set_title(f"window={w}s")
        ax.set_xlabel('epoch')
        ax.set_yscale('log')
    axes[0].set_ylabel('MSE (log)\nsolid=train, dashed=held-out val')
    axes[0].legend(fontsize=6.5)
    fig.suptitle(f"Train vs held-out-trial val — {SUBJECT} | {CONDITION} | {MODEL_VARIANT}")
    plt.tight_layout()
    fname = save_dir / f"mini_windowtest_trialholdout_curves_{SUBJECT}_{CONDITION}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved train/val curves -> {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# Stimulus / IDyOM loading (verbatim)
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
    raw_pitch = np.asarray(idyom_pitch_mat[song_name])
    raw_onset = np.asarray(idyom_onset_mat[song_name])
    pitch_surprisal_data[song_id] = np.clip(raw_pitch[0], 0, IC_CLIP)
    pitch_entropy_data[song_id]   = raw_pitch[1]
    onset_surprisal_data[song_id] = np.clip(raw_onset[0], 0, IC_CLIP)
    onset_entropy_data[song_id]   = raw_onset[1]

os.makedirs(constants.SAVE_DIR, exist_ok=True)

stim_pitch_surprisal_ndvars = {}
stim_pitch_entropy_ndvars   = {}
stim_onset_surprisal_ndvars = {}
stim_onset_entropy_ndvars   = {}


# ════════════════════════════════════════════════════════════════════════════════
# Single-subject data prep (verbatim)
# ════════════════════════════════════════════════════════════════════════════════

eeg_data = eeg_func.load_subject_raw_eeg(
    constants.EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)

preprocessed_trials = eeg_func.preprocess_eeg_trials(
    eeg_data, target_fs=SFREQ,
    lpf_hz=constants.HIGH_FREQUENCY, hpf_hz=constants.LOW_FREQUENCY, debug=DEBUG)
eeg_trial_lengths = [t.shape[0] for t in preprocessed_trials]

raw    = eeg_func.create_mne_raw_from_preprocessed(
    preprocessed_trials, SFREQ, eeg_data['chanlocs'])
events = eeg_func.create_eelbrain_events(raw)

envelopes = []
for i in range(len(events['event'])):
    env_raw = np.asarray(stimFeature[i], dtype=np.float64)
    n_eeg   = eeg_trial_lengths[i]
    env_resampled = sp_resample_poly(env_raw, stim_up, stim_down)
    n_min = min(len(env_resampled), n_eeg)
    diff  = len(env_resampled) - n_eeg
    if abs(diff) > 4 * SFREQ:
        warnings.warn(f"Trial {i}: unusually large stim/EEG mismatch "
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

n_channels = trials[0]['eeg'].shape[1]

feature_keys = (['envelope', 'onsets'] if CONDITION == 'acoustic' else
                ['envelope', 'onsets', 'pitch_surprisal', 'pitch_entropy',
                 'onset_surprisal', 'onset_entropy'])
n_features = len(feature_keys)

X_all = [np.column_stack([zscore(t[k]) for k in feature_keys]) for t in trials]
Y_all = [zscore(t['eeg']) for t in trials]

# ── [FIX] Trial-level split — BY SONG IDENTITY, done ONCE before any windowing ──
trial_song_ids = [int(sid % 10) or 10 for sid in events['event']]
unique_songs_in_order = list(dict.fromkeys(trial_song_ids))  # first-appearance order
if len(unique_songs_in_order) <= HELD_OUT_N_SONGS:
    raise ValueError(f"Only {len(unique_songs_in_order)} unique songs available; "
                      f"can't hold out {HELD_OUT_N_SONGS}. Lower HELD_OUT_N_SONGS.")
HELD_OUT_SONG_IDS = set(unique_songs_in_order[-HELD_OUT_N_SONGS:])
TRAIN_TRIAL_INDICES = [i for i, s in enumerate(trial_song_ids) if s not in HELD_OUT_SONG_IDS]
HELD_OUT_TRIAL_INDICES = [i for i, s in enumerate(trial_song_ids) if s in HELD_OUT_SONG_IDS]
assert set(trial_song_ids[i] for i in TRAIN_TRIAL_INDICES).isdisjoint(HELD_OUT_SONG_IDS), (
    "song-level leak guard failed — a training trial still shares a song with "
    "a held-out trial; do not proceed without fixing this.")

X_train_trials = [X_all[i] for i in TRAIN_TRIAL_INDICES]
Y_train_trials = [Y_all[i] for i in TRAIN_TRIAL_INDICES]
X_heldout_trials = [X_all[i] for i in HELD_OUT_TRIAL_INDICES]
Y_heldout_trials = [Y_all[i] for i in HELD_OUT_TRIAL_INDICES]

print(f"\n{SUBJECT} | {CONDITION} | variant={MODEL_VARIANT} "
      f"| {len(trials)} trials total, {len(unique_songs_in_order)} unique songs "
      f"| held-out songs {sorted(HELD_OUT_SONG_IDS)} -> "
      f"{len(TRAIN_TRIAL_INDICES)} train trials (indices {TRAIN_TRIAL_INDICES}), "
      f"{len(HELD_OUT_TRIAL_INDICES)} held-out trials (indices {HELD_OUT_TRIAL_INDICES}) "
      f"| features={n_features} channels={n_channels}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Run the window/overlap sweep
# ════════════════════════════════════════════════════════════════════════════════

results = []
for window_sec, overlap_frac in product(WINDOW_SECS_TO_TEST, OVERLAP_FRAC_TO_TEST):
    r = run_window_config(window_sec, overlap_frac,
                           X_train_trials, Y_train_trials,
                           X_heldout_trials, Y_heldout_trials,
                           n_features, n_channels)
    if r is not None:
        results.append(r)
        plot_alignment_per_config(r, constants.SAVE_DIR)

if results:
    save_summary_csv(results, constants.SAVE_DIR)
    plot_train_val_curves(results, constants.SAVE_DIR)
    plot_alignment_comparison(results, constants.SAVE_DIR)

    by_r = sorted(results, key=lambda r: r.heldout_r, reverse=True)
    print("\nRanked by held-out Pearson r (the metric that actually matters here):")
    for r in by_r:
        flag = ' [FLAT/MEAN-LIKE]' if r.flat_flag else ''
        null_flag = ' [NULL SURVIVES]' if r.null_survives else ''
        print(f"  window={r.window_sec:>4.1f}s overlap_frac={r.overlap_frac:>4.2f}  "
              f"heldout_r={r.heldout_r:+.4f}  verdict={r.verdict}  "
              f"pred_std_ratio={r.pred_std_ratio:.3f}{flag}  "
              f"shift_null={r.heldout_r_shift_null:+.3f} xshuffle_null={r.heldout_r_xshuffle_null:+.3f}{null_flag}")
else:
    print("No configs produced any windows.")


# ════════════════════════════════════════════════════════════════════════════════
# MODIFICATION SUMMARY vs TRF_conv_mini_windowtest_traincurves.py
# ════════════════════════════════════════════════════════════════════════════════
#
# [NEW] Trial-level train/held-out split (HELD_OUT_N_SONGS), computed once
#       after loading, BEFORE any windowing — removes the near-duplicate
#       train/val leakage that high-overlap configs were previously exposed to
#       when validation was a random slice of the same shuffled window pool.
# [FIX] Holdout is by SONG IDENTITY, not raw trial index (see top-of-file note
#       under WHAT IS NEW) — the index-based version leaked repeated songs
#       between train and held-out sets and produced implausible heldout_r.
# [CHANGED] Overlap specified as OVERLAP_FRAC_TO_TEST (fraction of window_sec)
#       instead of absolute seconds, so "low/high overlap" means the same
#       relative thing across different window lengths.
# [NEW] Whole-trial inference on the held-out trial(s) at the end of training:
#       heldout_r (real Pearson r, comparable in kind to production headline
#       numbers), heldout_mse, pred_std_ratio (cheap "is it still predicting
#       ~the mean" check: ratio of predicted to actual signal std on the
#       plotted channel).
# [NEW] plot_alignment_per_config / plot_alignment_comparison: predicted vs
#       actual EEG on the held-out trial, per config and as one overlaid
#       comparison figure across all configs.
# [NEW] Final ranking printed by heldout_r, not mse_ratio — mse_ratio is a
#       training-fit number; heldout_r is the number that answers whether a
#       config actually generalizes.
#
# UNCHANGED: receptive-field constants, CausalPad/StimToEEG architecture,
#            windowing utilities, stimulus/IDyOM loading, per-trial z-scoring,
#            the STABLE/OVERFITTING/UNDERFIT-OR-FLAT verdict logic (now fed
#            genuinely held-out windows instead of a leaky shuffled split).
