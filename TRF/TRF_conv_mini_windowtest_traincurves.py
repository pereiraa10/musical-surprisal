"""
TRF_conv_mini_windowtest_traincurves.py
────────────────────────────────────────────────────────────────────────────────
Fork of TRF_conv_mini_windowtest.py that adds train-vs-validation divergence
tracking, so you can see whether a windowing config is training "optimally"
(train and val error move together / both plateau) or overfitting (train keeps
dropping while val flattens out or rises).

WHY THIS SCRIPT EXISTS
────────────────────────────────────────────────────────────────────────────────
TRF_conv_mini_windowtest.py already computes a val_mse_hist per config (a
random VAL_FRACTION split of the ~100-window subset) but only ever reports and
plots the FINAL val MSE — the per-epoch trajectory, and specifically how it
compares to the train trajectory over time, was being thrown away. That
trajectory is exactly what tells you whether a config is "running optimally":
- Train and val dropping together, gap staying small and roughly constant →
  the model is fitting real, generalizable structure at that window/overlap.
- Train continuing to drop while val plateaus or turns upward after an early
  minimum → classic overfitting; the model is starting to memorize
  window-specific noise rather than a shared feature→EEG mapping.
- Both flat near the baseline the whole time → underfitting / no signal found
  at that config (already partly visible via mse_ratio in the base script).

IMPORTANT CAVEAT — read before trusting the divergence signal
────────────────────────────────────────────────────────────────────────────────
VAL_FRACTION carves off a slice of an already-tiny ~100-window subset — with
the defaults below that's ~20 validation windows. Epoch-to-epoch val MSE at
that sample size is noisy; do not over-read small wiggles. What's meaningful
is the SHAPE over many epochs (a sustained upward trend after an early
minimum) not single-epoch jitter. A light rolling-average smoothing is applied
to the plotted val curve for readability (raw values are still used for the
numeric summary/verdict). This script is still a small-N go/no-go signal, not
a substitute for the full LOOCV protocol.

WHAT IS COPIED VERBATIM FROM TRF_conv_mini_windowtest.py
────────────────────────────────────────────────────────────────────────────────
- All config constants, receptive-field derivation, CausalPad/StimToEEG,
  windowing utilities, WindowDataset, stimulus/IDyOM loading, single-subject
  data prep, and the WINDOW_SECS_TO_TEST x OVERLAP_SECS_TO_TEST sweep loop.
- run_window_config's training loop is unchanged (same model, same optimizer,
  same fixed EPOCHS budget, no early stopping) — only what it COMPUTES and
  RETURNS from the histories it was already tracking is extended.

WHAT IS NEW
────────────────────────────────────────────────────────────────────────────────
- Per-config generalization-gap metrics computed from the existing
  train_mse_hist / val_mse_hist: final_gap, gap_ratio, best_val_mse,
  best_val_epoch, val_uptick_from_best (how much val has risen from its own
  best point by the end of training — the key overfitting signal), and a
  heuristic verdict string (STABLE / OVERFITTING / UNDERFIT-OR-FLAT) per
  config, in the same spirit as TRF_conv_overfit_check.py's PASS/AMBIGUOUS/FLAG.
  best_val_epoch/best_val_mse/val_uptick_from_best are computed from the
  SMOOTHED val curve (same SMOOTH_WINDOW used for plotting) — on the raw
  per-epoch curve, a single noisy low point in a ~20-window val split can look
  like a spurious "best epoch" and make a stable run look overfit or mask a
  real one.
- plot_train_val_curves: small-multiples figure (one panel per window size,
  like the base script) with BOTH train (solid) and smoothed val (dashed)
  curves per overlap, so divergence is visible directly.
- plot_gap_curves: a second figure plotting (val - train) per epoch per
  config — the divergence trajectory itself, which is easier to read than
  eyeballing two overlapping curves when the gap is subtle.
- Extended summary CSV with the new gap columns.

Run from musical-surprisal/TRF/:
    python TRF_conv_mini_windowtest_traincurves.py

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
# Config — receptive field (verbatim from TRF_conv_mini_windowtest.py)
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
SUBJECT   = 'Sub2'                      # single subject for fast iteration
CONDITION = 'acoustic'                  # 'acoustic' | 'acoustic_and_surprisal'

# ── Windowing sweep — EDIT THESE AND RE-RUN, no other code changes needed ──────
WINDOW_SECS_TO_TEST  = [7.0]   # window length, seconds
OVERLAP_SECS_TO_TEST = [3.0, 4.0, 4.3]                    # overlap between consecutive
                                                      # windows, seconds
                                                      # (hop_sec = window_sec - overlap_sec)

N_WINDOWS_SUBSET = 600     # cap per config; uses fewer + warns if unavailable
VAL_FRACTION     = 0.2     # fraction of the subset held out (quick trend only —
                           # ~20 windows at defaults; see caveat in docstring)
EPOCHS           = 300     # fixed budget; NO early stopping in this harness
BATCH_SIZE       = 16

# ── [NEW] Divergence-tracking knobs ──────────────────────────────────────────────
SMOOTH_WINDOW = 1   # rolling-average window (epochs) applied to val curves for
                     # plotting only, to reduce small-val-set noise; set to 1 to
                     # disable smoothing.
# Heuristic verdict thresholds (not a formal test — see docstring caveat):
#   val_uptick_from_best, relative to baseline_mse, above this -> "OVERFITTING"
OVERFIT_UPTICK_FRAC = 0.05
#   final_train_mse / baseline_mse above this (i.e. it never really learned) -> "UNDERFIT-OR-FLAT"
UNDERFIT_RATIO = 0.9

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
    # 'same'-length convolution with edge padding so the smoothed curve still
    # spans the full epoch range (useful for plotting alongside the raw train curve).
    padded = np.pad(series, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


# ════════════════════════════════════════════════════════════════════════════════
# Windowing utilities (verbatim from TRF_conv_mini_windowtest.py)
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


# ════════════════════════════════════════════════════════════════════════════════
# Model (verbatim from TRF_conv_mini_windowtest.py / TRF_conv_2_windowed.py)
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
# [NEW] Per-config mini-test with train/val divergence tracking
# ════════════════════════════════════════════════════════════════════════════════

def run_window_config(window_sec, overlap_sec, X_trials, Y_trials,
                       n_features, n_channels):
    """Same training procedure as TRF_conv_mini_windowtest.py's run_window_config.
    Extends the returned SimpleNamespace with generalization-gap metrics derived
    from the train_mse_hist / val_mse_hist that were already being recorded.
    """
    hop_sec = window_sec - overlap_sec
    if hop_sec <= 0:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be < window_sec ({window_sec}); "
            f"got hop_sec={hop_sec:.3f} <= 0.")

    window_samples = int(round(window_sec * SFREQ))
    hop_samples = max(1, int(round(hop_sec * SFREQ)))

    all_X_wins, all_Y_wins = [], []
    for X, Y in zip(X_trials, Y_trials):
        xw, yw = _make_windows(X, Y, window_samples, hop_samples)
        if xw is not None:
            all_X_wins.append(xw)
            all_Y_wins.append(yw)

    if not all_X_wins:
        print(f"  [SKIP] window={window_sec}s overlap={overlap_sec}s "
              f"(window_samples={window_samples}) — no trial is long enough.")
        return None

    X_wins = np.concatenate(all_X_wins, axis=0)
    Y_wins = np.concatenate(all_Y_wins, axis=0)
    n_available = X_wins.shape[0]

    idx = _rng.permutation(n_available)
    n_used = min(N_WINDOWS_SUBSET, n_available)
    if n_used < N_WINDOWS_SUBSET:
        print(f"  [WARN] window={window_sec}s overlap={overlap_sec}s: only "
              f"{n_available} windows available (< requested {N_WINDOWS_SUBSET}); "
              f"using all {n_available}.")
    idx = idx[:n_used]
    X_wins, Y_wins = X_wins[idx], Y_wins[idx]

    n_val = max(1, int(round(n_used * VAL_FRACTION))) if n_used > 4 else 0
    n_val = min(n_val, n_used - 1) if n_used > 1 else 0
    X_val, Y_val = X_wins[:n_val], Y_wins[:n_val]
    X_tr, Y_tr = X_wins[n_val:], Y_wins[n_val:]

    baseline_mse = float(np.mean(Y_wins ** 2))

    train_loader = DataLoader(
        WindowDataset(X_tr, Y_tr),
        batch_size=min(BATCH_SIZE, len(X_tr)),
        shuffle=True, drop_last=False)
    val_loader = None
    if n_val > 0:
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

    # ── [NEW] Generalization-gap metrics from the existing histories ────────────
    # Verdict-relevant quantities (best_val_epoch/mse, uptick) are computed from
    # the SMOOTHED val curve, not the raw per-epoch values: with only ~n_val
    # windows (often ~20), a single noisy epoch can look like a spurious "best"
    # point and make an otherwise-stable run look overfit (or vice versa). The
    # smoothing here must match SMOOTH_WINDOW used for plotting so the verdict
    # is consistent with what the figures show.
    if val_mse_hist:
        val_smoothed = _smooth(val_mse_hist, SMOOTH_WINDOW)
        best_val_epoch = int(np.argmin(val_smoothed))
        best_val_mse = float(val_smoothed[best_val_epoch])
        final_val_mse_smoothed = float(val_smoothed[-1])
        val_uptick_from_best = final_val_mse_smoothed - best_val_mse
        final_gap = final_val_mse - final_train_mse
        gap_ratio = final_val_mse / final_train_mse if final_train_mse > 0 else float('nan')

        if val_uptick_from_best > OVERFIT_UPTICK_FRAC * baseline_mse:
            verdict = 'OVERFITTING'
        elif final_train_mse / baseline_mse > UNDERFIT_RATIO:
            verdict = 'UNDERFIT-OR-FLAT'
        else:
            verdict = 'STABLE'
    else:
        best_val_epoch, best_val_mse = -1, float('nan')
        val_uptick_from_best = float('nan')
        final_gap = float('nan')
        gap_ratio = float('nan')
        verdict = 'NO-VAL-SPLIT'

    print(f"  window={window_sec:>4.1f}s overlap={overlap_sec:>3.1f}s "
          f"hop={hop_sec:.2f}s  n_avail={n_available:>5d} n_used={n_used:>4d}  "
          f"train_mse={final_train_mse:.4f}  val_mse={final_val_mse:.4f}  "
          f"gap={final_gap:.4f}  best_val@{best_val_epoch}={best_val_mse:.4f}  "
          f"uptick={val_uptick_from_best:.4f}  [{verdict}]")

    return SimpleNamespace(
        window_sec=window_sec, overlap_sec=overlap_sec, hop_sec=hop_sec,
        window_samples=window_samples, hop_samples=hop_samples,
        n_available=n_available, n_used=n_used,
        baseline_mse=baseline_mse,
        train_mse_hist=train_mse_hist, val_mse_hist=val_mse_hist,
        final_train_mse=final_train_mse, final_val_mse=final_val_mse,
        mse_ratio=final_train_mse / baseline_mse,
        best_val_mse=best_val_mse, best_val_epoch=best_val_epoch,
        val_uptick_from_best=val_uptick_from_best,
        final_gap=final_gap, gap_ratio=gap_ratio,
        verdict=verdict,
    )


# ════════════════════════════════════════════════════════════════════════════════
# [NEW] Reporting — train/val overlay + gap trajectory
# ════════════════════════════════════════════════════════════════════════════════

def save_summary_csv(results, save_dir):
    fname = save_dir / f"mini_windowtest_traincurves_summary_{SUBJECT}_{CONDITION}.csv"
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['window_sec', 'overlap_sec', 'hop_sec', 'n_available', 'n_used',
                    'baseline_mse', 'final_train_mse', 'final_val_mse', 'mse_ratio',
                    'best_val_mse', 'best_val_epoch', 'val_uptick_from_best',
                    'final_gap', 'gap_ratio', 'verdict'])
        for r in results:
            w.writerow([r.window_sec, r.overlap_sec, r.hop_sec, r.n_available,
                        r.n_used, r.baseline_mse, r.final_train_mse,
                        r.final_val_mse, r.mse_ratio, r.best_val_mse,
                        r.best_val_epoch, r.val_uptick_from_best,
                        r.final_gap, r.gap_ratio, r.verdict])
    print(f"Saved summary CSV -> {fname}")


def plot_train_val_curves(results, save_dir):
    """Small-multiples figure (one panel per window size): solid = train MSE,
    dashed = smoothed val MSE, per overlap. Divergence shows up as the dashed
    line pulling away from (typically above) the solid line."""
    windows = sorted(set(r.window_sec for r in results))
    fig, axes = plt.subplots(1, len(windows), figsize=(4.5 * len(windows), 4),
                              sharey=True)
    if len(windows) == 1:
        axes = [axes]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, w in zip(axes, windows):
        for c, r in enumerate([r for r in results if r.window_sec == w]):
            color = colors[c % len(colors)]
            ax.plot(r.train_mse_hist, color=color, linestyle='-',
                    label=f"overlap={r.overlap_sec}s train")
            if r.val_mse_hist:
                smoothed_val = _smooth(r.val_mse_hist, SMOOTH_WINDOW)
                ax.plot(smoothed_val, color=color, linestyle='--',
                        label=f"overlap={r.overlap_sec}s val (smoothed)")
            ax.axhline(r.baseline_mse, linestyle=':', color='grey', linewidth=0.7)
        ax.set_title(f"window={w}s")
        ax.set_xlabel('epoch')
        ax.set_yscale('log')
    axes[0].set_ylabel('MSE (log scale)\nsolid=train, dashed=val, dotted=baseline')
    axes[0].legend(fontsize=6.5, loc='upper right')
    fig.suptitle(f"Train vs val — {SUBJECT} | {CONDITION} | {MODEL_VARIANT}")
    plt.tight_layout()
    fname = save_dir / f"mini_windowtest_traincurves_{SUBJECT}_{CONDITION}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved train/val curves -> {fname}")


def plot_gap_curves(results, save_dir):
    """Second figure: (val - train) per epoch per config — the divergence
    trajectory itself. A flat-near-zero or slowly-shrinking line is healthy;
    a line that climbs steadily and doesn't turn back down is the overfitting
    signature to watch for."""
    windows = sorted(set(r.window_sec for r in results))
    fig, axes = plt.subplots(1, len(windows), figsize=(4.5 * len(windows), 4),
                              sharey=True)
    if len(windows) == 1:
        axes = [axes]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, w in zip(axes, windows):
        for c, r in enumerate([r for r in results if r.window_sec == w]):
            if not r.val_mse_hist:
                continue
            color = colors[c % len(colors)]
            n = min(len(r.train_mse_hist), len(r.val_mse_hist))
            gap = np.asarray(r.val_mse_hist[:n]) - np.asarray(r.train_mse_hist[:n])
            gap_smoothed = _smooth(gap, SMOOTH_WINDOW)
            ax.plot(gap_smoothed, color=color, label=f"overlap={r.overlap_sec}s")
        ax.axhline(0, linestyle=':', color='grey', linewidth=0.7)
        ax.set_title(f"window={w}s")
        ax.set_xlabel('epoch')
    axes[0].set_ylabel('val MSE − train MSE (smoothed)\n(rising = diverging)')
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Train/val gap — {SUBJECT} | {CONDITION} | {MODEL_VARIANT}")
    plt.tight_layout()
    fname = save_dir / f"mini_windowtest_gap_{SUBJECT}_{CONDITION}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved gap-trajectory figure -> {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# Stimulus / IDyOM loading (verbatim from TRF_conv_mini_windowtest.py)
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
# Single-subject data prep (verbatim from TRF_conv_mini_windowtest.py)
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

print(f"\n{SUBJECT} | {CONDITION} | variant={MODEL_VARIANT} "
      f"| {len(trials)} trials | features={n_features} channels={n_channels}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Run the window/overlap sweep
# ════════════════════════════════════════════════════════════════════════════════

results = []
for window_sec, overlap_sec in product(WINDOW_SECS_TO_TEST, OVERLAP_SECS_TO_TEST):
    if overlap_sec >= window_sec:
        print(f"  [SKIP] overlap {overlap_sec}s >= window {window_sec}s")
        continue
    r = run_window_config(window_sec, overlap_sec, X_all, Y_all,
                           n_features, n_channels)
    if r is not None:
        results.append(r)

if results:
    save_summary_csv(results, constants.SAVE_DIR)
    plot_train_val_curves(results, constants.SAVE_DIR)
    plot_gap_curves(results, constants.SAVE_DIR)

    best = min(results, key=lambda r: r.mse_ratio)
    print(f"\nBest mse_ratio: window={best.window_sec}s overlap={best.overlap_sec}s "
          f"-> ratio={best.mse_ratio:.3f} (verdict={best.verdict}, "
          f"gap={best.final_gap:.4f})")

    stable = [r for r in results if r.verdict == 'STABLE']
    if stable:
        best_stable = min(stable, key=lambda r: r.mse_ratio)
        print(f"Best STABLE config (train/val tracked without overfitting): "
              f"window={best_stable.window_sec}s overlap={best_stable.overlap_sec}s "
              f"-> ratio={best_stable.mse_ratio:.3f}")
    else:
        print("No config was labeled STABLE — check the OVERFITTING / "
              "UNDERFIT-OR-FLAT configs and consider adjusting EPOCHS, "
              "WEIGHT_DECAY, or OVERFIT_UPTICK_FRAC.")
else:
    print("No configs produced any windows — check WINDOW_SECS_TO_TEST against "
          "trial lengths for this subject.")


# ════════════════════════════════════════════════════════════════════════════════
# MODIFICATION SUMMARY vs TRF_conv_mini_windowtest.py
# ════════════════════════════════════════════════════════════════════════════════
#
# [NEW] Generalization-gap metrics per config computed from the train/val
#       histories that were already being recorded but previously discarded
#       after the loop: best_val_mse, best_val_epoch, val_uptick_from_best,
#       final_gap, gap_ratio, and a heuristic STABLE/OVERFITTING/
#       UNDERFIT-OR-FLAT verdict (thresholds: OVERFIT_UPTICK_FRAC, UNDERFIT_RATIO).
# [NEW] plot_train_val_curves: train (solid) + smoothed val (dashed) overlay
#       per config, small-multiples by window size.
# [NEW] plot_gap_curves: (val - train) trajectory per config — the divergence
#       signal directly, easier to read than two overlapping curves.
# [NEW] SMOOTH_WINDOW rolling-average smoothing applied to val curves only,
#       for plotting/verdict legibility given the small (~20-window) val split.
# [NEW] Extended summary CSV with the gap columns; prints the best STABLE
#       config (not just the lowest mse_ratio overall) at the end of the run.
#
# UNCHANGED: receptive-field constants, CausalPad/StimToEEG architecture,
#            windowing utilities, stimulus/IDyOM loading and per-trial feature
#            assembly, per-trial z-scoring, the training loop itself (same
#            optimizer/epochs/batch size — only what's computed from its
#            output histories is new).
