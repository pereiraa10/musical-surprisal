"""
TRF_conv_mini_windowtest.py
────────────────────────────────────────────────────────────────────────────────
Fast, small-N diagnostic harness for comparing windowing/segmentation techniques
BEFORE committing to a full LOOCV cohort run with TRF_conv_2_windowed.py.

WHY THIS SCRIPT EXISTS
────────────────────────────────────────────────────────────────────────────────
TRF_conv_2_windowed.py (nonlinear variant) has been observed to mostly predict ~0
(i.e. collapse toward the per-trial mean) rather than tracking held-out EEG. Before
re-running the full 20-subject x 2-condition LOOCV pipeline for every windowing idea,
this script trains the SAME architecture on a small, fixed-size subset of windows
(~100 by default) from a SINGLE subject, across a sweep of window lengths and overlap
amounts, and reports whether training loss drops meaningfully below the trivial
"predict zero" baseline for each configuration.

IMPORTANT FRAMING — read before interpreting results
────────────────────────────────────────────────────────────────────────────────
TRF_ridge_3.py (the closed-form ground truth) only achieves held-out r ~ 0.02-0.03
on this dataset, i.e. even the "correct" linear model explains roughly 0.05-0.1% of
EEG variance. Against that, "predict ~0" is a strong MSE minimum, not obviously a
bug. The bar for a windowing config to clear here is NOT "produces dramatic
predictions" — it's "final train MSE drops measurably below the baseline_mse
(=mean(Y**2), i.e. the loss of predicting exactly zero) within a fixed epoch
budget." Use the printed/plotted mse_ratio (final_train_mse / baseline_mse) to
compare configs: closer to 0 is better, ~1.0 means "learned nothing beyond the
mean," and use it to shortlist 1-2 configs to validate on more subjects before
touching the full LOOCV pipeline.

This script does NOT reproduce the LOOCV protocol and its r values are NOT
comparable to TRF_ridge_3 / TRF_conv_1 / TRF_conv_2_windowed headline numbers. It
trains on a small held-in subset with a fixed epoch budget and no early stopping —
purely a fast go/no-go signal for the architecture + windowing combination.

WHAT IS COPIED VERBATIM FROM TRF_conv_2_windowed.py
────────────────────────────────────────────────────────────────────────────────
- Receptive-field constants (TMIN, TMAX, SFREQ, IC_CLIP -> N_LAGS/LAG_MIN/LAG_MAX)
- CausalPad, StimToEEG (all three variants) — identical architecture, so results
  here are informative about the actual production model, not a toy stand-in.
- zscore(), _make_windows(), _count_windows(), WindowDataset — identical windowing
  utilities (window_samples / hop_samples are supplied per-config here instead of
  fixed constants).
- Stimulus / IDyOM loading and per-trial feature assembly (the block that builds
  `trials` from `dataStim.mat` + IDyOM surprisal/entropy .mat files + MIDI onsets).
  Restricted to a single subject instead of looping `constants.SUBJECTS`.

WHAT IS NEW
────────────────────────────────────────────────────────────────────────────────
- WINDOW_SECS_TO_TEST / OVERLAP_SECS_TO_TEST sweep (constants below — edit and
  re-run; no code changes needed to try a new window/overlap combination).
- N_WINDOWS_SUBSET cap: each config uses at most this many windows (pooled across
  ALL trials of SUBJECT, shuffled with a fixed seed), for fast iteration. If a
  config's window/overlap combo can't produce N_WINDOWS_SUBSET windows from the
  available trials, it uses whatever is available and prints a clear warning —
  it does NOT fabricate data to hit the target count. (Sanity-checked with
  synthetic 10-trial/~60s data: the 8s and 10s window configs only yield
  ~60-80 windows total from one subject's trials, well short of 100 — expect
  the warning to fire for those configs unless you widen SUBJECT's trial count
  or lower N_WINDOWS_SUBSET.)
- No LOOCV, no early stopping: fixed EPOCHS budget, train/val is a simple
  random split of the small subset (VAL_FRACTION), purely to see a validation
  trend — not a rigorous held-out estimate.
- Baseline-relative reporting: every config reports mse_ratio = final_train_mse /
  baseline_mse(=mean(Y**2)) so "did it learn anything at all" is a number, not a
  visual judgment call.
- Summary CSV + a small-multiples loss-curve figure across all configs.

Run from musical-surprisal/TRF/:
    python TRF_conv_mini_windowtest.py

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
# Config — receptive field (verbatim from TRF_conv_2_windowed.py)
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
CONDITION = 'acoustic_and_surprisal'    # 'acoustic' | 'acoustic_and_surprisal'

# ── [NEW] Windowing sweep — EDIT THESE AND RE-RUN, no other code changes needed ──
WINDOW_SECS_TO_TEST  = [2.0, 3.0, 5.0, 8.0, 10.0]   # window length, seconds
OVERLAP_SECS_TO_TEST = [0.7, 1.0]                    # overlap between consecutive
                                                      # windows, seconds
                                                      # (hop_sec = window_sec - overlap_sec)

N_WINDOWS_SUBSET = 100     # cap per config; uses fewer + warns if unavailable
VAL_FRACTION     = 0.2     # fraction of the subset held out (quick trend only)
EPOCHS           = 300     # fixed budget; NO early stopping in this harness
BATCH_SIZE       = 16

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


# ════════════════════════════════════════════════════════════════════════════════
# Windowing utilities (verbatim from TRF_conv_2_windowed.py, window/hop passed
# explicitly per-config instead of fixed module constants)
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
# Model (verbatim from TRF_conv_2_windowed.py — identical architecture so results
# are informative about the real production model, not a toy stand-in)
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
# [NEW] Per-config mini-test
# ════════════════════════════════════════════════════════════════════════════════

def run_window_config(window_sec, overlap_sec, X_trials, Y_trials,
                       n_features, n_channels):
    """Build a ~N_WINDOWS_SUBSET-window dataset for one (window_sec, overlap_sec)
    config, train StimToEEG on it for a fixed epoch budget (no early stopping),
    and report whether training loss drops below the predict-zero baseline.

    Returns a SimpleNamespace with the config, window-availability info, and
    train/val loss histories.
    """
    hop_sec = window_sec - overlap_sec
    if hop_sec <= 0:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be < window_sec ({window_sec}); "
            f"got hop_sec={hop_sec:.3f} <= 0.")

    window_samples = int(round(window_sec * SFREQ))
    hop_samples = max(1, int(round(hop_sec * SFREQ)))

    # Pool windows from ALL trials of SUBJECT; no window crosses a trial boundary.
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

    # Shuffle (fixed seed) and cap at N_WINDOWS_SUBSET.
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

    baseline_mse = float(np.mean(Y_wins ** 2))  # loss of predicting exactly zero

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

    print(f"  window={window_sec:>4.1f}s overlap={overlap_sec:>3.1f}s "
          f"hop={hop_sec:.2f}s  n_avail={n_available:>5d} n_used={n_used:>4d}  "
          f"train_mse={final_train_mse:.4f}  baseline={baseline_mse:.4f}  "
          f"ratio={final_train_mse / baseline_mse:.3f}  val_mse={final_val_mse:.4f}")

    return SimpleNamespace(
        window_sec=window_sec, overlap_sec=overlap_sec, hop_sec=hop_sec,
        window_samples=window_samples, hop_samples=hop_samples,
        n_available=n_available, n_used=n_used,
        baseline_mse=baseline_mse,
        train_mse_hist=train_mse_hist, val_mse_hist=val_mse_hist,
        final_train_mse=final_train_mse, final_val_mse=final_val_mse,
        mse_ratio=final_train_mse / baseline_mse,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════════

def save_summary_csv(results, save_dir):
    fname = save_dir / f"mini_windowtest_summary_{SUBJECT}_{CONDITION}.csv"
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['window_sec', 'overlap_sec', 'hop_sec', 'n_available', 'n_used',
                    'baseline_mse', 'final_train_mse', 'final_val_mse', 'mse_ratio'])
        for r in results:
            w.writerow([r.window_sec, r.overlap_sec, r.hop_sec, r.n_available,
                        r.n_used, r.baseline_mse, r.final_train_mse,
                        r.final_val_mse, r.mse_ratio])
    print(f"Saved summary CSV -> {fname}")


def plot_loss_curves(results, save_dir):
    windows = sorted(set(r.window_sec for r in results))
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4),
                              sharey=True)
    if len(windows) == 1:
        axes = [axes]
    for ax, w in zip(axes, windows):
        for r in [r for r in results if r.window_sec == w]:
            ax.plot(r.train_mse_hist, label=f"overlap={r.overlap_sec}s")
            ax.axhline(r.baseline_mse, linestyle='--', color='grey', linewidth=0.8)
        ax.set_title(f"window={w}s")
        ax.set_xlabel('epoch')
        ax.set_yscale('log')
    axes[0].set_ylabel('train MSE (log scale)\n(dashed = predict-zero baseline)')
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Mini window-test — {SUBJECT} | {CONDITION} | {MODEL_VARIANT}")
    plt.tight_layout()
    fname = save_dir / f"mini_windowtest_curves_{SUBJECT}_{CONDITION}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved loss-curve figure -> {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# Stimulus / IDyOM loading (verbatim from TRF_conv_2_windowed.py)
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
# Single-subject data prep (restricted version of the TRF_conv_2_windowed.py loop —
# runs for SUBJECT only, not constants.SUBJECTS)
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
# [NEW] Run the window/overlap sweep
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
    plot_loss_curves(results, constants.SAVE_DIR)

    best = min(results, key=lambda r: r.mse_ratio)
    print(f"\nBest mse_ratio: window={best.window_sec}s overlap={best.overlap_sec}s "
          f"-> ratio={best.mse_ratio:.3f} (n_used={best.n_used})")
else:
    print("No configs produced any windows — check WINDOW_SECS_TO_TEST against "
          "trial lengths for this subject.")


# ════════════════════════════════════════════════════════════════════════════════
# MODIFICATION SUMMARY vs TRF_conv_2_windowed.py
# ════════════════════════════════════════════════════════════════════════════════
#
# [NEW] Single-subject scope (SUBJECT constant) instead of looping constants.SUBJECTS.
# [NEW] WINDOW_SECS_TO_TEST x OVERLAP_SECS_TO_TEST sweep (see run_window_config).
# [NEW] N_WINDOWS_SUBSET cap with graceful degradation + warning when a config
#       can't reach the target count from the available trials.
# [NEW] No LOOCV, no early stopping — fixed EPOCHS, simple train/val split of the
#       small subset, purely for a fast trend signal.
# [NEW] baseline_mse / mse_ratio reporting: turns "is it learning" into a number,
#       given how weak the true signal is here (ridge r ~ 0.02-0.03).
# [NEW] Per-config CSV summary + small-multiples loss-curve figure.
#
# UNCHANGED: receptive-field constants, CausalPad/StimToEEG architecture,
#            windowing utilities, stimulus/IDyOM loading and per-trial feature
#            assembly, per-trial z-scoring.
