"""
TRF_conv_overfit_check.py
────────────────────────────────────────────────────────────────────────────────
Pure optimization-capacity sanity check for the StimToEEG architecture used in
TRF_conv_2_windowed.py — separate from, and a prerequisite to, the windowing
comparison in TRF_conv_mini_windowtest.py.

WHY THIS SCRIPT EXISTS
────────────────────────────────────────────────────────────────────────────────
TRF_conv_2_windowed.py (nonlinear variant) has been observed to mostly predict ~0.
There are two very different explanations for that, and they call for opposite
fixes:

  (a) The true signal is genuinely tiny (ridge itself only reaches r ~ 0.02-0.03,
      i.e. ~0.05-0.1% of variance explained) and "predict near-zero" is close to
      the MSE-optimal thing to do given realistic regularization/early-stopping.
      Fix: better windowing/segmentation, more data, or accept the ceiling.

  (b) There is an optimization/architecture bug (dead GELU units, GroupNorm
      misbehaving at small batch size, learning rate too low, gradient not
      flowing through some layer) that prevents the model from fitting even
      signal it CAN in principle represent.
      Fix: architecture/training-loop bug, not a data problem.

This script isolates (b). It takes a small, fixed subset of windows (~100 by
default), disables every form of regularization (weight_decay=0, no early
stopping, no held-out split — the ENTIRE subset is the "training set" to
memorize), and runs many epochs of full-batch gradient descent. If train MSE
does not drop substantially below the predict-zero baseline, that is evidence
of (b), independent of whatever windowing scheme is used elsewhere.

A NOTE ON WHAT "OVERFIT" MEANS HERE
────────────────────────────────────────────────────────────────────────────────
StimToEEG is a convolutional network with weights shared across every time step
and every window — it is not a lookup table, so driving MSE to exactly 0 on
~100 windows (each ~hundreds of time steps x 64 channels) is not guaranteed or
even necessarily expected, unlike a fully-connected memorizer with one free
parameter per data point. The useful signal is not "did it reach exactly zero"
but "did it drop SUBSTANTIALLY and steadily below the predict-zero baseline, or
did it get stuck at/near baseline_mse for the whole run." The latter is the red
flag worth escalating as an architecture/optimization bug.

WHAT IS COPIED VERBATIM FROM TRF_conv_2_windowed.py
────────────────────────────────────────────────────────────────────────────────
- Receptive-field constants (TMIN, TMAX, SFREQ, IC_CLIP -> N_LAGS/LAG_MIN/LAG_MAX)
- CausalPad, StimToEEG (all three variants) — identical architecture.
- zscore(), _make_windows(), WindowDataset.
- Stimulus / IDyOM loading and per-trial feature assembly, restricted to a single
  subject.

WHAT IS NEW / DIFFERENT FROM PRODUCTION TRAINING
────────────────────────────────────────────────────────────────────────────────
- WEIGHT_DECAY = 0.0 (production default is 1e-3) — regularization is exactly
  the thing we're trying to rule out as a confound here.
- No early stopping, no train/val split — the whole ~100-window subset is used
  for gradient updates every epoch (this is intentional; we are testing whether
  the model CAN fit this data at all, not whether it generalizes).
- BATCH_SIZE = N_WINDOWS_SUBSET (full-batch gradient descent) for a clean,
  low-noise loss trajectory.
- LR is higher than production (see LR below) to reach a verdict faster; this
  is a diagnostic, not a training run whose checkpoint you'd ever use.
- EPOCHS is much larger than production (no early stopping to cut it short).

Run from musical-surprisal/TRF/:
    python TRF_conv_overfit_check.py

NOTE: requires PyTorch. No GPU required. Requires the same dataset dependencies
as TRF_conv_2_windowed.py.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
from math import gcd

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly as sp_resample_poly

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
# Config
# ════════════════════════════════════════════════════════════════════════════════

TMIN  = -0.1
TMAX  = 0.600
SFREQ = 64
IC_CLIP = 15.0

N_LAGS  = int(round((TMAX - TMIN) * SFREQ)) + 1
LAG_MIN = int(round(TMIN * SFREQ))
LAG_MAX = LAG_MIN + N_LAGS - 1

MODEL_VARIANT = 'nonlinear'   # match the variant that's collapsing in production
HIDDEN        = 32
N_BLOCKS      = 2

SUBJECT   = 'Sub2'
CONDITION = 'acoustic_and_surprisal'

# ── Windowing for the subset to memorize (edit if you want to test a different
#    window/overlap here too, but the point of this script is architecture, not
#    windowing — TRF_conv_mini_windowtest.py is where you sweep those) ──────────
WINDOW_SEC = 5.0
HOP_SEC    = 1.0
N_WINDOWS_SUBSET = 100

# ── [DIFFERENT FROM PRODUCTION] — see docstring for why ──────────────────────────
LR           = 5e-3      # production uses 1e-3; higher here to reach a verdict faster
WEIGHT_DECAY = 0.0        # production uses 1e-3; disabled to remove this confound
EPOCHS       = 1500       # no early stopping; run long enough to see the trend clearly
BATCH_SIZE   = N_WINDOWS_SUBSET   # full-batch

# Report every this many epochs.
LOG_EVERY = 50

# Verdict thresholds (heuristic, not a formal test): if final_mse / baseline_mse
# is below PASS_RATIO, treat this as "capacity looks fine." If it never drops
# below STUCK_RATIO of baseline, treat this as "looks stuck — investigate."
PASS_RATIO  = 0.5
STUCK_RATIO = 0.95

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
            assert hidden % 4 == 0
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
# Single-subject data prep
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


# ════════════════════════════════════════════════════════════════════════════════
# Build the fixed ~N_WINDOWS_SUBSET-window "memorize this" set
# ════════════════════════════════════════════════════════════════════════════════

window_samples = int(round(WINDOW_SEC * SFREQ))
hop_samples    = max(1, int(round(HOP_SEC * SFREQ)))

all_X_wins, all_Y_wins = [], []
for X, Y in zip(X_all, Y_all):
    xw, yw = _make_windows(X, Y, window_samples, hop_samples)
    if xw is not None:
        all_X_wins.append(xw)
        all_Y_wins.append(yw)

X_wins = np.concatenate(all_X_wins, axis=0)
Y_wins = np.concatenate(all_Y_wins, axis=0)
n_available = X_wins.shape[0]

idx = _rng.permutation(n_available)[:min(N_WINDOWS_SUBSET, n_available)]
X_wins, Y_wins = X_wins[idx], Y_wins[idx]
n_used = X_wins.shape[0]
if n_used < N_WINDOWS_SUBSET:
    print(f"[WARN] only {n_available} windows available; using all {n_used} "
          f"(< requested {N_WINDOWS_SUBSET}).")

baseline_mse = float(np.mean(Y_wins ** 2))

print(f"\n{SUBJECT} | {CONDITION} | variant={MODEL_VARIANT} | "
      f"n_used={n_used} windows | window={window_samples}smp hop={hop_samples}smp | "
      f"baseline_mse(predict-zero)={baseline_mse:.4f}\n")

loader = DataLoader(WindowDataset(X_wins, Y_wins),
                     batch_size=min(BATCH_SIZE, n_used),
                     shuffle=True, drop_last=False)


# ════════════════════════════════════════════════════════════════════════════════
# Train — full-batch-ish, no weight decay, no early stopping, fixed EPOCHS
# ════════════════════════════════════════════════════════════════════════════════

model = StimToEEG(n_features, n_channels, MODEL_VARIANT).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

mse_history = []
for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
        opt.zero_grad()
        pred = model(Xb)
        loss = loss_fn(pred, Yb)
        loss.backward()
        opt.step()
        batch_losses.append(loss.item())
    epoch_mse = float(np.mean(batch_losses))
    mse_history.append(epoch_mse)

    if DEBUG and (epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1):
        print(f"  epoch {epoch:>5d}  train_mse={epoch_mse:.5f}  "
              f"ratio={epoch_mse / baseline_mse:.3f}")

final_mse = mse_history[-1]
final_ratio = final_mse / baseline_mse
min_ratio = min(mse_history) / baseline_mse

print(f"\nFinal train MSE: {final_mse:.5f}  (baseline {baseline_mse:.5f}, "
      f"ratio {final_ratio:.3f})")
print(f"Best ratio reached during training: {min_ratio:.3f}")

if min_ratio <= PASS_RATIO:
    verdict = ("PASS — the model drove loss well below the predict-zero baseline "
               "on this subset. The full-training 'predicts ~0' behavior is more "
               "likely a regularization/early-stopping/data-scale issue than a "
               "fundamental optimization bug. Proceed to compare windowing "
               "configs with TRF_conv_mini_windowtest.py.")
elif min_ratio <= STUCK_RATIO:
    verdict = ("AMBIGUOUS — some improvement over baseline but not a lot. "
               "Try more epochs / higher LR before concluding either way.")
else:
    verdict = ("FLAG — loss never meaningfully left the predict-zero baseline "
               "even with no regularization, no early stopping, and a small "
               "fixed subset. This points toward an optimization/architecture "
               "issue (check GroupNorm behavior on this batch size, GELU "
               "saturation, LR, gradient flow) rather than a data/windowing "
               "problem. Worth checking before investing in windowing changes.")

print(f"\nVerdict: {verdict}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mse_history, color='steelblue', label='train MSE (full subset)')
ax.axhline(baseline_mse, color='grey', linestyle='--', label='predict-zero baseline')
ax.set_yscale('log')
ax.set_xlabel('epoch')
ax.set_ylabel('MSE (log scale)')
ax.set_title(f'Overfit check — {SUBJECT} | {CONDITION} | {MODEL_VARIANT} | '
             f'n={n_used} windows')
ax.legend(fontsize=9)
plt.tight_layout()
fname = constants.SAVE_DIR / f"overfit_check_{SUBJECT}_{CONDITION}_{MODEL_VARIANT}.png"
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved loss curve -> {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# MODIFICATION SUMMARY vs TRF_conv_2_windowed.py
# ════════════════════════════════════════════════════════════════════════════════
#
# [NEW] Single-subject, single fixed-size window subset (no LOOCV, no per-subject
#       loop, no multi-condition loop).
# [NEW] WEIGHT_DECAY=0, no early stopping, no train/val split — the entire subset
#       is trained on every epoch. This is intentional: the question is whether
#       the model CAN fit this data given unlimited optimization budget, not
#       whether it generalizes.
# [NEW] Full-batch-ish training (BATCH_SIZE = N_WINDOWS_SUBSET) for a clean loss
#       trajectory.
# [NEW] baseline_mse / ratio reporting + heuristic PASS / AMBIGUOUS / FLAG verdict
#       based on how far training loss drops from the predict-zero baseline.
#
# UNCHANGED: receptive-field constants, CausalPad/StimToEEG architecture,
#            windowing utilities, stimulus/IDyOM loading and per-trial feature
#            assembly, per-trial z-scoring.
