"""
TRF_conv_1.py
────────────────────────────────────────────────────────────────────────────────
Deep-learning counterpart to TRF_ridge_3.py.

The linear TRF in TRF_ridge_3.py is, mathematically, a single linear layer whose
weights are a Toeplitz lag matrix — i.e. a depthwise temporal convolution with a
fixed receptive field.  This script re-expresses that mapping as a 1-D temporal
ConvNet trained by SGD, so that nonlinearity and cross-channel weight sharing can
be added on top of a baseline that provably reproduces the ridge solution.

EVERYTHING upstream of "features → EEG" is reused unchanged from the ridge
pipeline: the same per-trial preprocessing (eeg_functions.preprocess_eeg_trials),
the same stimulus resampling block, the same feature set, the same z-scoring, and
— critically — the SAME leave-one-trial-out evaluation protocol with Pearson r
computed on concatenated held-out predictions per channel.  This isolation is the
whole point: the only thing that differs from ridge is the stimulus→EEG model, so
held-out r is directly comparable.

────────────────────────────────────────────────────────────────────────────────
The MODEL_VARIANT ladder  (climb one rung at a time)
────────────────────────────────────────────────────────────────────────────────
    'linear'       1 Conv1d, receptive field == TRF lag window, NO nonlinearity.
                   This is the sanity-check rung: trained by SGD it should
                   recover (approximately) the ridge TRF.  If it does not match
                   ridge here, do not trust any deeper rung.

    'separable'    Shared temporal filter bank (Conv1d, n_features → H) followed
                   by a 1x1 Conv1d readout (H → n_channels).  Still linear, but
                   factorizes the (lags*features × channels) weight matrix into
                   shared temporal filters × per-channel spatial readout.  Tests
                   whether the low-rank factorization alone costs accuracy.

    'nonlinear'    The requested model: a stack of temporal Conv1d layers with
                   BatchNorm + GELU between them, producing a SHARED latent
                   feature bank, then a 1x1 Conv1d that projects the shared bank
                   to all EEG channels at once.  Channels share the temporal
                   feature extractor and differ only in their readout weights —
                   the deep analog of factorizing the ridge weight matrix.

Run from musical-surprisal/TRF/:
    python TRF_conv_1.py

NOTE: requires PyTorch (pip install torch).  No GPU required; runs on CPU.
"""

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
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt

import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func
import eelbrain


# ─── Config (mirrors TRF_ridge_3.py where shared) ──────────────────────────────
TMIN  = -0.1   # seconds  — receptive field start (pre-stimulus)
TMAX  = 0.600  # seconds  — receptive field end   (post-stimulus)
SFREQ = 64     # Hz after resampling
IC_CLIP = 15.0

# Receptive field in taps, derived exactly like build_lag_matrix in the ridge code
N_LAGS  = int(round((TMAX - TMIN) * SFREQ)) + 1   # e.g. 0.7s * 64 + 1 = 45 taps
LAG_MIN = int(round(TMIN * SFREQ))                # e.g. -6  (pre-stimulus taps)
LAG_MAX = LAG_MIN + N_LAGS - 1                    # e.g.  38 (post-stimulus taps)

# ── Deep-learning knobs ──
MODEL_VARIANT = 'nonlinear'   # 'linear' | 'separable' | 'nonlinear'
HIDDEN        = 32            # width of the shared temporal feature bank
N_BLOCKS      = 2            # nonlinear conv blocks (only used by 'nonlinear')
EPOCHS        = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-3          # the SGD-era analog of ridge alpha
EARLY_STOP_PATIENCE = 25      # epochs without held-out improvement before stopping

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

torch.manual_seed(SEED)   # seeds CPU + CUDA + MPS generators in recent torch
np.random.seed(SEED)


def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


# ════════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════════
#
# Tensor convention: (batch=1, channels=features, time).  We treat each whole
# trial as one "batch" element of length T_i and convolve along time.  The
# receptive field is centered on the TRF lag window by asymmetric padding so the
# model can use both pre-stimulus (LAG_MIN<0) and post-stimulus context, exactly
# like the ridge Toeplitz matrix.

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
        # output_len == input_len.  LAG_MAX taps of past, |LAG_MIN| of future.
        pad_left, pad_right = LAG_MAX, max(0, -LAG_MIN)

        if variant == 'linear':
            # One conv, no nonlinearity, no bottleneck → equivalent to the TRF.
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
            # Stacked nonlinear temporal blocks → shared latent bank → readout.
            # First block carries the full receptive field; later blocks are
            # smaller and refine the shared representation.
            layers = [CausalPad(pad_left, pad_right),
                      nn.Conv1d(n_features, hidden, kernel_size=N_LAGS, bias=False),
                      nn.BatchNorm1d(hidden),
                      nn.GELU()]
            for _ in range(n_blocks - 1):
                # 'same'-length refinement blocks (small odd kernel, symmetric pad)
                layers += [nn.Conv1d(hidden, hidden, kernel_size=5, padding=2,
                                     bias=False),
                           nn.BatchNorm1d(hidden),
                           nn.GELU()]
            # 1x1 readout: every EEG channel reads from the shared latent bank.
            layers += [nn.Conv1d(hidden, n_channels, kernel_size=1, bias=True)]
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f'Unknown MODEL_VARIANT: {variant}')

    def forward(self, x):
        # x: (1, n_features, T)  →  (1, n_channels, T)
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════════
# Training / evaluation — leave-one-trial-out, matching the ridge protocol
# ════════════════════════════════════════════════════════════════════════════════

def _to_tensor(arr_2d):
    """(T, C) numpy → (1, C, T) float32 tensor on DEVICE."""
    return torch.from_numpy(
        np.ascontiguousarray(arr_2d.T[None].astype(np.float32))).to(DEVICE)


def _pearsonr_channels(pred, target):
    """Mean Pearson r across EEG channels. pred/target: (1, n_ch, T) torch tensors.

    Uses numpy/scipy for numerical stability; the detach+cpu conversion is
    cheap at eval time and avoids reimplementing Pearson in torch.
    """
    p = pred.detach().cpu().numpy()[0].T    # (T, n_ch)
    t = target.detach().cpu().numpy()[0].T
    rs = [pearsonr(p[:, c], t[:, c])[0] for c in range(p.shape[1])]
    return float(np.nanmean(rs))


def train_one_fold(X_tr, Y_tr, n_features, n_channels):
    """Train on a list of training trials; return the fitted model and epoch histories.

    X_tr, Y_tr : lists of (T_i, n_features) and (T_i, n_channels) numpy arrays,
    each already z-scored on its own trial (matching ridge's per-trial zscore).
    Held-out trial is NOT passed in — selection of epochs uses a small inner
    split of the training trials so nothing about the test trial leaks.

    MSE loss drives gradient updates; validation Pearson r drives early stopping
    and best-model checkpointing (higher r = better).

    Returns: model, train_mse_history, val_mse_history, val_r_history
    """
    model = StimToEEG(n_features, n_channels, MODEL_VARIANT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # Inner validation split for early stopping (last training trial held out).
    inner_val_idx = len(X_tr) - 1
    tr_idx = list(range(inner_val_idx))

    Xv = _to_tensor(X_tr[inner_val_idx])
    Yv = _to_tensor(Y_tr[inner_val_idx])

    train_mse_history, val_mse_history, val_r_history = [], [], []

    # Best-model selection by validation Pearson r (higher = better).
    best_val_r, best_state, since = -np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_mse = []
        for i in tr_idx:
            Xi, Yi = _to_tensor(X_tr[i]), _to_tensor(Y_tr[i])
            opt.zero_grad()
            loss = loss_fn(model(Xi), Yi)
            loss.backward()
            opt.step()
            epoch_train_mse.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred_v = model(Xv)
            v_mse = loss_fn(pred_v, Yv).item()
            v_r   = _pearsonr_channels(pred_v, Yv)

        train_mse_history.append(float(np.mean(epoch_train_mse)))
        val_mse_history.append(v_mse)
        val_r_history.append(v_r)

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
    """Leave-one-trial-out CV. Returns (Y_pred_concat, Y_true_concat, learning_curve_stats).

    Mirrors loocv_ridge: predict each held-out trial from a model trained on the
    other trials, then concatenate all held-out predictions for per-channel r.

    learning_curve_stats: SimpleNamespace with mean/std of training MSE, validation
    MSE, and validation Pearson r aggregated across folds (NaN-padded where folds
    stopped early).
    """
    Y_pred_all, Y_true_all = [], []
    fold_train_mse, fold_val_mse, fold_val_r = [], [], []

    for i in range(len(X_all)):
        X_tr = [X_all[j] for j in range(len(X_all)) if j != i]
        Y_tr = [Y_all[j] for j in range(len(Y_all)) if j != i]
        model, tr_mse, v_mse, v_r = train_one_fold(X_tr, Y_tr, n_features, n_channels)
        fold_train_mse.append(tr_mse)
        fold_val_mse.append(v_mse)
        fold_val_r.append(v_r)

        model.eval()
        with torch.no_grad():
            pred = model(_to_tensor(X_all[i])).cpu().numpy()[0].T   # (T, n_ch)
        Y_pred_all.append(pred)
        Y_true_all.append(Y_all[i])
        if DEBUG:
            print(f"    fold {i}: trained on {len(X_tr)} trials, "
                  f"held-out T={X_all[i].shape[0]}, "
                  f"epochs={len(tr_mse)}, best_val_r={max(v_r):.4f}")

    # Pad per-fold histories to equal length with NaN (folds may stop at different epochs).
    max_epochs = max(len(h) for h in fold_train_mse)

    def _pad_histories(histories):
        mat = np.full((len(histories), max_epochs), np.nan)
        for k, h in enumerate(histories):
            mat[k, :len(h)] = h
        return mat

    tr_mse_mat  = _pad_histories(fold_train_mse)
    val_mse_mat = _pad_histories(fold_val_mse)
    val_r_mat   = _pad_histories(fold_val_r)

    learning_curve_stats = SimpleNamespace(
        mean_train_mse = np.nanmean(tr_mse_mat,  axis=0),
        std_train_mse  = np.nanstd( tr_mse_mat,  axis=0),
        mean_val_mse   = np.nanmean(val_mse_mat, axis=0),
        std_val_mse    = np.nanstd( val_mse_mat, axis=0),
        mean_val_r     = np.nanmean(val_r_mat,   axis=0),
        std_val_r      = np.nanstd( val_r_mat,   axis=0),
        n_epochs       = max_epochs,
    )

    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all), learning_curve_stats


def make_trf_result(Y_true, Y_pred, sensor_dim):
    n_ch   = Y_true.shape[1]
    r_vals = np.array([pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(n_ch)])
    return SimpleNamespace(r=eelbrain.NDVar(r_vals, dims=(sensor_dim,), name='r'))


def plot_learning_curves(lc_stats, subject, condition, variant, save_dir):
    """Save a two-panel learning-curve figure for one subject/condition/variant.

    Panel 1: mean training MSE and validation MSE vs epoch (± std shading across folds).
    Panel 2: mean validation Pearson r vs epoch (± std shading across folds).

    Highlights overfitting, unstable optimization, early-stopping behaviour, and
    fold-to-fold variance.
    """
    epochs = np.arange(1, lc_stats.n_epochs + 1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"Learning curves — {subject} | {condition} | {variant}",
        fontsize=11, fontweight='bold')

    # ── Panel 1: MSE (optimization diagnostic) ──
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

    # ── Panel 2: Validation Pearson r (model-selection criterion) ──
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
    fname = save_dir / f"{subject}_{condition}_{variant}_learning_curves.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if DEBUG:
        print(f"  Saved learning curves → {fname}")


# ════════════════════════════════════════════════════════════════════════════════
# Stimulus / IDyOM loading  (verbatim from TRF_ridge_3.py)
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

stim_pitch_surprisal_ndvars = {}
stim_pitch_entropy_ndvars   = {}
stim_onset_surprisal_ndvars = {}
stim_onset_entropy_ndvars   = {}


# ════════════════════════════════════════════════════════════════════════════════
# Main loop over subjects
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
        # Per-trial z-score, matching the ridge pipeline exactly.
        X_all = [np.column_stack([zscore(t[k]) for k in feature_keys]) for t in trials]
        Y_all = [zscore(t['eeg']) for t in trials]
        n_features = len(feature_keys)

        print(f"\n  {SUBJECT} | {condition} | variant={MODEL_VARIANT} "
              f"| features={n_features} channels={n_channels}")

        Y_pred, Y_true, lc_stats = loocv_conv(X_all, Y_all, n_features, n_channels)
        trf_conv = make_trf_result(Y_true, Y_pred, sensor_dim)

        suffix  = 'acoustic_data' if condition == 'acoustic' else 'acoustic_and_surprisal_data'
        eelbrain.save.pickle(
            {'trf_cv': trf_conv, 'Y_pred': Y_pred, 'Y_true': Y_true},
            constants.SAVE_DIR / f'{SUBJECT}_{feature_keys}_conv_{MODEL_VARIANT}_{suffix}.pkl')

        plot_learning_curves(lc_stats, SUBJECT, condition, MODEL_VARIANT, constants.SAVE_DIR)

        print(f"  {SUBJECT} | {condition}: conv ({MODEL_VARIANT}) "
              f"mean r = {trf_conv.r.mean():.4f}")