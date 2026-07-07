"""
TRF_conv.py (experiments/ version)
────────────────────────────────────────────────────────────────────────────────
Windowed mini-batch conv TRF. Same model/training code as the original
../TRF_conv_2_windowed.py — the only structural change is that data loading,
preprocessing, alignment, and z-scoring now come from dataset.PreparedSubject +
dataset.TRFDataset instead of ~150 lines duplicated from TRF_sklearn.py /
TRF_mne.py inline. PreparedSubject runs the condition-independent pipeline
once per subject; both conditions below reuse it via .to_dataset(condition, ...).

Two additions beyond the port:
  - `extract_kernel()` pulls the literal TRF kernel out of 'linear'/'separable'
    models (their conv layers, correctly re-ordered to match the ridge lag
    convention — see its docstring) so conv results can be compared against
    ridge/MNE weight plots. 'nonlinear' has no single linear kernel, so this
    is None for that variant.
  - Results now go through results.build_result()/save_result() — see
    ../experiments/results.py for the schema and ../CLAUDE.md /
    EVALUATION_NOTES.md for the eval protocol this preserves unchanged
    (trial-based LOOCV, Pearson r on concatenated held-out predictions).

Run from musical-surprisal/TRF/ or musical-surprisal/TRF/experiments/:
    python experiments/TRF_conv_2_windowed.py

NOTE: requires PyTorch (pip install torch). No GPU required; runs on CPU.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt

from config import load_config
import utils
from dataset import PreparedSubject
import results as res

# Config is loaded at import time so the model-architecture constants below
# (N_LAGS / LAG_MIN / LAG_MAX, WINDOW_SAMPLES) can derive from it, exactly as
# they derived from the module-level SFREQ/TMIN/TMAX before. CLI overrides
# (e.g. --sfreq, --tmin) are honored via sys.argv.
config = load_config(cli_args=sys.argv[1:])
SFREQ = config.sfreq
TMIN = config.tmin
TMAX = config.tmax


# ── Deep-learning knobs ──
MODEL_VARIANT = 'nonlinear'   # 'linear' | 'separable' | 'nonlinear'
HIDDEN        = 32            # width of the shared temporal feature bank
N_BLOCKS      = 2             # nonlinear conv blocks (only used by 'nonlinear')
EPOCHS        = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-3          # the SGD-era analog of ridge alpha
EARLY_STOP_PATIENCE = 25      # epochs without validation-r improvement before stopping

# ── Windowing constants ──
# Windows are extracted independently from each trial; no window crosses trial
# boundaries, so the LOOCV held-out guarantee is preserved.
WINDOW_SEC     = 7.0          # window length in seconds
HOP_SEC        = 6.0          # hop (stride) in seconds
BATCH_SIZE     = 64           # windows per gradient step

WINDOW_SAMPLES = int(WINDOW_SEC * SFREQ)
HOP_SAMPLES    = int(HOP_SEC   * SFREQ)

# Receptive field in taps, derived exactly like build_lag_matrix in the ridge code
N_LAGS  = int(round((TMAX - TMIN) * SFREQ)) + 1
LAG_MIN = int(round(TMIN * SFREQ))
LAG_MAX = LAG_MIN + N_LAGS - 1

DEBUG = True
SEED  = 0


def _select_device():
    """Pick the best available backend: CUDA -> MPS (Apple Silicon) -> CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _select_device()
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════════
#
# Tensor convention: (batch, channels=features, time). CausalPad ensures each
# output sample sees the same [LAG_MIN, LAG_MAX] context the ridge Toeplitz
# matrix provides, independent of batch size or window length.

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
        pad_left, pad_right = LAG_MAX, max(0, -LAG_MIN)

        if variant == 'linear':
            # One conv, no nonlinearity — equivalent to the TRF lag matrix.
            self.net = nn.Sequential(
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, n_channels, kernel_size=N_LAGS, bias=True),
            )

        elif variant == 'separable':
            # Shared temporal bank (linear) -> 1x1 spatial readout (linear).
            self.net = nn.Sequential(
                CausalPad(pad_left, pad_right),
                nn.Conv1d(n_features, hidden, kernel_size=N_LAGS, bias=True),
                nn.Conv1d(hidden, n_channels, kernel_size=1, bias=True),
            )

        elif variant == 'nonlinear':
            # GroupNorm (not BatchNorm1d): normalises per-instance independent
            # of batch size, stable at batch size 1 (full-trial eval) and for
            # autocorrelated EEG sequences where BatchNorm statistics vary
            # across trials.
            assert hidden % 4 == 0, \
                f"HIDDEN={hidden} must be divisible by GroupNorm num_groups=4"
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


def extract_kernel(model, variant):
    """Pull the literal TRF kernel out of a trained model, reshaped to
    (n_channels, n_lags, n_features) to match the ridge/MNE weight schema.

    PyTorch's Conv1d computes cross-correlation, not convolution, and the
    model pads its input with `LAG_MAX` zeros on the left. Given that padding,
    kernel tap k=0 lines up with lag=LAG_MAX and tap k=n_lags-1 lines up with
    lag=LAG_MIN — the reverse of the ridge Toeplitz column order (column 0 =
    LAG_MIN, see build_lag_matrix in TRF_sklearn.py). The kernel axis is
    flipped here before reshaping so the two are directly comparable.

    'linear'    : the single Conv1d weight IS the kernel.
    'separable' : composes the temporal bank and 1x1 spatial readout, both
                  linear, into one effective kernel.
    'nonlinear' : no single linear kernel exists; returns None.
    """
    if variant == 'linear':
        conv = model.net[1]
        w = conv.weight.detach().cpu().numpy()          # (n_channels, n_features, n_lags)
        w = w[:, :, ::-1]
        return np.ascontiguousarray(np.transpose(w, (0, 2, 1)))

    elif variant == 'separable':
        conv1, conv2 = model.net[1], model.net[2]
        w1 = conv1.weight.detach().cpu().numpy()         # (hidden, n_features, n_lags)
        w1 = w1[:, :, ::-1]
        w2 = conv2.weight.detach().cpu().numpy()[:, :, 0]  # (n_channels, hidden)
        eff = np.einsum('ch,hfl->cfl', w2, w1)           # (n_channels, n_features, n_lags)
        return np.ascontiguousarray(np.transpose(eff, (0, 2, 1)))

    return None


# ════════════════════════════════════════════════════════════════════════════════
# Training / evaluation — leave-one-trial-out, matching the ridge protocol
# ════════════════════════════════════════════════════════════════════════════════

def _to_tensor(arr_2d):
    """(T, C) numpy -> (1, C, T) float32 tensor on DEVICE (for full-trial inference)."""
    return torch.from_numpy(
        np.ascontiguousarray(arr_2d.T[None].astype(np.float32))).to(DEVICE)


def _pearsonr_channels(pred, target):
    """Mean Pearson r across EEG channels. pred/target: (1, n_ch, T) torch tensors."""
    p = pred.detach().cpu().numpy()[0].T
    t = target.detach().cpu().numpy()[0].T
    rs = [pearsonr(p[:, c], t[:, c])[0] for c in range(p.shape[1])]
    return float(np.nanmean(rs))


def train_one_fold(ds, train_trial_ids, n_features, n_channels):
    """Train on the windows of `train_trial_ids` using mini-batch updates.

    ds : windowed TRFDataset — the DataLoader-ready window source.
    train_trial_ids : ordered list of trial indices used for training (the
        held-out LOOCV trial already excluded). The LAST id is the inner
        validation trial (matching the old 'last training trial' split): its
        windows drive val-MSE monitoring and its FULL trial drives the
        Pearson-r early-stopping metric; the remaining ids supply the training
        windows.

    Returns: model, train_mse_history, val_mse_history, val_r_history
    """
    model = StimToEEG(n_features, n_channels, MODEL_VARIANT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # Inner validation split: last training trial held out (unchanged).
    inner_val_trial = train_trial_ids[-1]
    pure_train_trials = train_trial_ids[:-1]

    # Full inner-val trial tensors — Pearson-r early-stopping metric.
    val_trial = ds.trials[inner_val_trial]
    Xv_full = _to_tensor(np.column_stack([val_trial[k] for k in ds.feature_keys]))
    Yv_full = _to_tensor(val_trial['eeg'])

    # Training windows = every window whose source trial is a pure-train trial;
    # val windows (MSE monitoring only) = windows of the inner-val trial. 
    train_window_idx = [w for ti in pure_train_trials for w in ds.windows_for_trial(ti)]
    val_window_idx = ds.windows_for_trial(inner_val_trial)

    train_loader = DataLoader(
        Subset(ds, train_window_idx),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_window_idx),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )

    train_mse_history, val_mse_history, val_r_history = [], [], []
    best_val_r, best_state, since = -np.inf, None, 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_mse = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            opt.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)
            loss.backward()
            opt.step()
            epoch_train_mse.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_mse_batches = []
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                val_mse_batches.append(loss_fn(model(X_batch), Y_batch).item())
            v_mse = float(np.mean(val_mse_batches))

            # Pearson r on the FULL validation trial — checkpoint metric.
            pred_full = model(Xv_full)
            v_r = _pearsonr_channels(pred_full, Yv_full)

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


def loocv_conv(ds, n_features, n_channels):
    """Leave-one-trial-out CV over the windowed TRFDataset `ds`.

    Returns (Y_pred_concat, Y_true_concat, trial_boundaries, kernels, lc_stats).
    kernels is a list of per-fold extracted TRF kernels (None entries for
    'nonlinear', since it has no single linear kernel).
    """
    n_trials = ds.n_trials
    Y_pred_all, Y_true_all, trial_boundaries = [], [], []
    fold_train_mse, fold_val_mse, fold_val_r = [], [], []
    kernels = []
    offset = 0

    for i in range(n_trials):
        # Training trials in the old X_tr order: every trial except held-out i.
        train_trial_ids = [j for j in range(n_trials) if j != i]
        # Leakage guard: the held-out trial is not in the training set.
        assert i not in train_trial_ids, f"Fold {i}: held-out trial leaked into training."

        if DEBUG:
            n_pure_tr = len(train_trial_ids) - 1
            n_tr_wins = sum(len(ds.windows_for_trial(k)) for k in train_trial_ids[:-1])
            n_val_wins = len(ds.windows_for_trial(train_trial_ids[-1]))
            heldout_n = ds.trial_lengths[i]
            print(f"  fold {i}:  train_trials={n_pure_tr}  "
                  f"train_windows={n_tr_wins}  "
                  f"val_windows={n_val_wins}  "
                  f"heldout_samples={heldout_n}")

        model, tr_mse, v_mse, v_r = train_one_fold(ds, train_trial_ids, n_features, n_channels)
        fold_train_mse.append(tr_mse)
        fold_val_mse.append(v_mse)
        fold_val_r.append(v_r)
        kernels.append(extract_kernel(model, MODEL_VARIANT))

        # Held-out eval on the full trial i (same full-trial inference as before).
        held = ds.trials[i]
        X_test = np.column_stack([held[k] for k in ds.feature_keys])
        model.eval()
        with torch.no_grad():
            pred = model(_to_tensor(X_test)).cpu().numpy()[0].T   # (T, n_ch)
        Y_pred_all.append(pred)
        Y_true_all.append(held['eeg'])
        trial_boundaries.append((offset, offset + len(pred)))
        offset += len(pred)

        if DEBUG:
            print(f"    -> epochs={len(tr_mse)}, best_val_r={max(v_r):.4f}")

    max_epochs = max(len(h) for h in fold_train_mse)

    def _pad(histories):
        mat = np.full((len(histories), max_epochs), np.nan)
        for k, h in enumerate(histories):
            mat[k, :len(h)] = h
        return mat

    lc_stats = SimpleNamespace(
        mean_train_mse=np.nanmean(_pad(fold_train_mse), axis=0),
        std_train_mse=np.nanstd(_pad(fold_train_mse), axis=0),
        mean_val_mse=np.nanmean(_pad(fold_val_mse), axis=0),
        std_val_mse=np.nanstd(_pad(fold_val_mse), axis=0),
        mean_val_r=np.nanmean(_pad(fold_val_r), axis=0),
        std_val_r=np.nanstd(_pad(fold_val_r), axis=0),
        n_epochs=max_epochs,
    )
    return (np.concatenate(Y_pred_all), np.concatenate(Y_true_all),
            trial_boundaries, kernels, lc_stats)


# ════════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(lc_stats, subject, condition, variant, save_dir):
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
        print(f"  Saved learning curves -> {fname}")


def average_kernel(kernels):
    """Average per-fold kernels; None if the variant has none (or any fold failed)."""
    if any(k is None for k in kernels):
        return None
    return np.mean(kernels, axis=0)


# ════════════════════════════════════════════════════════════════════════════════
# Main loop over subjects
# ════════════════════════════════════════════════════════════════════════════════

def main():
    save_dir = config.paths.save_dir
    os.makedirs(save_dir, exist_ok=True)

    hyperparams = {
        'MODEL_VARIANT': MODEL_VARIANT, 'HIDDEN': HIDDEN, 'N_BLOCKS': N_BLOCKS,
        'EPOCHS': EPOCHS, 'LR': LR, 'WEIGHT_DECAY': WEIGHT_DECAY,
        'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE,
        'WINDOW_SEC': WINDOW_SEC, 'HOP_SEC': HOP_SEC, 'BATCH_SIZE': BATCH_SIZE,
        'N_LAGS': N_LAGS, 'LAG_MIN': LAG_MIN, 'LAG_MAX': LAG_MAX,
        'SEED': SEED, 'DEVICE': str(DEVICE),
    }

    for SUBJECT in config.subjects:
        # Load raw EEG + run the condition-independent pipeline once per
        # subject (PreparedSubject); each condition below only reruns the
        # cheap per-condition z-scoring step, not the full preprocessing.
        eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=SUBJECT)
        eeg_data = utils.load_subject_raw_eeg(eeg_path, SUBJECT)
        prepared = PreparedSubject(SUBJECT, eeg_data, config, debug=DEBUG)

        for condition, feature_keys in config.conditions.items():
            ds = prepared.to_dataset(condition, window_samples=WINDOW_SAMPLES,
                                     hop_samples=HOP_SAMPLES)
            n_features = len(feature_keys)

            print(f"\n  {SUBJECT} | {condition} | variant={MODEL_VARIANT} (windowed) "
                  f"| features={n_features} channels={ds.n_channels} "
                  f"| window={WINDOW_SAMPLES}smp hop={HOP_SAMPLES}smp batch={BATCH_SIZE}")

            Y_pred, Y_true, trial_boundaries, kernels, lc_stats = loocv_conv(
                ds, n_features, ds.n_channels)

            weights = average_kernel(kernels)
            training_history = {
                'mean_train_mse': lc_stats.mean_train_mse, 'std_train_mse': lc_stats.std_train_mse,
                'mean_val_mse': lc_stats.mean_val_mse, 'std_val_mse': lc_stats.std_val_mse,
                'mean_val_r': lc_stats.mean_val_r, 'std_val_r': lc_stats.std_val_r,
                'n_epochs': lc_stats.n_epochs,
            }

            result = res.build_result(
                subject=SUBJECT, subject_type=ds.subject_type, condition=condition,
                feature_keys=feature_keys, model_family='conv',
                model_variant=MODEL_VARIANT, channel_names=ds.channel_names,
                Y_true=Y_true, Y_pred=Y_pred, trial_boundaries=trial_boundaries,
                weights=weights, training_history=training_history,
                extra_meta={'hyperparams': hyperparams},
            )
            path = res.result_filename(
                save_dir, SUBJECT, 'conv', condition, variant=MODEL_VARIANT)
            res.save_result(path, result)

            plot_learning_curves(lc_stats, SUBJECT, condition, MODEL_VARIANT, save_dir)

            print(f"  {SUBJECT} | {condition}: conv2_windowed ({MODEL_VARIANT}) "
                  f"mean r = {result['r_per_channel'].mean():.4f}")


if __name__ == '__main__':
    main()
