# CNN TRF Model Context

Comprehensive guide for agents working on `TRF_conv_1.py` and `TRF_conv_2_windowed.py`
---

## What these scripts do

They model the mapping **musical stimulus features → continuous EEG** using a 1-D temporal ConvNet (CNN-TRF).  The scientific question is whether IDyOM *predictive-coding* features (pitch/onset surprisal and entropy) explain EEG variance beyond low-level acoustic features (envelope, onsets).

The CNN-TRF is a **drop-in replacement** for the ridge TRF in `TRF_ridge_3.py`.  All upstream preprocessing is reused verbatim; only the feature→EEG model differs.  This isolation makes held-out Pearson r directly comparable across:

| Script | Optimiser | Training unit | Default variant |
|--------|-----------|---------------|------------------|
| `TRF_ridge_3.py` | Closed-form ridge | Full dataset | — |
| `TRF_conv_1.py` | Adam (full trial) | 1 trial / step | `nonlinear` |
| `TRF_conv_2_windowed.py` | Adam (mini-batch) | 64 windows / step | `nonlinear` |

## Dataset

- **Subjects**: 20 (Sub1–Sub20).  Musicians: Sub11–Sub20.
- **EEG**: 64 channels, resampled to **64 Hz**.
- **Trials per subject**: ~10 trials, variable length (~60 s each).
- **Two model conditions** run per subject:
  - `acoustic`: features `['envelope', 'onsets']`
  - `acoustic_and_surprisal`: the above plus `['pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy']`

---

## Shared constants (both scripts)

```
SFREQ    = 64 Hz
TMIN     = -0.1 s   → LAG_MIN = -6 taps
TMAX     =  0.6 s   → LAG_MAX = 39 taps
N_LAGS   = 46 taps  (receptive field)
IC_CLIP  = 15.0     (IDyOM surprisal clip)
```

Per-trial z-scoring is applied to **both** X (features) and Y (EEG) before any model sees the data.

---

## Model architecture — `StimToEEG`

Three variants selectable via `MODEL_VARIANT`:

### `linear`
```
CausalPad(LAG_MAX, |LAG_MIN|)
Conv1d(n_features, n_channels, kernel_size=N_LAGS)
```
Mathematically equivalent to the ridge Toeplitz lag matrix.  Use to sanity-check
that conv ≈ ridge before trusting nonlinear variants.

### `separable`
```
CausalPad → Conv1d(n_features, HIDDEN, N_LAGS) → Conv1d(HIDDEN, n_channels, 1)
```
Shared temporal bank + per-channel 1×1 readout.  Still linear, but low-rank factorisation.

### `nonlinear`
```
CausalPad → Conv1d(n_features, HIDDEN, N_LAGS) → Norm → GELU
         → [Conv1d(HIDDEN, HIDDEN, 5, pad=2) → Norm → GELU] × (N_BLOCKS−1)
         → Conv1d(HIDDEN, n_channels, 1)
```

**Norm layer differs between scripts:**
- `TRF_conv_1.py`: `BatchNorm1d(HIDDEN)`
- `TRF_conv_2_windowed.py`: `GroupNorm(num_groups=4, num_channels=HIDDEN)`

GroupNorm was chosen for TRF_conv_2 because BatchNorm statistics are unstable for autocorrelated EEG sequences and when batch size = 1 (full-trial eval).

**Default**: `HIDDEN=32`, `N_BLOCKS=2`.

### Padding design — `CausalPad`

The receptive field is aligned to the TRF lag window by asymmetric padding:
- `pad_left  = LAG_MAX = 39` (past context)
- `pad_right = max(0, -LAG_MIN) = 6` (future context — pre-stimulus)

This ensures output length == input length for any input length, including windows.

---

## Evaluation protocol (identical across all scripts)

**Leave-one-trial-out cross-validation (LOOCV)**

```
For fold i:
    test  = trial i   (never seen during training or epoch selection)
    inner-val = last of remaining trials
    train = all other remaining trials
```

Metric: **Pearson r**, computed on the **concatenated held-out predictions** per channel, then averaged across 64 channels.

Early stopping: validation Pearson r on the **full inner-val trial** (not on windowed excerpts).  Best-r checkpoint is restored at end of training.

---

## Key difference: TRF_conv_1 vs TRF_conv_2

### `TRF_conv_1.py` — full-trial training
Each epoch iterates over all training trials, running one full forward/backward pass per trial.  Effective batch size = 1 trial (~3840 samples).

### `TRF_conv_2_windowed.py` — windowed mini-batch training

```
WINDOW_SEC     = 5.0  →  WINDOW_SAMPLES = 320
HOP_SEC        = 0.1  →  HOP_SAMPLES    = 6
BATCH_SIZE     = 64
```


Each trial is sliced into overlapping windows using a sliding window with stride
`HOP_SAMPLES`. Windows from all training trials are pooled into a `WindowDataset` and
fed to a `DataLoader(shuffle=True, drop_last=False)`. A typical epoch sees on the
order of 10k windows (depends on trial length and `WINDOW_SEC`), providing much
better gradient variance for nonlinear models than the 1-trial-per-step scheme in
`TRF_conv_1.py`.

**Critical invariant**: no window crosses a trial boundary, so the held-out test trial never contributes any window to training.  A runtime assertion in `loocv_conv` enforces this.

---

## Training hyperparameters

```
EPOCHS               = 200
LR                   = 1e-3
WEIGHT_DECAY         = 1e-3
EARLY_STOP_PATIENCE  = 25 epochs
```

Optimizer: Adam.  Loss: MSELoss.

---

## Outputs

### Per subject/condition pickle (keys unchanged between scripts)
```python
{
    'trf_cv': SimpleNamespace(r=eelbrain.NDVar),  # Pearson r per channel
    'Y_pred': np.ndarray,   # (T_total, 64), concatenated held-out predictions
    'Y_true': np.ndarray,   # (T_total, 64), concatenated held-out ground truth
}
```

File naming:
- TRF_conv_1: `{subject}_{features}_conv_{variant}_{suffix}.pkl`
- TRF_conv_2: `{subject}_{features}_conv2_windowed_{variant}_{suffix}.pkl`

### Figures (saved to `constants.SAVE_DIR`)
- `*_learning_curves.png` — MSE and validation Pearson r vs epoch (mean ± std across folds)
- `*_alignment_ch{ch}.png` — predicted vs actual EEG for `CHANNEL_IDX` (TRF_conv_2 only)

---

## Active diagnostic concern

`TRF_conv_1.py` with `linear` variant produces r ~3× higher than ridge for Sub2 (0.0759 vs 0.0236 for acoustic).  A linear model cannot legitimately beat the closed-form ridge optimum on held-out data — this is almost certainly a **regularisation mismatch** or **protocol bug**, not a genuine result.  See `TRF_conv_DIAGNOSTICS.md` for the full diagnostic plan.  **Do not interpret nonlinear results until the linear rung matches ridge.**

Status update: `diagnostic_lag_alignment.py` (D1) has been run once on synthetic
data and the conv/ridge kernels agreed, which *partially clears* the lag-alignment
hypothesis (H1) and makes **regularisation mismatch (H2) the leading suspect**.
D2 (shuffle test) and D3
(regularisation sweep) have not been confirmed as run. See `EXPERIMENT_LOG.md` for
the full run registry and open-question list.

**New concern (as of this update):** the windowed `nonlinear` model
(`TRF_conv_2_windowed.py`) is being observed to mostly predict ~0 (i.e. collapse
toward predicting the per-trial mean) rather than tracking the held-out EEG signal —
consistent with the optimiser settling into a low-variance local minimum rather than
learning real structure. This is a separate, newer issue from the linear-inflation
concern above and is the current focus of investigation.

Two small diagnostic scripts exist for this specific concern (see the file map
below): `TRF_conv_overfit_check.py` rules in/out an optimization-capacity bug
(can the architecture drive loss below the predict-zero baseline at all, with
regularization and early stopping disabled), and `TRF_conv_mini_windowtest.py`
sweeps window length x overlap on a ~100-window subset of one subject to
shortlist a windowing scheme before committing to a full LOOCV cohort run.

---

## How to edit this model

### Change the receptive field
Modify `TMIN`/`TMAX` at the top.  `N_LAGS`, `LAG_MIN`, `LAG_MAX`, `pad_left`, `pad_right` all recompute automatically.

### Add a new variant
Add an `elif variant == 'myvariant':` block in `StimToEEG.__init__`.  The `forward` method is unchanged.

### Adjust windowing (TRF_conv_2 only)
Change `WINDOW_SEC` / `HOP_SEC` / `BATCH_SIZE`.  `WINDOW_SAMPLES` and `HOP_SAMPLES` recompute from them.  Smaller hop → more windows → slower epochs.

### Change the normalization
In `StimToEEG`, replace `nn.GroupNorm(...)` with any norm that accepts `(N, C, L)` tensors.  Note: `BatchNorm1d` will behave differently at batch=1 (full-trial eval) vs large batches (mini-batch training).

### Switch to a different early-stopping metric
In `train_one_fold`, change the `if v_r > best_val_r` comparison to use a different quantity from the eval block.

### Run on a subset of subjects
Edit `constants.SUBJECTS` (it's a list at the top of `constants.py`).

---

## File map (TRF directory)

| File | Role |
|------|------|
| `TRF_ridge_3.py` | Closed-form ridge TRF baseline.  Ground truth for comparison. |
| `TRF_conv_1.py` | CNN-TRF, full-trial training, BatchNorm. |
| `TRF_conv_2_windowed.py` | CNN-TRF, windowed mini-batch training, GroupNorm, default `nonlinear`. |
| `TRF_conv_mini_windowtest.py` | Mini-test harness for the "mostly predicts ~0" concern. Single subject, ~100-window subset, sweeps `WINDOW_SECS_TO_TEST x OVERLAP_SECS_TO_TEST` (edit constants and re-run), reports `mse_ratio` (final train MSE / predict-zero baseline MSE) per config. Not LOOCV — a fast go/no-go signal before a full cohort run. |
| `TRF_conv_overfit_check.py` | Optimization-capacity sanity check, run alongside/before the windowing sweep. Same architecture, ~100 windows, `weight_decay=0`, no early stopping — tests whether the model CAN drive loss below the predict-zero baseline at all, to rule out an optimization/architecture bug as the cause of the collapse (as opposed to genuinely weak signal). |
| `eeg_functions.py` | EEG loading and preprocessing. |
| `midi_func.py` | Places IDyOM per-note values onto the 64 Hz time grid. |
| `constants.py` | Paths, subject list, frequency band edges, SAVE_DIR. |
| `TRF_conv_DIAGNOSTICS.md` | Active diagnostic plan for the linear-conv vs ridge mismatch. |
| `EXPERIMENT_LOG.md` | Run registry (which script/variant/subjects produced each pickle subfolder) and open-question list. |
| `CLAUDE.md` | High-level TRF project context for AI agents. |
