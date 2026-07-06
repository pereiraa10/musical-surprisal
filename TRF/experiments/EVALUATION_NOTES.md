# Evaluation approaches across the 4 experiment scripts

`dataset.py` centralizes data loading/preprocessing only. Model fitting,
cross-validation, alpha selection, and lag-matrix construction are
deliberately left in each script, because they differ enough between model
families that a shared abstraction would be premature. This doc is the
knowledge dump for *if/when* we want to build a shared eval harness later —
what's genuinely common, what's genuinely different, and one real
methodological gap worth knowing about.

## What's common to all 4 (already handled by `dataset.py`)

- Per-trial z-scoring of every feature and the EEG.
- Trial-level structure: a list of trials, each independently z-scored.
- The lag window convention: `TMIN=-0.1`, `TMAX=0.6` s, `SFREQ=64` Hz (46 taps,
  lags -6..+39).

## `TRF_sklearn.py` — explicit Toeplitz ridge

- Builds the lag matrix itself (`build_lag_matrix` / `build_design_matrix`):
  vectorized sliding-window view over each z-scored feature, concatenated
  across features into one `Phi` per trial.
- **Alpha selection**: trial-based LOOCV directly on the Toeplitz `Phi`
  matrices, using an XTX/XTY rank-1 update trick so each fold is
  `O(T_i * p^2)` instead of re-stacking n-1 trials from scratch. Picks the
  alpha maximizing mean held-out Pearson r across channels and folds.
- **Held-out evaluation**: same rank-1-update LOOCV, at the selected alpha.
  `Y_pred`/`Y_true` are the concatenation of every trial's held-out fold.
- Produces genuine `(n_channels, n_lags*n_features)` ridge coefficients per
  fold — reshape to `(n_channels, n_lags, n_features)` for a TRF-kernel plot.

## `TRF_mne.py` — MNE `ReceptiveField`

- Does **not** build its own lag matrix; passes raw (non-lagged) per-trial
  feature arrays to `mne.decoding.ReceptiveField`, which constructs its lag
  matrix internally.
- **Alpha selection**: trial-based LOOCV, but re-fits a fresh
  `ReceptiveField(estimator=Ridge(alpha=...))` per fold per alpha candidate —
  much more expensive than sklearn's rank-1-update approach, since MNE has no
  equivalent shortcut exposed.
- Important: **alpha values selected here are not comparable to
  `TRF_sklearn.py`'s**, because the two implementations construct/scale their
  lag matrices differently. This is intentional (see the docstring at the top
  of both original scripts) — don't "fix" this by sharing one alpha search.
- `ReceptiveField.coef_` can be reshaped to the same
  `(n_channels, n_lags, n_features)` TRF-kernel shape as sklearn's.

## `TRF_pickle_A_and_AM.py` — eelbrain `boosting`

- Uses `eelbrain.boosting(..., partitions=10, test=True)`, which has **its
  own internal train/validate/test partitioning** — not the same mechanism as
  the manual trial-based LOOCV the other three scripts implement by hand.
  `test=True` does hold out data the model never trained on, but the split
  granularity and selection procedure are boosting's own algorithm, not an
  externally-controlled "leave exactly one trial out, cycle through all
  trials" loop. **This is a genuine methodological difference worth knowing
  before comparing boosting's r directly against the other three** — it is
  *not* simply "boosting's version of LOOCV."
- Historically used `TMIN=-0.05, TMAX=0.550` (different from the rest of the
  ladder's `-0.1/0.6`) and a different envelope source (on-the-fly
  `eelbrain.load.wav(...).envelope()` at 100 Hz, vs. `dataStim.mat`'s
  precomputed envelope at 64 Hz used everywhere else). The rewritten
  `experiments/TRF_pickle_A_and_AM.py` now goes through `dataset.py`, so it
  uses the same `TMIN/TMAX/SFREQ` and envelope source as the other three —
  results from the new script are **not numerically comparable** to pickles
  produced by the old top-level `TRF_pickle_A_and_AM.py`.
- `trf.h[i]` from the boosting result is already a proper per-predictor TRF
  NDVar (time x sensor) — richer than the ridge scripts' raw coefficient
  arrays, but not directly reducible to the same
  `(n_channels, n_lags, n_features)` ndarray without picking a fixed
  predictor ordering.

## `TRF_conv_2_windowed.py` — SGD-trained 1-D conv

- No explicit lag matrix; the `CausalPad` + `Conv1d(kernel_size=N_LAGS)`
  gives every output sample the same `[LAG_MIN, LAG_MAX]` receptive field the
  ridge Toeplitz matrix provides.
- **Training**: mini-batch SGD over overlapping windows sliced from training
  trials (no window crosses a trial boundary). Early stopping / checkpoint
  selection uses Pearson r computed on the *full* held-out validation trial
  (not window-level MSE, which need not be monotone with full-trial r).
- **Outer LOOCV**: identical structure to the ridge scripts (leave one trial
  out, cycle through all trials), but the "training" step inside each fold is
  a full from-scratch SGD run with its own inner train/val split (last
  training trial held out for early stopping) — much more expensive per fold
  than the ridge closed-form solve.
- For `MODEL_VARIANT='linear'` and `'separable'`, the first `Conv1d` layer's
  weight tensor *is* a TRF kernel in the same sense as ridge's coefficients
  (reshape to `(n_channels, n_lags, n_features)` — mind the flipped
  correlation-vs-convolution kernel orientation before comparing to ridge).
  For `'nonlinear'`, there's no single linear kernel to extract.

## If we build a shared eval module later

The one piece that's *actually* shareable without fighting the differences
above is the outer LOOCV loop shape (`for i in range(n_trials): train on
the rest, evaluate on i, concatenate`) plus `per_trial_r` / pickle-saving
(already factored into `results.py`). Alpha selection, lag-matrix
construction, and the inner training loop are genuinely model-specific and
should probably stay that way even in a future shared module — the shared
piece would be a thin "run this LOOCV loop and save the result" wrapper that
takes a `fit_fold(train_trials) -> predict_fn` callback, not a single
"one true trainer" function.
