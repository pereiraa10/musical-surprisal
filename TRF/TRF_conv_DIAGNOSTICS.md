# TRF_conv_1 — Diagnostics: why does the `linear` conv beat ridge 3×?

**Status: OPEN.** Resolve before trusting any `nonlinear` result.

## The observation

For Sub2, leave-one-trial-out mean Pearson r:

| condition               | ridge  | conv `linear` |
|-------------------------|-------:|--------------:|
| acoustic                | 0.0236 |        0.0759 |
| acoustic_and_surprisal  | 0.0327 |        0.0771 |

## Why this is suspicious (not a win)

The closed-form ridge solution **is** the optimal linear least-squares fit for its
chosen alpha. A linear `Conv1d` trained by SGD on the same objective, same data, same
split cannot beat it ~3× on held-out data unless they are **not** solving the same
problem on the same split. So a 3× jump points to a leak or a mismatch, not a better
linear model. Treat the high number as a bug signal.

---

## Hypotheses, ranked

### H1 — Lag/padding alignment mismatch (was ranked most likely; PARTIALLY CLEARED — see D1 update below)
The conv padding may not place the kernel over the same lag window the ridge Toeplitz
matrix uses, so the conv predicts EEG sample `t` from stimulus samples ridge would
assign to a *different* output time. Because response latency in this dataset is real
and sharp (that's the entire point of `TRF_offset_diagnostic.py`), even a
half-window misalignment can change r a lot.

Exact conventions to reconcile:
- **Ridge** (`build_lag_matrix`, TRF_ridge_3.py L34–51): `n_lags=46`, `lag_min=-6`,
  `lag_max=39`. Pads `lag_max=39` zeros *before* and `|lag_min|=6` zeros *after*,
  builds sliding windows, then **reverses column order** so col 0 → lag_min,
  col 45 → lag_max. Row `t` of the design matrix is `x[t-lag]` for lag in
  −6…+39 (positive lag = stimulus in the past → causal response).
- **Conv** (`StimToEEG`, TRF_conv_1.py): `CausalPad(left=LAG_MAX=39, right=6)`,
  kernel length 46, `output_len==input_len`. PyTorch conv is cross-correlation
  (no kernel flip), so the mapping from kernel tap index → stimulus lag must be
  checked against ridge's reversed-column convention. **A flip/offset here is the
  prime suspect.**

→ **Diagnostic D1 settles this definitively** (see below).

### H2 — Regularization is not matched (LIKELY contributor, possibly "real")
Ridge picks alpha from `logspace(1,7,25)` via LOOCV — possibly a large alpha that
heavily shrinks the fit, giving conservative (low) held-out r. The conv uses
`weight_decay=1e-3` + early stopping, which may be far weaker effective
regularization. Weaker reg usually *hurts* held-out r — but if ridge is *over*-shrunk
(picked too large an alpha), the conv legitimately fits more signal. Either way the
two aren't a fair comparison until matched.

→ Print the alpha ridge actually selected for Sub2 (it already prints
`selected alpha = ...`). If it's near the top of the range (1e6–1e7), ridge is likely
over-regularized and *that* explains part of the gap.

### H3 — Early-stopping inner split leaks (POSSIBLE, lower probability)
`train_one_fold` holds out the **last training trial** as inner validation for epoch
selection. This does not directly expose the test trial, so it shouldn't triple r on
its own — but combined with tiny data it could mildly inflate. Worth neutralising
during diagnosis by training to a fixed epoch budget.

### H4 — Per-output bias term (MINOR)
Conv `linear` has a per-channel bias; ridge has no intercept (z-scored data).
Negligible effect on r, but remove it during the equivalence test to be clean.

### H5 — Metric/z-score mismatch (UNLIKELY — checked)
Both use per-trial z-scoring and Pearson r on concatenated held-out predictions via
the same `make_trf_result` logic. Believed identical; ruled out unless D1/D2 surprise us.

---

## Diagnostic plan (do in order)

> **UPDATE (already run once on synthetic data):** `diagnostic_lag_alignment.py`
> was executed in the single-feature / single-channel / near-noise-free case.
> Ridge and conv recovered the **same** kernel peak (156.2 ms vs an injected
> 150 ms — one sample of grid quantization, identical for both). So **H1 did NOT
> reproduce in the simple case**, which lowers its probability and raises H2's.
> Caveat: the real pipeline has 6 features × 64 channels × real noise, where an
> alignment issue could still appear — so re-run D1 in a multi-feature setting
> before fully clearing H1. But current best guess is now **H2 (regularization
> mismatch) is the leading explanation.**

### D1 — Kernel-equivalence test on synthetic data  ← START HERE
Goal: prove (or disprove) that the `linear` conv and the ridge lag matrix recover the
**same kernel** from the **same** input with a known impulse-response latency.

Scaffold provided: `diagnostic_lag_alignment.py`. It:
1. Makes a synthetic 1-D stimulus and convolves it with a known TRF kernel (a bump at
   a chosen latency) to make a synthetic "EEG" channel.
2. Fits the ridge closed-form solution via `build_lag_matrix` (imported from
   TRF_ridge_3) and reads back the recovered kernel.
3. Fits the `linear` `Conv1d` (no bias, no weight decay, no early stop, to convergence)
   and reads back its kernel.
4. Aligns both to the lag axis and reports whether their peaks land at the same lag.

**Pass criterion:** both recover a peak at the injected latency and the kernels match
(up to numerical tolerance). **If the conv peak is shifted or flipped vs ridge → H1
confirmed**, and the fix is to correct `CausalPad`/kernel orientation in
`TRF_conv_1.py` so col/tap→lag conventions agree, then re-run the equivalence test
until they match before touching real data.

### D2 — Null / shuffle test on real data
Train the `linear` conv on Sub2 but break the stimulus↔EEG correspondence on the
held-out trial (circularly shift EEG by a large offset, or permute which stimulus
pairs with which held-out EEG). Recompute r.
- Ridge under the same shuffle should collapse to ~0.
- If the conv stays well above 0 → leak/autocorrelation artifact (revisit H3).

### D3 — Match regularization, retest
Disable conv early stopping (fixed epoch budget) and sweep `weight_decay`. Separately,
log ridge's selected alpha. Compare conv-vs-ridge r when both are at comparable
effective regularization. This tells us how much of any *remaining* gap (after D1) is
the over-/under-regularization of H2 rather than a bug.

### Exit condition
The `linear` conv reproduces ridge's per-channel r within numerical tolerance (or a
small, *explained* margin from H2). Only then proceed to validate `separable`, then
interpret `nonlinear`.

---

## Notes for whoever picks this up
- The conv script could not be run end-to-end in the drafting sandbox (no EEG `.mat`,
  no eelbrain/midi_func). It was syntax-checked and smoke-tested on synthetic tensors
  (shapes, float32-safety for MPS, forward/backward for all three variants). The real
  pipeline code is copied verbatim from TRF_ridge_3.py.
- `TRF_offset_diagnostic.py` is the reference for "latency is real here," which is the
  physical reason H1 would matter so much.
- Keep all changes behind the `MODEL_VARIANT` switch; don't entangle the variants.
