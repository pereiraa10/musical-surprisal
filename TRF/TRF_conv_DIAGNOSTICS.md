# TRF_conv_1 — Diagnostics: why does the `linear` conv beat ridge 3×?

**Status: OPEN.** Resolve before trusting any `nonlinear` result.

**Update (2026-07-01):** `diagnostic_d2_shuffle.py` has been run on Sub2 for both
conditions. Headline finding: the cross-trial pairing shuffle does **not**
collapse (r stays within ~1-6% of the normal held-out r), which is the
"[!!] SUSPICIOUS — distribution leak" signature the script itself flags. The
circular-shift null **does** collapse correctly. See "D2 results" below —
this is now the most important open thread in this file, and it also surfaced
a discrepancy in the ridge baseline itself that needs resolving first.

**Update (2026-07-02) — likely root cause found, NOT YET CONFIRMED against real
data:** A single-subject trial-holdout mini-test (`TRF_conv_mini_windowtest_trialholdout.py`,
unrelated script, separate from this investigation) produced held-out Pearson
r of 0.5-0.6 on Sub2 — an order of magnitude higher than anything else in this
project (ridge ~0.02-0.03; conv's "inflated" numbers topped out ~0.08). That
is not plausible as genuine improvement; it pointed at a new, more severe leak.
Root cause hypothesis: `song_id = int(stimulus_id % 10) or 10` is computed
identically in every script in this codebase, implying only **10 unique songs**
— and `diagnostic_d2_shuffle.py`'s own console output already showed **30
trials** for Sub2, i.e. likely 3 repetitions per song. If songs repeat across
trials, **any evaluation protocol that splits by raw trial index (including
the production LOOCV in `TRF_ridge_3.py`, `TRF_conv_1.py`, and
`TRF_conv_2_windowed.py`, not just the mini-test scripts) may be leaving
repeated presentations of the same song in the training set when testing on a
held-out trial of that same song.** This would not make training and test
literally identical (different listening instance, some independent neural
noise), but repeated-stimulus EEG responses are known to be partially
reproducible, so this could inflate r for every model in this project to some
degree — potentially including ridge's own "ground truth" numbers, and
possibly explaining why D2's cross-trial shuffle above didn't collapse: if the
shuffle's target trial `j` happens to replay the same song as `i` (very likely
with only 10 songs and ~3 reps each), the "shuffled" pairing isn't actually an
independent null — both trials share real, common stimulus-locked structure.

**This has NOT been confirmed against real Sub2 data yet.** `check_trial_song_repeats.py`
(new, standalone, no model/torch needed) prints the song_id per trial for a
subject and directly checks for train/held-out song overlap under both an
index-based and a song-based split — run it before treating this as settled.
If confirmed, this becomes a **project-wide** methodology question, not just a
conv-vs-ridge one: does the current LOOCV protocol split by trial or by song?
If by trial, every ridge/conv r reported anywhere in this project (including
the "ground truth" ridge numbers used as the comparison baseline throughout
this document) may need re-examination with a song-level held-out split.

## The observation

For Sub2, leave-one-trial-out mean Pearson r (as originally documented; see
the ridge-value discrepancy flagged in "D2 results" below before treating
these as settled):

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

### H2 — Regularization is not matched (DOWNGRADED — see D2 results)
Ridge picks alpha from `logspace(1,7,25)` via LOOCV — possibly a large alpha that
heavily shrinks the fit, giving conservative (low) held-out r. The conv uses
`weight_decay=1e-3` + early stopping, which may be far weaker effective
regularization. Weaker reg usually *hurts* held-out r — but if ridge is *over*-shrunk
(picked too large an alpha), the conv legitimately fits more signal. Either way the
two aren't a fair comparison until matched.

→ Print the alpha ridge actually selected for Sub2 (it already prints
`selected alpha = ...`). If it's near the top of the range (1e6–1e7), ridge is likely
over-regularized and *that* explains part of the gap.

**D2 update (2026-07-01):** `diagnostic_d2_shuffle.py` reports ridge selected
`alpha=1.00e+01` for both conditions on Sub2 — the *minimum* of the tested
`logspace(1,7,25)` grid, i.e. ridge wants as little shrinkage as the grid allows,
not more. The script's own printed verdict: "Ridge alpha=1e+01 is moderate —
reg mismatch alone is unlikely to explain 3× gap." This weakens H2 as a
sufficient explanation on its own — regularization mismatch may still be a
minor contributor, but it is very unlikely to be the primary cause of the
inflation. **H3 (leak) is now the better-supported explanation** — see below.

### H3 — Early-stopping inner split leaks / distribution leak (ELEVATED — partially confirmed by D2)
`train_one_fold` holds out the **last training trial** as inner validation for epoch
selection. This does not directly expose the test trial, so it shouldn't triple r on
its own — but combined with tiny data it could mildly inflate. Worth neutralising
during diagnosis by training to a fixed epoch budget.

**D2 update (2026-07-01):** `diagnostic_d2_shuffle.py`'s cross-trial pairing
shuffle test (D2b) — where each held-out trial's predictions are scored against
a *different* trial's real EEG instead of its own — does **not** collapse to
~0 for either condition:

| condition               | conv normal r | conv xshuffle r | collapse? |
|--------------------------|-------------:|-----------------:|:---------:|
| acoustic                 |       0.0759 |            0.0757 | **NO** |
| acoustic_and_surprisal   |       0.0771 |            0.0726 | **NO** |

xshuffle r sits within ~1-6% of the normal r in both conditions — essentially
unchanged. That means a large share of the measured r reflects something the
model predicts that correlates with *almost any* real held-out EEG trial, not
specifically the trial it was actually generated from. This is the "[!!]
SUSPICIOUS — distribution leak" signature the diagnostic script itself flags.

The circular-shift null (D2a — held-out EEG shifted by T/2 before scoring)
**does** collapse correctly in both conditions (acoustic: r=-0.0032;
acoustic_and_surprisal: r=-0.0011), which rules out the crudest failure mode
(the model isn't simply ignoring time alignment or predicting a temporally
meaningless constant). So the leak is specifically exposed by breaking the
*correct stimulus-EEG trial pairing* while preserving each signal's own
temporal structure — consistent with the model latching onto some structure
shared across trials (e.g. a common spectral/autocorrelation shape in this
subject's EEG, such as 1/f drift or a shared rhythm/artifact) rather than a
genuinely stimulus-locked response.

**Important open follow-up, called out directly in the script's own output:**
run the equivalent cross-trial shuffle on **ridge**, not just conv. If ridge's
r *also* fails to collapse under cross-trial shuffling, the leak is a property
of this dataset's cross-trial autocorrelation (relevant to every LOOCV
encoding model run on it, ridge included) rather than something specific to
the conv architecture — which would be a much bigger deal than a conv-specific
bug. This has not yet been tested.

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

### D2 — Null / shuffle test on real data  ✅ COMPLETED (2026-07-01)
Train the `linear` conv on Sub2 but break the stimulus↔EEG correspondence on the
held-out trial (circularly shift EEG by a large offset, or permute which stimulus
pairs with which held-out EEG). Recompute r.
- Ridge under the same shuffle should collapse to ~0.
- If the conv stays well above 0 → leak/autocorrelation artifact (revisit H3).

**Result:** Run via `diagnostic_d2_shuffle.py` on Sub2, both conditions.

```
acoustic:                ridge r=0.0794 (alpha=1e+01)   conv normal r=0.0759 (0.96x ridge)
                         conv shift-null r=-0.0032 (OK, collapsed)
                         conv xshuffle r=0.0757 (SUSPICIOUS, did not collapse)

acoustic_and_surprisal:  ridge r=0.0821 (alpha=1e+01)   conv normal r=0.0771 (0.94x ridge)
                         conv shift-null r=-0.0011 (OK, collapsed)
                         conv xshuffle r=0.0726 (SUSPICIOUS, did not collapse)
```

Verdict from the script: H1 (lag alignment) remains cleared, H2 (regularization)
downgraded (alpha is at the low end of the grid, not over-shrunk), and the
cross-trial shuffle result flags a genuine leak/autocorrelation concern — see
the H3 update above for full interpretation.

**Also surfaced, not yet resolved:** this script's own ridge computation
(r=0.0794 / 0.0821, using the identical `build_lag_matrix` + LOOCV-alpha-select
protocol as `TRF_ridge_3.py`) is dramatically higher than the ridge r
originally documented at the top of this file for the same subject/conditions
(0.0236 / 0.0327) — high enough that the conv/ridge ratio collapses from the
reported "3×" down to ~0.94-0.96× (conv roughly at parity with, or slightly
below, ridge) once *this* ridge number is used as the comparison point. Also
notable: `diagnostic_d2_shuffle.py` loads **30 trials** for Sub2, not the ~10
trials/subject documented in `CLAUDE.md`. Before drawing further conclusions
from the ratio itself, reconcile:
1. Why does this script's ridge r differ so much from the canonical
   `TRF_ridge_3.py` baseline — stale/outdated baseline number, a preprocessing
   change since it was recorded, or a genuine difference in trial segmentation
   (30 vs ~10 trials)?
2. Run the cross-trial-shuffle null on ridge itself (see H3 update) to check
   whether the leak is conv-specific or a property of the dataset shared by
   any LOOCV encoding model run on it.

### D3 — Match regularization, retest
Disable conv early stopping (fixed epoch budget) and sweep `weight_decay`. Separately,
log ridge's selected alpha. Compare conv-vs-ridge r when both are at comparable
effective regularization. This tells us how much of any *remaining* gap (after D1) is
the over-/under-regularization of H2 rather than a bug.

### Exit condition
The `linear` conv reproduces ridge's per-channel r within numerical tolerance (or a
small, *explained* margin from H2). Only then proceed to validate `separable`, then
interpret `nonlinear`.

**Added prerequisites (2026-07-01), before this exit condition can be evaluated
in good faith:**
1. Reconcile the ridge r discrepancy surfaced by D2 (0.0794/0.0821 from
   `diagnostic_d2_shuffle.py` vs 0.0236/0.0327 documented above from
   `TRF_ridge_3.py`) — until it's clear which number is the correct baseline
   (and why they differ), "conv reproduces ridge" isn't a well-defined target.
2. Run the cross-trial-shuffle null on ridge itself. If ridge also fails to
   collapse, the leak is a dataset property (affects the whole LOOCV protocol,
   not just conv) and the exit condition above needs to be rethought — matching
   a ridge r that is itself partly leak-inflated would not actually validate
   the conv model.

---

## Notes for whoever picks this up
- The conv script could not be run end-to-end in the drafting sandbox (no EEG `.mat`,
  no eelbrain/midi_func). It was syntax-checked and smoke-tested on synthetic tensors
  (shapes, float32-safety for MPS, forward/backward for all three variants). The real
  pipeline code is copied verbatim from TRF_ridge_3.py.
- `TRF_offset_diagnostic.py` is the reference for "latency is real here," which is the
  physical reason H1 would matter so much.
- Keep all changes behind the `MODEL_VARIANT` switch; don't entangle the variants.
