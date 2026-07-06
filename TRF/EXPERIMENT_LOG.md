# TRF Experiment Log

> Last updated: 2026-07-01
> Project: EEG encoding of musical surprisal (Liberi/diliBach dataset, 19 subjects: Sub2–Sub20)
> Key metric: Pearson r (mean over 64 channels, leave-one-trial-out CV)
> SFREQ = 64 Hz | Lag window: −100 to 600 ms (46 taps) | Conditions: `acoustic`, `acoustic_and_surprisal`

---

## Script Registry

| Script | Purpose | Model type | Output files | Status |
|--------|---------|------------|--------------|--------|
| `TRF_ridge_3.py` | **Baseline.** Closed-form ridge TRF with LOOCV alpha selection over `logspace(1,7,25)`. Cross-checked against MNE `ReceptiveField` and sklearn. Ground-truth comparison for all conv variants. | Ridge (closed-form) | `Sub*_[features]_mne_ridge_<condition>_data.pkl`, `Sub*_[features]_sklearn_ridge_<condition>_data.pkl` | ✅ Active baseline |
| `TRF_conv_1.py` | Conv TRF trained by SGD. Full-trial batches. Three variants via `MODEL_VARIANT` switch. Used to test whether deep learning can beat ridge. | Conv (linear / separable / nonlinear) | `Sub*_[features]_conv_<variant>_<condition>_data.pkl`, `Sub*_<condition>_<variant>_learning_curves.png` | ⚠️ Active — linear variant has unexplained ~3× r inflation vs ridge |
| `TRF_conv_2_windowed.py` | Windowed mini-batch training variant of conv_1. Fixes gradient variance issues with full-trial batches. Uses GroupNorm instead of BatchNorm. Early-stopping on full-trial Pearson r. Default: nonlinear. | Conv windowed (nonlinear) | `Sub*_[features]_conv2_windowed_<variant>_<condition>_data.pkl`, `*_windowed_learning_curves.png`, `*_windowed_alignment_ch0.png` | ✅ Active |
| `TRF_conv_2_windowed_linear.py` | Same as `TRF_conv_2_windowed.py` but `MODEL_VARIANT = 'linear'`. Used to check whether windowed training resolves the linear conv inflation issue. | Conv windowed (linear) | Same as above with `linear` variant tag | ✅ Active |
| `TRF_conv_mini_windowtest.py` | Small-N (~100 window) harness for the windowed-nonlinear "mostly predicts ~0" concern. Sweeps window length x overlap on a single subject, no LOOCV, reports `mse_ratio` vs the predict-zero baseline per config. | Conv (nonlinear, mini-test) | `mini_windowtest_summary_<subject>_<condition>.csv`, `mini_windowtest_curves_<subject>_<condition>.png` | ✅ Active — see Diagnostic Scripts below |
| `TRF_conv_mini_windowtest_traincurves.py` | Fork of `TRF_conv_mini_windowtest.py` adding train-vs-val divergence tracking per config: best_val_epoch/mse, val_uptick_from_best, final_gap/gap_ratio, and a heuristic STABLE/OVERFITTING/UNDERFIT-OR-FLAT verdict (computed from a smoothed val curve to reduce noise from the small ~20-window val split). | Conv (nonlinear, mini-test) | `mini_windowtest_traincurves_summary_<subject>_<condition>.csv`, `mini_windowtest_traincurves_<subject>_<condition>.png` (train+val overlay), `mini_windowtest_gap_<subject>_<condition>.png` (val−train trajectory) | ✅ Active |
| `TRF_conv_mini_windowtest_trialholdout.py` | Fork of `_traincurves.py` fixing a leakage confound: validation there was a random slice of the same shuffled window pool used for training, so high-overlap configs had near-duplicate train/val windows (up to ~86% content overlap at hop=0.7s/window=5s), making them look artificially better as overlap increased. This version holds out whole trials (never windowed for training) and evaluates via genuine whole-trial inference: `heldout_r` (real Pearson r), `pred_std_ratio` (flags "still predicting ~the mean"). Overlap is now `OVERLAP_FRAC_TO_TEST` (fraction of window_sec) so it's comparable across window sizes. **Fixed 2026-07-02:** holdout is now by SONG identity (`HELD_OUT_N_SONGS`), not raw trial index — see the song-repetition open question below; the index-based version produced implausible heldout_r ~0.5-0.6. | Conv (nonlinear, mini-test) | `mini_windowtest_trialholdout_summary_<subject>_<condition>.csv`, `trialholdout_alignment_<subject>_<condition>_w<window>_of<overlap>.png` (per-config), `trialholdout_alignment_comparison_<subject>_<condition>.png` (all configs vs actual EEG, overlaid) | ✅ Active — suggested interim step before full LOOCV |
| `check_trial_song_repeats.py` | Standalone, no-model diagnostic: prints song_id (from `stimulus_id % 10 or 10`) per trial for a subject, and checks whether an index-based vs song-based holdout would leak repeated songs between train and held-out sets. | Diagnostic | Printed only | 🆕 Created 2026-07-02, not yet run against real data |
| `TRF_conv_overfit_check.py` | Optimization-capacity sanity check for the same "predicts ~0" concern: ~100 windows, `weight_decay=0`, no early stopping, full-batch — tests whether the architecture CAN drive loss below the predict-zero baseline at all. | Conv (nonlinear, mini-test) | `overfit_check_<subject>_<condition>_<variant>.png` | ✅ Active — see Diagnostic Scripts below |
| `TRF_boosting.py` | Boosting approach to the TRF problem. | Boosting | `Sub*_[features]_boosting_<condition>_data.pkl` | 🔬 Experimental |
| `TRF_offset_diagnostic.py` | Stimulus-offset sweep (0–350 ms) finding the latency maximising envelope↔EEG Pearson r per subject/channel. Confirms that response latency is real. Independent of the conv work. | Diagnostic | `TRF_offset_diagnostic_output/offset_correlations_per_subject_channel.csv`, `offset_correlations_avg_over_channels.csv`, `offset_correlations_avg_over_subjects.csv`, `offset_sweep_bar_chart.png` | ✅ Complete (one-shot) |
| `TRF_ridge.py` | Early ridge implementation. | Ridge | `*_all_data.pkl` | 🗄️ Superseded by TRF_ridge_3 |
| `TRF_ridge_2.py` | Second ridge iteration. | Ridge | — | 🗄️ Superseded by TRF_ridge_3 |
| `TRF_sklearn.py` | sklearn-based TRF. | Ridge (sklearn) | — | 🗄️ Superseded |
| `TRF_mne.py` | MNE-based TRF. | MNE ReceptiveField | — | 🗄️ Superseded |
| `TRF_pickle_A.py` / `TRF_pickle_AM.py` / `TRF_pickle_A_and_AM.py` | Early pickle generation scripts. | — | — | 🗄️ Superseded |

---

## Diagnostic Scripts

| Script | What it tests | Hypothesis from DIAGNOSTICS.md | Output | Last result |
|--------|--------------|-------------------------------|--------|-------------|
| `diagnostic_lag_alignment.py` | D1: do conv and ridge recover the same TRF kernel from synthetic data? (1 feature, 1 channel) | H1: lag misalignment | Printed only | ✅ Passed — conv and ridge agree on simple case |
| `diagnostic_d1_multifeature.py` | D1 extended + D3: 6 features × multiple channels; regularisation sweep (alpha vs conv LOOCV r) | H1 (extended) + H2: regularisation mismatch | Printed only | Status unknown — check if this was run |
| `diagnostic_d2_shuffle.py` | D2: shuffle test on real Sub2 — does r collapse when held-out pairing is broken? Also reports LOOCV-selected ridge alpha. | H3: data leak | Printed only | ✅ **Run 2026-07-01.** Circular-shift null collapses correctly (r≈0) in both conditions, but the cross-trial pairing shuffle does **NOT** collapse (acoustic: normal r=0.0759 vs xshuffle r=0.0757; surprisal: 0.0771 vs 0.0726) — flagged by the script itself as a distribution/autocorrelation leak. Ridge alpha selected = 1e+01 (low end of grid) in both conditions, weakening H2. Also surfaced: this script's own ridge r (0.0794/0.0821) is much higher than the canonical `TRF_ridge_3.py` baseline documented below (0.0236/0.0327) — unreconciled; script also reports 30 trials for Sub2, vs the ~10/subject documented in `CLAUDE.md`. Full detail in `TRF_conv_DIAGNOSTICS.md`. |
| `TRF_conv_overfit_check.py` | Optimization-capacity check: can StimToEEG (nonlinear) drive train MSE below the predict-zero baseline on a small fixed subset with regularization disabled? | Rules in/out an optimization/architecture bug as the cause of the windowed-nonlinear "mostly predicts ~0" behavior | Printed + `overfit_check_*.png` | ✅ **Run 2026-07-01** (Sub2, acoustic_and_surprisal). N_WINDOWS_SUBSET=50/LR=5e-3: best ratio 0.369. N=100/LR=5e-3: 0.502. N=200/LR=5e-3: 0.554. N=200/LR=1e-2: 0.545. All comfortably below the 0.95 "stuck" threshold — **no evidence of an optimization/architecture bug.** Ratio worsening with N is the expected fixed-capacity-vs-more-data scaling (total time samples = N x window_samples grows faster than the ~16k-parameter model's fitting capacity), not a red flag. Note: this script does not test generalization (no held-out split by design) — see `TRF_conv_mini_windowtest.py` for that. |

---

## Run Log

| # | Pickle subfolder | Date | Script | Variant / settings | Subjects | Conditions | Extra outputs | Notes |
|---|-----------------|------|--------|--------------------|----------|------------|---------------|-------|
| 1 | `misc/` | pre-2026 | Early scripts | — | Sub10 only | acoustic, surprisal+entropy (old names) | TRF pkl files, individual feature pkls | Exploratory. Old feature naming: `surprisal`, `entropy` (not `pitch_surprisal` etc.) |
| 2 | `encoding_decoding/` | pre-2026 | Early scripts | — | Sub10–Sub19 | acoustic, surprisal+entropy | — | Old feature naming. Also contains decoding data. |
| 3 | `encoding/` | pre-2026 | TRF_ridge.py (or early) | — | Sub10–Sub20 | acoustic, acoustic_and_surprisal | — | First full-cohort run with current feature names. No model tag in filenames. |
| 4 | `encoding_22_03/` | ~2022-03 | Early script | — | Sub10–Sub20 | acoustic, acoustic_and_surprisal, acoustic_music | — | Has `_acoustic_music_` condition (3 conditions). Likely pre-IDyOM feature finalisation. |
| 5 | `zorka_encoding_2026-03-25/` | 2026-03-25 | Unknown early script | — | Sub10–Sub20 | acoustic only | — | Acoustic-only run. Named "zorka" — possibly collaboration or specific code version. |
| 6 | `encoding_2026-04-20/` | 2026-04-20 | TRF_ridge_2.py or early ridge_3 | — | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | — | No model tag in filenames (`_acoustic_data.pkl`). Pre-dates mne/sklearn labelling. |
| 7 | `encoding_2026-04-22/` | 2026-04-22 | TRF_ridge_2.py or early ridge_3 | — | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | — | Same format as 04-20. Likely a re-run with minor changes. |
| 8 | `encoding_2026-04-23/` | 2026-04-23 | **TRF_ridge_3.py** | MNE + sklearn ridge | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | — | First run with `_mne_ridge_` / `_sklearn_ridge_` in filenames → confirms TRF_ridge_3. |
| 9 | `encoding_2026-05-01/` | 2026-05-01 | **TRF_ridge_3.py** | MNE + sklearn ridge | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | — | Full cohort ridge run. |
| 10 | `encoding_2026-05-15/` | 2026-05-15 | **TRF_ridge_3.py** | MNE + sklearn ridge | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | — | Full cohort ridge run. Likely after a code change — check vs 05-01. |
| 11 | `encoding_2026-05-22/` | 2026-05-22 | **TRF_conv_1.py** | `nonlinear` | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | — | First full-cohort nonlinear conv run. No learning curve plots saved. |
| 12 | `encoding_2026-05-28/` | 2026-05-28 | **TRF_conv_1.py** | `nonlinear` | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | `Sub*_{condition}_nonlinear_learning_curves.png` (per subject) | Full cohort + learning curves. Re-run of #11 with plot saving added. |
| 13 | `encoding_2026-05-29/` | 2026-05-29 | **TRF_ridge_3.py** | MNE + sklearn ridge | Sub1–Sub20 (20) | acoustic, acoustic_and_surprisal | — | Ridge re-run, possibly after preprocessing or code fix. Use this as the most recent ridge baseline. |
| 14 | `encoding_2026-06-05/` | 2026-06-05 | **TRF_conv_1.py** | `nonlinear` | Sub2 only (1) | acoustic, acoustic_and_surprisal | `Sub2_{condition}_nonlinear_learning_curves.png` | Single-subject debug run. Sub2 is the diagnostic subject used in CLAUDE.md comparisons. |
| 15 | `encoding_2026-06-11/` | 2026-06-11 | **TRF_conv_2_windowed_linear.py** | `linear` (windowed) | Sub1–Sub16 (16) | acoustic, acoustic_and_surprisal | `Sub*_{condition}_linear_windowed_learning_curves.png`, `Sub*_{condition}_linear_windowed_alignment_ch0.png` | First windowed-linear run. Tests whether windowed training + proper early-stopping resolves the 3× r inflation seen with conv_1 linear. Sub17–Sub20 not yet included. |
| 16 | `encoding_2026-06-12/` | 2026-06-12 | **TRF_conv_2_windowed.py** | `nonlinear` (windowed) | Sub1 only (1) | acoustic | `Sub1_acoustic_nonlinear_windowed_learning_curves.png`, `Sub1_acoustic_nonlinear_windowed_alignment_ch0.png` | Partial run (today). First windowed nonlinear run. |

---

## Open Questions / Active Blockers

- [ ] **⚠️ Linear conv r inflation — re-scoped by D2 results (2026-07-01)** — Originally: `TRF_conv_1.py` linear variant gives r ~3× higher than ridge on Sub2 (acoustic: 0.0759 vs 0.0236; surprisal: 0.0771 vs 0.0327). `diagnostic_d2_shuffle.py` now shows its own internal ridge computation gives r=0.0794/0.0821 for the same subject/conditions — much closer to conv's numbers (ratio ~0.94-0.96×, not 3×). H2 (regularisation mismatch) is downgraded (ridge selected alpha=1e+01, the low end of the grid, not an over-shrunk fit). **New leading concern:** the cross-trial pairing shuffle test does NOT collapse (see below), suggesting a leak/autocorrelation artifact. Full detail in `TRF_conv_DIAGNOSTICS.md`.
- [ ] **🆕 Ridge r discrepancy (2026-07-01)** — `diagnostic_d2_shuffle.py`'s internal ridge r for Sub2 (0.0794 acoustic / 0.0821 surprisal, using the same `build_lag_matrix`/LOOCV-alpha-select code path as `TRF_ridge_3.py`) is much higher than the canonical baseline documented in this file and in `CLAUDE.md` (0.0236 / 0.0327). Also, the diagnostic script loads **30 trials** for Sub2 vs the ~10/subject documented elsewhere. Needs reconciling: stale baseline number, a preprocessing change since that baseline was recorded, or a genuine difference in trial segmentation between scripts. Until resolved, don't treat the "3×" framing as settled in either direction.
- [ ] **🆕🆕 Possible song-repetition leak — project-wide, unconfirmed (2026-07-02)** — `TRF_conv_mini_windowtest_trialholdout.py` returned held-out r=0.5-0.6 (vs ~0.02-0.08 everywhere else in this project), which is implausible as genuine improvement. Likely cause: `song_id = stimulus_id % 10 (or 10)` implies only 10 unique songs; Sub2 has 30 trials (per D2's console output above) → ~3 repetitions/song. The mini-test's original trial-index-based holdout very likely held out repeats of songs already in training — fixed there by holding out whole songs instead. **Open question, not yet confirmed against real data:** does this same issue affect the production LOOCV in `TRF_ridge_3.py` / `TRF_conv_1.py` / `TRF_conv_2_windowed.py`, which all split by trial index? If so, ridge's own headline r values (the "ground truth" baseline used throughout `TRF_conv_DIAGNOSTICS.md`) may also be affected. Run `check_trial_song_repeats.py` (new, standalone, no model needed) to confirm song repetition and index/song split overlap before drawing conclusions. This may also explain the D2 cross-trial-shuffle non-collapse below (a "shuffled" pairing could land on another repetition of the same song).
- [ ] **🆕 D2 cross-trial-shuffle leak (2026-07-01)** — `diagnostic_d2_shuffle.py` on Sub2: circular-shift null collapses correctly (r≈0, both conditions) but cross-trial pairing shuffle does NOT (acoustic 0.0759→0.0757, surprisal 0.0771→0.0726 — essentially unchanged). This is the script's own "[!!] SUSPICIOUS — distribution leak" flag. Next step (called out in the script's own output, not yet done): run the same cross-trial shuffle on ridge to check whether the leak is conv-specific or a property of this dataset's cross-trial autocorrelation that would affect ridge too.
- [ ] **🆕 Overfit-capacity check — resolved, no bug (2026-07-01)** — `TRF_conv_overfit_check.py` run on Sub2 (acoustic_and_surprisal) across N_WINDOWS_SUBSET=50/100/200 and LR=5e-3/1e-2. All runs drove train MSE well below the predict-zero baseline (best ratio 0.37-0.55), far from the 0.95 "stuck" threshold — no evidence of an optimization/architecture bug behind the windowed-nonlinear "mostly predicts ~0" behavior. Ratio worsening with N reflects expected capacity-vs-data-volume scaling for this small shared-weight conv net, not a red flag. This script deliberately doesn't test generalization (no held-out split) — next step is the windowing sweep in `TRF_conv_mini_windowtest.py`.
- [ ] **Diagnostic D3** — `diagnostic_d1_multifeature.py` (D1 extended + D3 regularisation sweep) status still unconfirmed.
- [ ] **Windowed linear parity** — Does run #15 (`encoding_2026-06-11`) show linear windowed conv matching ridge? Check results.
- [ ] **Complete windowed nonlinear run** — Run #16 is partial (Sub1 only). Needs full cohort.
- [ ] **Sub1 appears in runs** — `constants.SUBJECTS` in CLAUDE.md excludes Sub1, but Sub1 appears in several pickle subfolders. Investigate.

---

## Key Results Summary

| Run # | Condition | Model | Variant | Subjects | Mean r (over channels) | Notes |
|-------|-----------|-------|---------|----------|----------------------|-------|
| #9/#13 (ridge) | acoustic | TRF_ridge_3 | MNE ridge | Sub2 | 0.0236 | From CLAUDE.md |
| #9/#13 (ridge) | acoustic_and_surprisal | TRF_ridge_3 | MNE ridge | Sub2 | 0.0327 | From CLAUDE.md |
| #11/#12 (conv_1) | acoustic | TRF_conv_1 | linear | Sub2 | 0.0759 ⚠️ | Inflated vs the 0.0236 baseline above — see open question |
| #11/#12 (conv_1) | acoustic_and_surprisal | TRF_conv_1 | linear | Sub2 | 0.0771 ⚠️ | Inflated vs the 0.0327 baseline above — see open question |
| — | acoustic | TRF_conv_1 | nonlinear | full cohort | TBD | Results not yet extracted |
| — | acoustic_and_surprisal | TRF_conv_1 | nonlinear | full cohort | TBD | Results not yet extracted |
| D2 (2026-07-01) | acoustic | diagnostic_d2_shuffle.py (internal ridge) | ridge, alpha=1e+01 | Sub2 | 0.0794 | **Discrepant** with the 0.0236 baseline above — unreconciled, see open questions |
| D2 (2026-07-01) | acoustic | diagnostic_d2_shuffle.py | conv linear (normal) | Sub2 | 0.0759 | Now only 0.96× this run's own ridge number (not 3×) |
| D2 (2026-07-01) | acoustic | diagnostic_d2_shuffle.py | conv linear (xshuffle null) | Sub2 | 0.0757 | Did NOT collapse — leak signature |
| D2 (2026-07-01) | acoustic_and_surprisal | diagnostic_d2_shuffle.py (internal ridge) | ridge, alpha=1e+01 | Sub2 | 0.0821 | **Discrepant** with the 0.0327 baseline above — unreconciled |
| D2 (2026-07-01) | acoustic_and_surprisal | diagnostic_d2_shuffle.py | conv linear (normal) | Sub2 | 0.0771 | Now only 0.94× this run's own ridge number |
| D2 (2026-07-01) | acoustic_and_surprisal | diagnostic_d2_shuffle.py | conv linear (xshuffle null) | Sub2 | 0.0726 | Did NOT collapse — leak signature |
