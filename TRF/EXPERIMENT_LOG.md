# TRF Experiment Log

> Last updated: 2026-06-12
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
| `diagnostic_d2_shuffle.py` | D2: shuffle test on real Sub2 — does r collapse when held-out pairing is broken? Also reports LOOCV-selected ridge alpha. | H3: data leak | Printed only | Status unknown — check if this was run |

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

- [ ] **⚠️ Linear conv r inflation** — `TRF_conv_1.py` linear variant gives r ~3× higher than ridge on Sub2 (acoustic: 0.0759 vs 0.0236; surprisal: 0.0771 vs 0.0327). This cannot be legitimate. Hypotheses and diagnostics tracked in `TRF_conv_DIAGNOSTICS.md`. Leading suspect: regularisation mismatch.
- [ ] **Diagnostics D2 and D3** — Need to confirm whether `diagnostic_d2_shuffle.py` and `diagnostic_d1_multifeature.py` have been run and what they found.
- [ ] **Windowed linear parity** — Does run #15 (`encoding_2026-06-11`) show linear windowed conv matching ridge? Check results.
- [ ] **Complete windowed nonlinear run** — Run #16 is partial (Sub1 only). Needs full cohort.
- [ ] **Sub1 appears in runs** — `constants.SUBJECTS` in CLAUDE.md excludes Sub1, but Sub1 appears in several pickle subfolders. Investigate.

---

## Key Results Summary

| Run # | Condition | Model | Variant | Subjects | Mean r (over channels) | Notes |
|-------|-----------|-------|---------|----------|----------------------|-------|
| #9/#13 (ridge) | acoustic | TRF_ridge_3 | MNE ridge | Sub2 | 0.0236 | From CLAUDE.md |
| #9/#13 (ridge) | acoustic_and_surprisal | TRF_ridge_3 | MNE ridge | Sub2 | 0.0327 | From CLAUDE.md |
| #11/#12 (conv_1) | acoustic | TRF_conv_1 | linear | Sub2 | 0.0759 ⚠️ | Inflated — see open question |
| #11/#12 (conv_1) | acoustic_and_surprisal | TRF_conv_1 | linear | Sub2 | 0.0771 ⚠️ | Inflated — see open question |
| — | acoustic | TRF_conv_1 | nonlinear | full cohort | TBD | Results not yet extracted |
| — | acoustic_and_surprisal | TRF_conv_1 | nonlinear | full cohort | TBD | Results not yet extracted |
