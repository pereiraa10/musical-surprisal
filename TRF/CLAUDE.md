# CLAUDE.md

Context for working on the TRF (Temporal Response Function) encoding models in this
directory. Read this first, then `TRF_conv_DIAGNOSTICS.md` for the active task.

---

## What this project is

EEG encoding analysis for a music-listening experiment. We model the mapping from
musical stimulus features → continuous EEG using **Temporal Response Functions
(TRFs)**. The scientific question is whether *predictive-coding* features derived
from IDyOM (pitch/onset **surprisal** and **entropy**) explain EEG variance beyond
low-level **acoustic** features (envelope, onsets).

Dataset: 19 subjects (Sub2–Sub20; Sub1 excluded in `constants.SUBJECTS`),
~10 trials/subject, 64 EEG channels, processed at **64 Hz**. Musicians are
Sub11–Sub20, non-musicians Sub2–Sub10 (see `load_subject_raw_eeg`).

Two model conditions are run throughout:
- `acoustic` → features `['envelope', 'onsets']`
- `acoustic_and_surprisal` → the above plus
  `['pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy']`

---

## Current high-level goal

Improve on the **linear ridge TRF** baseline using deep learning, *without fooling
ourselves*. The strategy is a model ladder where each rung must be validated against
the rung below before we trust it:

1. **`linear`** conv — must reproduce the ridge solution (sanity rung). ← WE ARE HERE
2. **`separable`** conv — shared temporal bank + 1×1 spatial readout (still linear).
3. **`nonlinear`** conv — stacked Conv1d+BatchNorm+GELU → shared latent → 1×1 readout.

Longer-horizon ideas discussed but **not yet started**: learnable per-channel
latencies (would subsume `TRF_offset_diagnostic.py`), a learnable nonlinearity on the
surprisal features (a real neuroscience hypothesis test), and a cross-subject shared
backbone with per-subject adapters (most likely to actually beat per-subject ridge,
given how little data each subject has).

---

## ⚠️ ACTIVE BLOCKER — read before doing anything else

The `linear` conv variant is producing held-out Pearson **r ~3× higher than ridge**,
which is a **red flag, not a win**. A linear model trained by SGD cannot legitimately
beat the closed-form ridge optimum on held-out data by 3×; the two are supposed to be
solving the same problem. This almost certainly indicates a **leak or a
protocol/alignment mismatch**, and it must be resolved before any nonlinear result is
trustworthy.

Observed (Sub2):

| condition               | ridge (TRF_ridge_3) | conv `linear` (TRF_conv_1) |
|-------------------------|--------------------:|---------------------------:|
| acoustic                |              0.0236 |                     0.0759 |
| acoustic_and_surprisal  |              0.0327 |                     0.0771 |

Full hypotheses and the exact diagnostic plan are in **`TRF_conv_DIAGNOSTICS.md`**.
A runnable first diagnostic (`diagnostic_lag_alignment.py`) is already provided and
has been run once: in the simple synthetic case the conv and ridge recover the *same*
kernel, so the lag-alignment hypothesis is partially cleared and **regularization
mismatch is now the leading suspect**. Do not interpret the nonlinear variant's
numbers until the linear rung matches ridge.

---

## Codebase map

| File | Role |
|------|------|
| `TRF_ridge_3.py` | **Baseline.** Linear ridge TRF. Closed-form solve with XTX rank-1 LOOCV, alpha selected per condition via trial-based LOOCV over `RIDGE_ALPHAS = logspace(1,7,25)`. Also runs an MNE `ReceptiveField` cross-check. This is ground truth. |
| `TRF_conv_1.py` | **New deep model.** Re-expresses the TRF as a 1-D temporal ConvNet trained by SGD. `MODEL_VARIANT` switch (`linear`/`separable`/`nonlinear`). MPS/CUDA/CPU aware. Reuses ridge's entire preprocessing + feature pipeline verbatim; only the feature→EEG mapping differs. |
| `TRF_offset_diagnostic.py` | Stimulus-offset sweep (0–350 ms) finding the latency that maximises envelope↔EEG Pearson r per subject/channel, plus a continuous xcorr peak search. Outputs CSVs + a bar chart. Independent of the conv work but relevant: it documents that response latency is real and sharp, which is *why* lag alignment matters in the blocker above. |
| `diagnostic_lag_alignment.py` | **Diagnostic D1.** Standalone (numpy+torch, no EEG) test of whether the linear conv and ridge lag matrix recover the same kernel from synthetic data with known latency. Already run once → they agree in the simple case. See `TRF_conv_DIAGNOSTICS.md`. |
| `eeg_functions.py` | EEG loading + preprocessing. Key fns: `load_subject_raw_eeg` (no resampling, keeps padding), `preprocess_eeg_trials` (per-trial LPF→downsample→HPF→strip padding, matching MATLAB CNSP order), `create_mne_raw_from_preprocessed`, `create_eelbrain_events`. |
| `constants.py` | Paths, subject list, band edges (`LOW_FREQUENCY=1`, `HIGH_FREQUENCY=8`). `SAVE_DIR` is dated per run. |
| `midi_func.py` | `make_surprisal_timeseries` — places IDyOM per-note values onto the 64 Hz time grid via MIDI onsets. (Not shown in current context; referenced by both model scripts.) |
| `offset_correlations_per_subject_channel.csv` | Output of the offset diagnostic (1216 rows = 19 subj × 64 ch). |

---

## Conventions & gotchas (please preserve)

- **SFREQ = 64 Hz everywhere.** Must match between preprocessing and stimulus
  resampling or alignment breaks.
- **Lag window** `TMIN=-0.1`, `TMAX=0.6` s → **46 taps**, lags **−6 … +39** at 64 Hz.
- **Preprocessing order matters**: LPF (at orig fs) → downsample → HPF (at target fs)
  → remove padding *last*. Per-trial, never on the concatenated signal (avoids filter
  bleed across trials). Don't "simplify" this.
- **Per-trial z-scoring** of both features and EEG, exactly as ridge does. The conv
  script must keep this identical for the comparison to be valid.
- **Evaluation = leave-one-trial-out**, Pearson r per channel on the **concatenated
  held-out predictions**. Mean r over channels is the headline number. Any deep model
  MUST use this same protocol — no leakage of the test trial into model/epoch/alpha
  selection.
- **`IC_CLIP = 15.0`** clips IDyOM surprisal. Currently fixed; a candidate to make
  learnable later.
- **MPS (Apple Silicon)**: `TRF_conv_1.py` selects CUDA→MPS→CPU automatically. MPS
  requires **float32** (no float64 anywhere) — the `_to_tensor` helper enforces this.
  If an op errors on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1` to drop unsupported
  ops to CPU. For this tiny model, MPS may not beat CPU; correctness is identical.

## How to run

```bash
cd <repo>/musical-surprisal/TRF/    # the directory holding these scripts
python TRF_ridge_3.py               # baseline
python TRF_conv_1.py                # deep model (set MODEL_VARIANT at top)
```

For fast iteration, temporarily set `constants.SUBJECTS = ['Sub2']` and reduce
`EPOCHS`. Outputs (pickles) land in `constants.SAVE_DIR` (dated).

## Environment

- Python + numpy, scipy, scikit-learn, **eelbrain**, **mne**, **pretty_midi**, **torch**.
- The EEG `.mat` files live under `constants.DATA_ROOT` (Dryad dataset); not in git.
- Sanity checks for the conv model can run on synthetic tensors without the dataset.
