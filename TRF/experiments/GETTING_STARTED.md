# Getting Started — TRF Experiments

`TRF/experiments/` is a self-contained pipeline (no dependency on the older top-level
`TRF/*.py` scripts): one config file, one data-loading class (`TRFDataset`), and 4 model
variants that all consume it identically.

## Prerequisites

- A Python environment with: `mne`, `eelbrain`, `torch`, `scipy`, `numpy`, `scikit-learn`,
  `pretty_midi`, `pyyaml` (e.g. `conda activate eelbrain-env`, or your own env — if
  `pyyaml` is missing: `pip install pyyaml`).
- The dataset in place: `TRF/liberi_dataset/...` (EEG/wav/midi) and the IDyOM surprisal
  `.mat` outputs at `TRF/../IDyOM/...` (paths come from `experiments/config.yaml`).

## Running a model

Run from either `TRF/` or `TRF/experiments/` — paths resolve from each file's own
location, not your current directory.

```bash
cd musical-surprisal/TRF

# Explicit Toeplitz ridge (own lag-matrix + own alpha search)
python experiments/TRF_sklearn.py

# MNE ReceptiveField ridge (MNE's own internal lag-matrix + own alpha search)
python experiments/TRF_mne.py

# eelbrain boosting TRF
python experiments/TRF_pickle_A_and_AM.py

# Conv (windowed mini-batch SGD) — set MODEL_VARIANT inside the script first:
# 'linear' | 'separable' | 'nonlinear'
python experiments/TRF_conv_2_windowed.py
```

Each script loops over every subject in `config.subjects` and both conditions
(`acoustic`, `acoustic_and_surprisal`), writing one pickle per (subject, condition[,
variant]) to `config.paths.save_dir` (default `TRF/pickles/encoding_<today>/`), named
`{subject}__{model_family}[_{variant}]__{condition}.pkl`. See `results.py`'s module
docstring for the full pickle schema, and `EVALUATION_NOTES.md` for how each model's
evaluation protocol differs — read that before comparing r-values across scripts.

## Fast iteration

Trim `config.yaml`'s `subjects:` list to one subject (e.g. just `Sub2`) for a quick
end-to-end smoke test of any script. For the conv script, also lower `EPOCHS` near the
top of `TRF_conv_2_windowed.py` (default 200).

Inspect what config a script would actually resolve to, without running anything:

```bash
python experiments/config.py
python experiments/config.py --help          # list every overridable flag
```

## Changing configuration

Three ways, in increasing order of permanence.

### 1. One-off CLI override (no file edit)

Every script honors the same flags (they all call
`config.load_config(cli_args=sys.argv[1:])`):

```bash
python experiments/TRF_sklearn.py --sfreq 128 --tmin -0.2 --tmax 0.8
python experiments/TRF_conv_2_windowed.py --window-samples 256 --hop-samples 128
python experiments/TRF_mne.py --save-dir pickles/my_test_run
```

Available flags: `--config`, `--eeg-filename-pattern`, `--sfreq`, `--tmin`, `--tmax`,
`--low-frequency`, `--high-frequency`, `--ic-clip`, `--save-dir`, `--window-samples`,
`--hop-samples`.

### 2. Edit `config.yaml` (persistent, affects every script)

```yaml
preprocessing:
  sfreq: 128        # was 64

subjects:
  - Sub2            # trimmed for fast local iteration
  - Sub3
```

### 3. Point at a different config file entirely

Useful for a parallel experiment without touching the default:

```bash
cp experiments/config.yaml experiments/config_lofi.yaml
# edit config_lofi.yaml...
python experiments/TRF_sklearn.py --config experiments/config_lofi.yaml
```

### 4. Programmatic override (Python shell / notebook / one-off script)

`load_config()` returns a plain `Config` dataclass — mutate it directly. How you inject
it back depends on *when* a script loads config:

`TRF_sklearn.py` / `TRF_mne.py` / `TRF_pickle_A_and_AM.py` call `load_config()` lazily,
inside `main()` — patch the module's `load_config` name before calling `main()`:

```python
from pathlib import Path
from config import load_config
import TRF_sklearn

def patched(*a, **k):
    c = load_config()
    c.subjects = ['Sub2']
    c.paths.save_dir = Path('/tmp/scratch')   # keep test output out of pickles/
    return c

TRF_sklearn.load_config = patched
TRF_sklearn.main()
```

`TRF_conv_2_windowed.py` loads config once at *import time* (a module-level `config`
object, since its architecture constants derive from it) — mutate that object directly:

```python
import TRF_conv_2_windowed as conv
conv.config.subjects = ['Sub2']
conv.config.conditions = {'acoustic': conv.config.conditions['acoustic']}
conv.EPOCHS = 2          # script-level hyperparameters override the same way
conv.main()
```

### Model-specific hyperparameters (not in config.yaml)

`TRF_conv_2_windowed.py` has its own knobs near the top of the file — architecture/
training choices, not dataset/preprocessing config: `MODEL_VARIANT`
('linear' | 'separable' | 'nonlinear'), `HIDDEN`, `N_BLOCKS`, `EPOCHS`, `LR`,
`WEIGHT_DECAY`, `EARLY_STOP_PATIENCE`, `WINDOW_SEC`/`HOP_SEC` (seconds; converted to
samples via `config.sfreq`), `BATCH_SIZE`. Edit them directly in the script.
`TRF_pickle_A_and_AM.py` similarly has `BASIS`/`PARTITIONS`/`ERROR` as boosting-specific
knobs near its top.

## Smoke-testing the data pipeline alone (no model fitting)

```bash
python experiments/dataset.py
```

Loads Sub2, builds a `TRFDataset` in both full-trial and fixed-window modes, and asserts
shapes/dtypes are correct. Good first check after any `config.yaml`/`utils.py` change,
before running a full model script.
