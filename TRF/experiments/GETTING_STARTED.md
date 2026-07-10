# Getting Started — TRF Experiments

`TRF/experiments/` is a pipeline to train 4 neural encoding model variants on different EEG datasets of participants listening to music. 
Datasets are saved in `/datasets/` folder and contain 3 datasets:
- [liberi_dataset](https://datadryad.org/dataset/doi:10.5061/dryad.g1jwstqmh): 20 participants listening to midi piano recordings per [Di Liberti 2020](https://elifesciences.org/articles/51784) study
- [MIIR_dataset](https://datadryad.org/dataset/doi:10.5061/dryad.g1jwstqmh): 10 participants listening to music from [Stober 2015](https://ismir2015.uma.es/articles/224_Paper.pdf) study on music imagery 
- [daly_dataset](https://github.com/OpenNeuroDatasets/ds002725/tree/master): 10 participants listening to classical music from [Daly 2019](https://www.nature.com/articles/s41598-019-45105-2)

Each dataset comes with its own config file which contains information on the parameters of each dataset. 

Surprisal data is calculated based on the stimulus provided in the dataset using the [IDyOMpy](https://github.com/GuiMarion/IDyOMpy/tree/master/idyom) library. Set-up instructions are available in the respective repo. 

## Prerequisites

- Make a conda python environment with requirements per `package-list.txt`
  ```
  conda create --name conda-env python=3.11 --file package-list.txt
  
  ```

## Running a model

To run all models on the Di Liberti dataset (default), run:

```bash
python run_all_models.py
```

To run individual models only, call their respective scripts: 

```bash
# Explicit Toeplitz ridge (own lag-matrix + own alpha search)
python TRF_sklearn.py

# MNE ReceptiveField ridge (MNE's own internal lag-matrix + own alpha search)
python TRF_mne.py

# eelbrain boosting TRF
python TRF_boosting.py

# Conv (windowed mini-batch SGD) — set MODEL_VARIANT inside the script first:
# 'linear' | 'separable' | 'nonlinear'
python TRF_conv.py
```

Each script loops over every subject in `config.subjects` and all feature_sets provided 
(default config compares 2 feature_sets: `acoustic`, `acoustic_and_surprisal`), writing one pickle per (subject, feature_set,
[variant]) to `config.paths.save_dir` (default `experiments/results/encoding_<today>/`), named
`{subject}__{model_family}[_{variant}]__{condition}.pkl`. See `results.py`'s module
docstring for the full pickle schema, and `EVALUATION_NOTES.md` for how each model's
evaluation protocol differs — read that before comparing r-values across scripts.

## Fast iteration

Inspect what config a script would actually resolve to, without running anything:

```bash
python config.py --help          # list every overridable flag
```

## Changing configuration

Three ways, in increasing order of permanence.

### 1. One-off CLI override (no file edit)

Every script honors the same flags (they all call
`config.load_config(cli_args=sys.argv[1:])`):

For example: 
```bash
python TRF_sklearn.py --sfreq 128 --tmin -0.2 --tmax 0.8
python TRF_conv.py --window-samples 256 --hop-samples 128
python TRF_mne.py --save-dir results/my_test_run
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

Use the different config files to source different datasets. 

```bash

python TRF_sklearn.py --config config_daly.yaml
```

#### Running on OpenMIIR (`config_openmiir.yaml`)

`config_openmiir.yaml` configures the OpenMIIR music-imagery dataset: `.fif` EEG
(loaded via `utils._load_eeg_from_fif`), envelope computed on demand from
`audio/full.v{1,2}/*.wav` (`stimulus_source_type: audio_files`, no precomputed
`dataStim.mat`), and only the 60 "perception condition" trials per subject, across
the 10 subjects with local `.fif` files (P01, P04, P05, P06, P07, P09, P11, P12,
P13, P14). Intended usage is the same `--config` pattern as any other dataset:

```bash

python TRF_sklearn.py --config config_openmiir.yaml
python run_all_models.py --config config_openmiir.yaml
```


### Model-specific hyperparameters (not in config.yaml)

`TRF_conv.py` has its own knobs near the top of the file — architecture/
training choices, not dataset/preprocessing config: `MODEL_VARIANT`
('linear' | 'separable' | 'nonlinear'), `HIDDEN`, `N_BLOCKS`, `EPOCHS`, `LR`,
`WEIGHT_DECAY`, `EARLY_STOP_PATIENCE`, `WINDOW_SEC`/`HOP_SEC` (seconds; converted to
samples via `config.sfreq`), `BATCH_SIZE`. Edit them directly in the script.
`TRF_boosting.py` similarly has `BASIS`/`PARTITIONS`/`ERROR` as boosting-specific
knobs near its top.

## Smoke-testing the data pipeline alone (no model fitting)

```bash
python dataset.py
```

Loads Sub2, builds a `TRFDataset` in both full-trial and fixed-window modes, and asserts
shapes/dtypes are correct. Good first check after any `config.yaml`/`utils.py` change,
before running a full model script.
