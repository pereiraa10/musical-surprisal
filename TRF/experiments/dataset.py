"""
dataset.py — TRFDataset, the single pipeline entry point for every TRF script.

TRFDataset is a torch.utils.data.Dataset subclass that owns the orchestration:
given raw EEG (as loaded by utils.load_subject_raw_eeg) plus a resolved config,
it runs the full pipeline — per-trial LPF/downsample/HPF/strip-padding, MNE +
eelbrain construction, stimulus/IDyOM alignment, per-trial z-scoring — by
calling utils.* functions in order, then exposes the result in a PyTorch-friendly
form (window/sample-level __getitem__ for DataLoader) while ALSO keeping the
per-trial numpy arrays (self.trials) and the eelbrain events (self.events) that
the non-PyTorch scripts need.

The pipeline behavior is numerically identical to the previous Dataset class;
this is a reorganization of where it runs (inside one torch Dataset) and how it
is consumed, not a change to the preprocessing.

Consumption
-----------
    from config import load_config
    import utils
    from dataset import TRFDataset

    config   = load_config()
    eeg_data = utils.load_subject_raw_eeg(config.paths.eeg_dir / 'dataSub2.mat', 'Sub2')

    # Full-trial mode (sklearn / mne / boosting): ds.trials is the list of
    # per-trial z-scored numpy dicts the old Dataset.get_trials(condition) gave.
    ds = TRFDataset('Sub2', eeg_data, 'acoustic', config, window_samples=None)
    ds.trials          # list[dict]: {'eeg': (T,n_ch), <feature>: (T,), ...}
    ds.events          # eelbrain Dataset, for eelbrain.boosting()

    # Windowed mode (conv): DataLoader-ready; group windows by trial for LOOCV.
    ds = TRFDataset('Sub2', eeg_data, 'acoustic', config,
                    window_samples=448, hop_samples=384)
    len(ds)                       # number of windows
    X, Y = ds[0]                  # (n_features, win) float32, (n_ch, win) float32
    ds.windows_for_trial(3)       # window indices whose source trial is 3
    ds.window_trial_indices       # len(ds)-long array of source trial ids
"""

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

import utils

class TRFDataset(TorchDataset):
    """One subject's preprocessed, feature-aligned, z-scored EEG + stimulus data
    for one condition, indexable at the window (or whole-trial) level.

    Parameters
    ----------
    subject : str
    eeg_data : dict
        From utils.load_subject_raw_eeg(path, subject). The caller loads the raw
        EEG (once per subject) and passes it in — TRFDataset reads no EEG file
        itself, so one eeg_data can be reused across conditions/windowings.
    condition : str
        Key into config.conditions ('acoustic' | 'acoustic_and_surprisal').
    config : Config
        From config.load_config().
    stimulus_library : optional
        A utils._StimulusLibrary. If None, utils.get_stimulus_library(config) is
        used (cached, so instances across a subject loop share one library).
    window_samples : int | None
        None (and no config default) -> one window per trial spanning the whole
        (variable-length) trial. A positive int -> fixed-length sliding windows
        (1 is valid = single-sample access).
    hop_samples : int | None
        Window stride; defaults to window_samples (non-overlapping).
    """

    def __init__(self, subject, eeg_data, condition, config, stimulus_library=None,
                 window_samples=None, hop_samples=None, debug=False):
        self.subject = subject
        self.condition = condition
        self.config = config
        self.debug = debug
        self.feature_keys = config.conditions[condition]

        # Subject-level metadata (independent of eeg_data, but exposed here).
        self.subject_type = utils.subject_type_for(subject, config.subject_type)

        # ── Pipeline ───────────────────────────────────────────────────────
        # 1. per-trial LPF -> downsample -> HPF -> strip padding
        preprocessed_trials = utils.preprocess_eeg_trials(
            eeg_data, target_fs=config.sfreq,
            lpf_hz=config.high_frequency, hpf_hz=config.low_frequency, debug=debug)
        # 2. concatenate into an MNE RawArray with trial-onset stim markers
        raw = utils.create_mne_raw_from_preprocessed(
            preprocessed_trials, config.sfreq, eeg_data['chanlocs'])
        # 3. eelbrain events
        self.events = utils.create_eelbrain_events(raw)
        # 4. shared stimulus / IDyOM library (cached across subjects)
        self._lib = stimulus_library or utils.get_stimulus_library(config)
        # 5. resample/align stimulus + place IDyOM features (mutates self.events)
        utils.align_stimulus_and_idyom(
            self.events, preprocessed_trials, self._lib, subject, config.sfreq,
            config.trial_to_song_id, config.feature_keys_surprisal, debug=debug)
        # 6. per-trial numpy assembly (all features)
        feature_names = config.feature_keys_acoustic + config.feature_keys_surprisal
        trials, sensor_dim, channel_names, n_channels = utils.build_trials(
            self.events, feature_names, subject)
        self.sensor_dim = sensor_dim
        self.channel_names = channel_names
        self.n_channels = n_channels
        # 7. per-trial z-scoring of this condition's features + EEG (+ checks)
        self.trials = utils.zscore_trials(
            trials, feature_keys=self.feature_keys, subject=subject, debug=debug)

        # ── Flat window index for __getitem__ / DataLoader ──
        ws = window_samples if window_samples is not None else config.window_samples
        hs = hop_samples if hop_samples is not None else config.hop_samples
        self.window_samples = ws
        self.hop_samples = hs
        self._build_window_index()

    # ── window indexing ──────────────────────────────────────────────────────

    def _build_window_index(self):
        """Build self._index (list of (trial_idx, start, end)) and the parallel
        self.window_trial_indices array.

        window_samples None -> one whole-trial entry per trial (default,
        preserving today's variable-length behavior). A positive int -> sliding
        fixed-length windows per trial, never crossing a trial boundary (same
        guarantee as TRF_conv_2_windowed's old _make_windows), hop defaulting to
        the window length. Raises if a trial is shorter than window_samples.
        """
        index = []
        trial_ids = []
        ws = self.window_samples
        for ti, t in enumerate(self.trials):
            T = t['eeg'].shape[0]
            if ws is None:
                index.append((ti, 0, T))
                trial_ids.append(ti)
            else:
                hs = self.hop_samples if self.hop_samples is not None else ws
                if T < ws:
                    raise ValueError(
                        f"{self.subject} trial {ti}: length {T} < window_samples "
                        f"{ws}; cannot extract any windows. Shorten window_samples "
                        "or check the data.")
                for s in range(0, T - ws + 1, hs):
                    index.append((ti, s, s + ws))
                    trial_ids.append(ti)
        self._index = index
        self.window_trial_indices = np.asarray(trial_ids, dtype=int)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        """Return one (X, Y) window as float32 tensors, channel-first:
            X : (n_features, window_len)  — feature order = config.conditions[cond]
            Y : (n_channels, window_len)
        In full-trial mode window_len is that trial's full length.
        """
        trial_idx, start, end = self._index[idx]
        t = self.trials[trial_idx]
        X = np.stack([t[k][start:end] for k in self.feature_keys], axis=0)  # (n_feat, win)
        Y = t['eeg'][start:end].T                                          # (n_ch, win)
        return (torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)))

    def windows_for_trial(self, trial_idx):
        """List of flat-index positions whose source trial is `trial_idx`.
        Use with torch.utils.data.Subset to build LOOCV folds."""
        return np.nonzero(self.window_trial_indices == trial_idx)[0].tolist()

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def n_trials(self):
        return len(self.trials)

    @property
    def trial_lengths(self):
        return [t['eeg'].shape[0] for t in self.trials]


if __name__ == '__main__':
    # Smoke test: full-trial mode + windowed mode for Sub2.
    from config import load_config

    config = load_config()
    subject = 'Sub2'
    eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=subject)
    eeg_data = utils.load_subject_raw_eeg(eeg_path, subject)

    # ── full-trial mode ──
    ds_full = TRFDataset(subject, eeg_data, 'acoustic_and_surprisal', config,
                         window_samples=None, debug=True)
    assert len(ds_full) == len(ds_full.trial_lengths) == ds_full.n_trials
    X0, Y0 = ds_full[0]
    T0 = ds_full.trial_lengths[0]
    assert X0.shape == (len(ds_full.feature_keys), T0), (X0.shape, T0)
    assert Y0.shape == (ds_full.n_channels, T0), (Y0.shape, T0)
    assert X0.dtype == torch.float32 and Y0.dtype == torch.float32
    print(f"[full-trial] len(ds)={len(ds_full)} == n_trials, "
          f"ds[0] X{tuple(X0.shape)} Y{tuple(Y0.shape)} float32: OK")

    # ── windowed mode ──
    ws, hs = 64, 64
    ds_win = TRFDataset(subject, eeg_data, 'acoustic_and_surprisal', config,
                        window_samples=ws, hop_samples=hs)
    # windows never cross a trial boundary; counts match a manual computation
    manual = sum(max(0, (T - ws) // hs + 1) for T in ds_win.trial_lengths)
    assert len(ds_win) == manual == len(ds_win.window_trial_indices), (len(ds_win), manual)
    Xw, Yw = ds_win[0]
    assert Xw.shape == (len(ds_win.feature_keys), ws)
    assert Yw.shape == (ds_win.n_channels, ws)
    assert Xw.dtype == torch.float32 and Yw.dtype == torch.float32
    # windows_for_trial lines up with window_trial_indices
    for ti in range(ds_win.n_trials):
        idxs = ds_win.windows_for_trial(ti)
        assert all(ds_win.window_trial_indices[j] == ti for j in idxs)
    total = sum(len(ds_win.windows_for_trial(ti)) for ti in range(ds_win.n_trials))
    assert total == len(ds_win)
    print(f"[windowed]  len(ds)={len(ds_win)} == manual window count, "
          f"ds[0] X{tuple(Xw.shape)} Y{tuple(Yw.shape)} float32, "
          f"windows_for_trial consistent: OK")
    print("SMOKE TEST PASSED")
