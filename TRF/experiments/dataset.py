"""
dataset.py — PreparedSubject + TRFDataset, the single pipeline entry point for
every TRF script.

Two classes, split at the one point in the pipeline where `feature_set` first
matters:

  PreparedSubject : runs everything that is feature_set-INDEPENDENT for one
      subject — per-trial LPF/downsample/HPF, MNE + eelbrain construction,
      stimulus/IDyOM alignment, per-trial numpy assembly (utils.* steps 1-6).
      Built ONCE per subject.
  TRFDataset      : a torch.utils.data.Dataset finalizing one PreparedSubject
      for one feature_set — per-trial z-scoring of that feature_set's feature
      subset (step 7, cheap) plus window/sample-level indexing for DataLoader.
      Built once per (subject, feature_set) via PreparedSubject.to_dataset(...).

Why the split: every consumer script needs BOTH feature sets ('acoustic' and
'acoustic_and_surprisal') for each subject, but feature sets differ only in
step 7 — steps 1-6 (the expensive/most of the work) are identical either way.
Before this split, TRFDataset.__init__ reran steps 1-6 from scratch for every
feature_set (2x redundant work per subject, 20x across the cohort). Now:
PreparedSubject(subject, eeg_data, config) once, then
.to_dataset('acoustic', ...) and .to_dataset('acoustic_and_surprisal', ...)
each just redo the cheap step 7 + window indexing.

The pipeline behavior is numerically identical to before this split — this is
a reorganization of when/how often each step runs, not a change to what it
computes. `zscore_trials` (step 7) builds new arrays/dicts rather than
mutating its input, so calling it twice against the same
PreparedSubject._trials_raw (once per feature_set) is safe: no aliasing
between the two feature sets' z-scored outputs.

Consumption
-----------
    from config import load_config
    import utils
    from dataset import PreparedSubject

    config   = load_config()
    eeg_data = utils.load_subject_raw_eeg(config.paths.eeg_dir / 'dataSub2.mat', 'Sub2')
    prepared = PreparedSubject('Sub2', eeg_data, config)   # steps 1-6, ONCE

    # Full-trial mode (sklearn / mne / boosting): ds.trials is the list of
    # per-trial z-scored numpy dicts the old Dataset.get_trials(feature_set) gave.
    ds = prepared.to_dataset('acoustic', window_samples=None)   # step 7 only
    ds.trials          # list[dict]: {'eeg': (T,n_ch), <feature>: (T,), ...}
    ds.events          # eelbrain Dataset, for eelbrain.boosting()

    # Windowed mode (conv): DataLoader-ready; group windows by trial for LOOCV.
    ds = prepared.to_dataset('acoustic', window_samples=448, hop_samples=384)
    len(ds)                       # number of windows
    X, Y = ds[0]                  # (n_features, win) float32, (n_ch, win) float32
    ds.windows_for_trial(3)       # window indices whose source trial is 3
    ds.window_trial_indices       # len(ds)-long array of source trial ids

    # TRFDataset can still be constructed directly (builds its own throwaway
    # PreparedSubject internally) when only one feature_set is ever needed:
    from dataset import TRFDataset
    ds = TRFDataset('Sub2', eeg_data, 'acoustic', config, window_samples=None)
"""

import sys

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

import utils


class PreparedSubject:
    """One subject's feature_set-independent pipeline state (utils.* steps 1-6),
    computed once and shared across as many per-feature_set TRFDataset views as
    needed via `to_dataset(feature_set, ...)`.

    Parameters
    ----------
    subject : str
    eeg_data : dict
        From utils.load_subject_raw_eeg(path, subject). The caller loads the
        raw EEG (once per subject) and passes it in.
    config : Config
        From config.load_config().
    stimulus_library : optional
        A utils._StimulusLibrary. If None, utils.get_stimulus_library(config)
        is used (cached, so instances across a subject loop share one library).
    debug : bool
        Verbosity for the preprocessing/alignment steps.
    preprocess_capture, align_capture : optional
        Opt-in callbacks forwarded verbatim to utils.preprocess_eeg_trials
        (capture=) and utils.align_stimulus_and_idyom (capture=) respectively.
        Unused by every existing caller (defaults to None, identical to
        before this parameter existed); intended for methods-figure
        generation (see viz_smoke_test.py) that needs intermediate pipeline
        stages without duplicating the pipeline's numeric logic.
    """

    def __init__(self, subject, eeg_data, config, stimulus_library=None, debug=False,
                 preprocess_capture=None, align_capture=None):
        self.subject = subject
        self.config = config
        self.debug = debug
        self.subject_type = utils.subject_type_for(subject, config.subject_type)

        # 1. per-trial LPF -> downsample -> HPF -> strip padding
        preprocessed_trials = utils.preprocess_eeg_trials(
            eeg_data, target_fs=config.sfreq,
            lpf_hz=config.high_frequency, hpf_hz=config.low_frequency, debug=debug,
            capture=preprocess_capture)
        # 2. concatenate into an MNE RawArray with trial-onset stim markers
        raw = utils.create_mne_raw_from_preprocessed(
            preprocessed_trials, config.sfreq, eeg_data['chanlocs'])
        self.raw = raw
        # 3. eelbrain events
        self.events = utils.create_eelbrain_events(raw)
        # 4. shared stimulus / IDyOM library (cached across subjects)
        self._lib = stimulus_library or utils.get_stimulus_library(config)
        # 6 (reordered ahead of 5, see below). Union of every configured
        # feature_set's keys, in first-seen order — the full feature list this
        # subject's trials get built with, regardless of which feature_set(s)
        # will eventually be requested via to_dataset().
        feature_names = []
        for keys in config.feature_sets.values():
            for k in keys:
                if k not in feature_names:
                    feature_names.append(k)

        # 5. resample/align stimulus + place IDyOM features (mutates self.events).
        #    Builds ALL surprisal features regardless of which feature_set will
        #    eventually be requested — still feature_set-independent, since
        #    every named feature_set draws from this same superset.
        #    'envelope'/'onsets' are computed unconditionally inside
        #    align_stimulus_and_idyom itself; everything else in the union
        #    (pitch_surprisal, pitch_entropy, ...) requires IDyOM placement.
        surprisal_keys = [k for k in feature_names if k not in ('envelope', 'onsets')]
        utils.align_stimulus_and_idyom(
            self.events, preprocessed_trials, self._lib, subject, config.sfreq,
            config.trial_to_song_id, surprisal_keys,
            stimulus_paths=eeg_data.get('stimulus_paths'), debug=debug,
            capture=align_capture)
        # 6. per-trial numpy assembly (all features, not yet z-scored)
        (self._trials_raw, self.sensor_dim,
         self.channel_names, self.n_channels) = utils.build_trials(
            self.events, feature_names, subject)

    def to_dataset(self, feature_set, window_samples=None, hop_samples=None):
        """Finalize this prepared subject for one feature_set: step 7
        (zscore_trials, cheap) + window indexing. Returns a TRFDataset.

        Safe to call more than once (e.g. once per feature_set) on the same
        PreparedSubject — zscore_trials returns new arrays/dicts rather than
        mutating self._trials_raw, so the resulting TRFDatasets' `.trials`
        are fully independent of each other.
        """
        return TRFDataset._from_prepared(self, feature_set, window_samples, hop_samples)

# could also be all of the subjects
class TRFDataset(TorchDataset):
    """One subject's preprocessed, feature-aligned, z-scored EEG + stimulus data
    for one feature_set, indexable at the window (or whole-trial) level.

    Prefer constructing this via PreparedSubject.to_dataset(feature_set, ...)
    when a subject's EEG will be used for more than one feature_set (the normal
    case) — it skips re-running steps 1-6. The constructor below remains
    available for one-off, single-feature_set use; it just builds a throwaway
    PreparedSubject internally.

    Parameters
    ----------
    subject : str
    eeg_data : dict
        From utils.load_subject_raw_eeg(path, subject).
    feature_set : str
        Key into config.feature_sets ('acoustic' | 'acoustic_and_surprisal').
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

    def __init__(self, subject, eeg_data, feature_set, config, stimulus_library=None,
                 window_samples=None, hop_samples=None, debug=False):
        prepared = PreparedSubject(subject, eeg_data, config, stimulus_library, debug)
        self._init_from_prepared(prepared, feature_set, window_samples, hop_samples)

    @classmethod
    def _from_prepared(cls, prepared, feature_set, window_samples=None, hop_samples=None):
        self = cls.__new__(cls)
        self._init_from_prepared(prepared, feature_set, window_samples, hop_samples)
        return self

    def _init_from_prepared(self, prepared, feature_set, window_samples, hop_samples):
        config = prepared.config
        self.subject = prepared.subject
        self.feature_set = feature_set  # named subset of features considered
        self.config = config
        self.feature_keys = config.feature_sets[feature_set]

        # Subject-level metadata and eelbrain events, shared read-only from
        # the PreparedSubject (not recomputed).
        self.subject_type = prepared.subject_type
        self.events = prepared.events
        self.sensor_dim = prepared.sensor_dim
        self.channel_names = prepared.channel_names
        self.n_channels = prepared.n_channels

        # 7. per-trial z-scoring of THIS feature_set's features + EEG (+ checks).
        # The only step that depends on `feature_set`; new arrays, no mutation
        # of prepared._trials_raw, so this is safe to run once per feature_set.
        self.trials = utils.zscore_trials(
            prepared._trials_raw, feature_keys=self.feature_keys,
            subject=self.subject, debug=prepared.debug)

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
            X : (n_features, window_len)  — feature order = config.feature_sets[feature_set]
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
    # Smoke test: PreparedSubject shared across both feature sets for Sub2, plus
    # full-trial mode + windowed mode, confirming no cross-feature_set aliasing.
    from config import load_config

    config = load_config()
    subject = 'Sub2'
    eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=subject)
    eeg_data = utils.load_subject_raw_eeg(eeg_path, subject)

    prepared = PreparedSubject(subject, eeg_data, config, debug=True)

    # ── both feature sets share the same PreparedSubject (steps 1-6 run once) ──
    ds_acoustic = prepared.to_dataset('acoustic', window_samples=None)
    ds_full = prepared.to_dataset('acoustic_and_surprisal', window_samples=None)
    assert ds_acoustic.events is ds_full.events is prepared.events, \
        "both feature sets should share the same PreparedSubject.events object"
    assert ds_acoustic.trials is not ds_full.trials
    # 'acoustic' only z-scores 2 features; 'acoustic_and_surprisal' z-scores 6.
    assert set(ds_acoustic.trials[0].keys()) == {'eeg', 'envelope', 'onsets'}
    assert set(ds_full.trials[0].keys()) == {
        'eeg', 'envelope', 'onsets', 'pitch_surprisal', 'pitch_entropy',
        'onset_surprisal', 'onset_entropy'}
    # calling to_dataset twice must not have mutated prepared._trials_raw in a
    # way that cross-contaminates: envelope z-scored the same way in both.
    assert np.allclose(ds_acoustic.trials[0]['envelope'], ds_full.trials[0]['envelope'])
    print("[PreparedSubject] shared events, independent per-feature_set trials: OK")

    assert len(ds_full) == len(ds_full.trial_lengths) == ds_full.n_trials
    X0, Y0 = ds_full[0]
    T0 = ds_full.trial_lengths[0]
    assert X0.shape == (len(ds_full.feature_keys), T0), (X0.shape, T0)
    assert Y0.shape == (ds_full.n_channels, T0), (Y0.shape, T0)
    assert X0.dtype == torch.float32 and Y0.dtype == torch.float32
    print(f"[full-trial] len(ds)={len(ds_full)} == n_trials, "
          f"ds[0] X{tuple(X0.shape)} Y{tuple(Y0.shape)} float32: OK")

    # ── windowed mode, derived from the SAME prepared subject ──
    ws, hs = 64, 64
    ds_win = prepared.to_dataset('acoustic_and_surprisal', window_samples=ws, hop_samples=hs)
    manual = sum(max(0, (T - ws) // hs + 1) for T in ds_win.trial_lengths)
    assert len(ds_win) == manual == len(ds_win.window_trial_indices), (len(ds_win), manual)
    Xw, Yw = ds_win[0]
    assert Xw.shape == (len(ds_win.feature_keys), ws)
    assert Yw.shape == (ds_win.n_channels, ws)
    assert Xw.dtype == torch.float32 and Yw.dtype == torch.float32
    for ti in range(ds_win.n_trials):
        idxs = ds_win.windows_for_trial(ti)
        assert all(ds_win.window_trial_indices[j] == ti for j in idxs)
    total = sum(len(ds_win.windows_for_trial(ti)) for ti in range(ds_win.n_trials))
    assert total == len(ds_win)
    print(f"[windowed]  len(ds)={len(ds_win)} == manual window count, "
          f"ds[0] X{tuple(Xw.shape)} Y{tuple(Yw.shape)} float32, "
          f"windows_for_trial consistent: OK")

    # ── TRFDataset's direct constructor still works (builds its own throwaway
    # PreparedSubject internally) ──
    ds_direct = TRFDataset(subject, eeg_data, 'acoustic', config, window_samples=None)
    assert np.allclose(ds_direct.trials[0]['eeg'], ds_acoustic.trials[0]['eeg'])
    print("[direct constructor] TRFDataset(...) still works standalone: OK")

    print("SMOKE TEST PASSED")

    # ── methodology figures (opt-in via --visualize, default on; pass
    # --no-visualize for a fast CI-style run of just the assertions above) ──
    VISUALIZE = '--no-visualize' not in sys.argv
    if VISUALIZE:
        try:
            import viz_smoke_test
        except ImportError as e:
            print(f"\n[viz] skipping figure generation: could not import a required "
                  f"dependency ({type(e).__name__}: {e}). Run with --no-visualize to "
                  "skip this step, or install the missing package(s).")
        else:
            viz_smoke_test.run()
    else:
        print("\n[viz] --no-visualize passed: skipping figure generation.")
