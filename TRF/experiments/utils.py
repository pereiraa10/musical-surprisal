"""
utils.py — reusable data-loading / preprocessing / alignment functions for the
TRF experiments.

These are plain, testable functions: they take their inputs (including config
values) as explicit arguments rather than reading module-level globals, and they
do NOT orchestrate the pipeline themselves — dataset.TRFDataset calls them in
order. The behavior is byte-for-byte the same as the old dataset.py methods this
code was extracted from (LPF -> downsample -> HPF -> strip padding, per trial;
stimulus/IDyOM alignment; per-trial z-scoring; the three boundary assertions);
only the "read globals" -> "take arguments" plumbing changed.

Sections:
  1. General-purpose utilities (zscore, align_trial) — not liberi-specific.
  2. Dataset-specific loading/preprocessing (EEG load + format dispatch,
     filtering, MNE/eelbrain construction, IDyOM placement, trial assembly,
     z-scoring) and the shared stimulus/IDyOM library.
"""

import warnings
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.io import loadmat
from scipy.signal import resample_poly, butter, sosfiltfilt

import eelbrain
import mne
from mne.channels import make_dig_montage


# ═══════════════════════════════════════════════════════════════════════════════
# 1. General-purpose utilities (not liberi-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def align_trial(eeg, stim_arrays, trial_idx, subject, max_diff=2):
    """Trim EEG and every stimulus array to the same length.

    Rounding during resampling can cause +-1 sample discrepancies; anything
    larger than max_diff samples is treated as a real alignment error. Every
    feature array is aligned independently so a mismatch in any one of them
    is caught (rather than only checking envelope vs eeg).
    """
    n_eeg = eeg.shape[0]
    n_stim = len(next(iter(stim_arrays.values())))
    diff = abs(n_eeg - n_stim)
    if diff > max_diff:
        raise ValueError(
            f"{subject} trial {trial_idx}: EEG/stimulus length mismatch "
            f"(EEG={n_eeg}, stim={n_stim}, diff={diff} samples). "
            "Check padding removal and resampling."
        )
    n = min(n_eeg, n_stim)
    return eeg[:n], {k: v[:n] for k, v in stim_arrays.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset-specific loading / preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def subject_type_for(subject, subject_type_table):
    """Musician / non-musician label for `subject`, from an explicit table.
    Raises a clear error (not a bare KeyError) for an unlisted subject."""
    try:
        return subject_type_table[subject]
    except KeyError:
        raise KeyError(
            f"Subject {subject!r} is not in the subject_type table; add it to "
            f"config.yaml (known subjects: {sorted(subject_type_table)})."
        ) from None


def song_id_for_marker(stimulus_id, trial_to_song_id_table):
    """Song id for a trial's event-marker id, from an explicit table.
    Raises a clear error (not a bare KeyError) for an unlisted marker."""
    try:
        return trial_to_song_id_table[int(stimulus_id)]
    except KeyError:
        raise KeyError(
            f"Event-marker id {int(stimulus_id)} is not in the trial_to_song_id "
            f"table; add it to config.yaml (known markers: "
            f"{sorted(trial_to_song_id_table)})."
        ) from None


# ── EEG loading (format dispatch) ──────────────────────────────────────────────

def load_subject_raw_eeg(filepath, subject):
    """Load one subject's raw EEG, dispatching on file format.

    Thin dispatcher: `.mat` files go through _load_eeg_from_mat; any other
    extension is handed to _load_eeg_from_other_format (a stub to fill in for a
    new dataset). Both return the same `eeg_data` dict shape, so everything
    downstream (preprocess_eeg_trials, create_mne_raw_from_preprocessed) is
    format-agnostic:

        {
          'trials':    list of per-trial (n_time, n_channels) arrays, in raw
                       counts/volts BEFORE any filtering (still padded),
          'fs':        int, original sampling rate in Hz (no resampling done),
          'chanlocs':  iterable of channel objects/dicts exposing a `.labels`
                       name and `.X`/`.Y`/`.Z` position attributes,
          'pad_start': int, leading-padding sample count expressed at `fs`,
        }

    subject_type is deliberately NOT part of this dict — it's subject metadata,
    not EEG-file content; TRFDataset looks it up from config.subject_type.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == '.mat':
        return _load_eeg_from_mat(filepath, subject)
    return _load_eeg_from_other_format(filepath, subject)


def _load_eeg_from_mat(filepath, subject):
    """Load raw EEG from a CNSP-style .mat file at its original sampling
    frequency — NO resampling.

    Padding is preserved here (not removed) so that the LPF and HPF can filter
    through the padding before it is discarded — exactly as MATLAB does
    (padding is removed last, after all filtering).
    """
    mat_data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg = mat_data["eeg"]
    orig_fs = int(eeg.fs)

    for i in range(len(eeg.data)):
        # Scale int32 counts to float32 (~microvolt range) to match the
        # pre-processing standard in the CNSP MATLAB pipeline.
        eeg.data[i] = 100 * eeg.data[i].astype(np.float32) / np.iinfo(np.int32).max

    eeg_data = {
        'trials': eeg.data,
        'fs': orig_fs,                              # original fs; no resampling done
        'chanlocs': eeg.chanlocs,
        'pad_start': int(eeg.paddingStartSample),    # in original-fs samples
    }

    print(f"✓ Loaded raw EEG ({subject}): {len(eeg_data['trials'])} trials, "
          f"{eeg_data['trials'][0].shape[1]} channels @ {orig_fs} Hz")

    return eeg_data


def _load_eeg_from_other_format(filepath, subject):
    """Stub for loading raw EEG from a non-.mat source (e.g. .fif, .edf, .set).

    A future implementation must return the same `eeg_data` dict shape that
    _load_eeg_from_mat produces, so it flows through preprocess_eeg_trials
    unchanged:

        {
          'trials':    list of per-trial (n_time, n_channels) numpy arrays, in
                       raw counts/volts BEFORE filtering, WITH leading padding
                       still attached (padding is stripped later, after
                       filtering, in preprocess_eeg_trials),
          'fs':        int, original sampling rate in Hz,
          'chanlocs':  iterable of per-channel objects/dicts exposing a
                       `.labels` name and `.X`/`.Y`/`.Z` head-coordinate
                       attributes (used to build the MNE montage),
          'pad_start': int, number of leading padding samples at `fs`
                       (0 if the format has no padding convention),
        }
    """
    raise NotImplementedError(
        f"EEG loading for '{filepath.suffix}' files is not implemented "
        f"(subject {subject}, path {filepath}). Only '.mat' is supported today; "
        "implement _load_eeg_from_other_format to return the documented "
        "eeg_data dict shape for this format."
    )


def preprocess_eeg_trials(eeg_data, target_fs, lpf_hz, hpf_hz, debug=False):
    """Preprocess EEG trials independently, replicating the MATLAB CNSP workflow:

      Step 1 — LPF  each trial at original fs
      Step 2 — Downsample to target_fs
      Step 3 — HPF  each trial at target_fs
      Step 4 — Remove leading padding

    Filtering happens per trial (not on the concatenated signal) to prevent
    filter transients from one trial bleeding into the next. Padding is
    removed last so the filters settle through it rather than cold-starting at
    the real signal boundary.
    """
    trials = eeg_data['trials']
    orig_fs = eeg_data['fs']
    pad_start_orig = eeg_data['pad_start']

    g = gcd(orig_fs, target_fs)
    up = target_fs // g
    down_factor = orig_fs // g

    # Replicate MATLAB cndDownsample: scale padding to the target domain.
    pad_start_target = int(round(pad_start_orig * target_fs / orig_fs))

    nyq_orig = orig_fs / 2.0
    lpf_sos = butter(4, lpf_hz / nyq_orig, btype='low', output='sos')

    nyq_tgt = target_fs / 2.0
    hpf_sos = butter(4, hpf_hz / nyq_tgt, btype='high', output='sos')

    preprocessed_trials = []
    for i, trial in enumerate(trials):
        trial_f64 = trial.astype(np.float64)   # (n_time, n_channels)

        if debug:
            print(f"  [pre] Trial {i+1}: raw = {trial_f64.shape[0]} samples "
                  f"@ {orig_fs} Hz  (pad = {pad_start_orig} samples)")

        trial_lpf = sosfiltfilt(lpf_sos, trial_f64, axis=0)
        trial_down = resample_poly(trial_lpf, up, down_factor, axis=0)
        trial_hpf = sosfiltfilt(hpf_sos, trial_down, axis=0)
        trial_clean = trial_hpf[pad_start_target:, :]

        if debug:
            n_out = trial_clean.shape[0]
            print(f"  [pre] Trial {i+1}: post-preproc = {n_out} samples "
                  f"@ {target_fs} Hz")

        preprocessed_trials.append(trial_clean)

    return preprocessed_trials


def create_mne_raw_from_preprocessed(preprocessed_trials, target_fs, chanlocs):
    """Build an MNE RawArray from already-preprocessed trials (filtered,
    downsampled, padding-removed) by concatenating them end-to-end and
    inserting trial-onset stim-channel markers."""
    ch_names = []
    positions = []
    for ch in chanlocs:
        ch_names.append(ch.labels)
        if hasattr(ch, 'X') and hasattr(ch, 'Y') and hasattr(ch, 'Z'):
            positions.append([ch.Y, ch.X, ch.Z])

    all_trials = []
    trial_lengths = []
    for trial in preprocessed_trials:
        t = trial.T   # -> (n_channels, n_time)
        all_trials.append(t)
        trial_lengths.append(t.shape[1])

    eeg_continuous = np.hstack(all_trials)   # (n_channels, total_time)
    n_channels, n_samples = eeg_continuous.shape

    stim_data = np.zeros((1, n_samples))
    current_sample = 0
    for i, length in enumerate(trial_lengths):
        # Keep first-trial marker off sample 0 (MNE treats sample-0 events as
        # ambiguous on some pipelines).
        marker_sample = max(1, current_sample)
        stim_data[0, marker_sample] = i + 1
        current_sample += length

    data_with_stim = np.vstack([eeg_continuous, stim_data])
    ch_names_full = ch_names + ['STI']
    ch_types = ['eeg'] * n_channels + ['stim']

    info = mne.create_info(ch_names=ch_names_full, sfreq=float(target_fs), ch_types=ch_types)
    raw = mne.io.RawArray(data_with_stim, info)

    montage = make_dig_montage(
        ch_pos=dict(zip(ch_names[:n_channels], positions)), coord_frame='head')
    raw.set_montage(montage)
    return raw


def create_eelbrain_events(raw):
    """Create eelbrain events with correct column structure."""
    mne_events = mne.find_events(raw, stim_channel='STI', verbose=False)
    events_data = {
        'i_start': mne_events[:, 0],
        'trigger': mne_events[:, 2],
        'event': mne_events[:, 2],
    }
    events = eelbrain.Dataset(events_data)
    events.info['raw'] = raw
    return events


def make_surprisal_timeseries(midi_path, surprisal_vec, sfreq, n_times):
    """Place per-note IDyOM surprisal/entropy values onto the target time grid
    via MIDI note onsets. Returns a (n_times,) ndarray."""
    pm = pretty_midi.PrettyMIDI(midi_path)

    onsets = sorted(
        note.start
        for instrument in pm.instruments
        for note in instrument.notes
    )
    onsets = np.array(sorted(onsets))

    n = min(len(onsets), len(surprisal_vec))
    onsets = onsets[:n]
    surprisal_vec = surprisal_vec[:n]

    surprisal_ts = np.zeros(n_times)
    for t, s in zip(onsets, surprisal_vec):
        sample = int(round(t * sfreq))
        if 0 <= sample < n_times:
            surprisal_ts[sample] += s   # impulse

    return surprisal_ts


# ── Shared stimulus / IDyOM library (same for every subject) ────────────────────

class _StimulusLibrary:
    """dataStim.mat + the IDyOM .mat files, parsed once and reused across
    subjects (they are identical for everyone). This is a plain class now — the
    old singleton machinery moved out to get_stimulus_library() (an lru_cache
    factory), so sharing is explicit and injectable.

    All the config values it needs are passed in; it reads no module globals.
    """

    def __init__(self, eeg_dir, sfreq, pitch_surprisal_file, onset_surprisal_file,
                 ic_clip, midi_dir):
        self.sfreq = sfreq
        self.midi_dir = Path(midi_dir)

        stim_mat = loadmat(Path(eeg_dir) / "dataStim.mat",
                           struct_as_record=False, squeeze_me=True)
        stim = stim_mat["stim"]
        self.stim_fs = int(stim.fs)
        self.stim_feature = stim.data[0, :]   # per-trial envelope arrays @ stim_fs

        g = gcd(self.stim_fs, sfreq)
        self.stim_up = sfreq // g
        self.stim_down = self.stim_fs // g

        idyom_pitch_mat = loadmat(pitch_surprisal_file, squeeze_me=True)
        idyom_onset_mat = loadmat(onset_surprisal_file, squeeze_me=True)
        unique_song_ids = np.unique(stim.stimIdxs)

        self.pitch_surprisal, self.pitch_entropy = {}, {}
        self.onset_surprisal, self.onset_entropy = {}, {}
        for song_id in unique_song_ids:
            song_name = f"audio{song_id}"
            if song_name not in idyom_pitch_mat:
                raise KeyError(f"{song_name} not found in pitch_surprisal_file")
            if song_name not in idyom_onset_mat:
                raise KeyError(f"{song_name} not found in onset_surprisal_file")
            raw_pitch = np.asarray(idyom_pitch_mat[song_name])
            raw_onset = np.asarray(idyom_onset_mat[song_name])
            self.pitch_surprisal[song_id] = np.clip(raw_pitch[0], 0, ic_clip)
            self.pitch_entropy[song_id] = raw_pitch[1]
            self.onset_surprisal[song_id] = np.clip(raw_onset[0], 0, ic_clip)
            self.onset_entropy[song_id] = raw_onset[1]

        # Cache key is (song_id, n_times), NOT song_id alone: the same song can
        # repeat across trials and each occurrence gets independently trimmed to
        # min(envelope_len, eeg_len), so two presentations of the same song are
        # not guaranteed to have the same n_times.
        self._surprisal_cache = {}

    def raw_envelope(self, trial_idx):
        return np.asarray(self.stim_feature[trial_idx], dtype=np.float64)

    def surprisal_timeseries(self, song_id, n_times):
        """The 4 surprisal/entropy arrays for `song_id`, placed onto a
        length-`n_times` grid at self.sfreq. Cached per (song_id, n_times)."""
        key = (song_id, n_times)
        if key not in self._surprisal_cache:
            midi_path = self.midi_dir / f"audio{song_id}.mid"
            self._surprisal_cache[key] = {
                'pitch_surprisal': make_surprisal_timeseries(
                    midi_path, self.pitch_surprisal[song_id], self.sfreq, n_times),
                'pitch_entropy': make_surprisal_timeseries(
                    midi_path, self.pitch_entropy[song_id], self.sfreq, n_times),
                'onset_surprisal': make_surprisal_timeseries(
                    midi_path, self.onset_surprisal[song_id], self.sfreq, n_times),
                'onset_entropy': make_surprisal_timeseries(
                    midi_path, self.onset_entropy[song_id], self.sfreq, n_times),
            }
        return self._surprisal_cache[key]


@lru_cache(maxsize=None)
def _get_stimulus_library_cached(eeg_dir, sfreq, pitch_file, onset_file, ic_clip, midi_dir):
    """lru_cache keyed on hashable (string paths + scalars), so one library is
    built per distinct set of these values and reused thereafter."""
    return _StimulusLibrary(
        eeg_dir=Path(eeg_dir), sfreq=sfreq,
        pitch_surprisal_file=Path(pitch_file), onset_surprisal_file=Path(onset_file),
        ic_clip=ic_clip, midi_dir=Path(midi_dir))


def get_stimulus_library(config):
    """Return a shared _StimulusLibrary for `config`, built once and cached
    (keyed on the resolved path strings + sfreq + ic_clip). Multiple TRFDataset
    instances across a subject loop automatically share one library."""
    return _get_stimulus_library_cached(
        str(config.paths.eeg_dir), config.sfreq,
        str(config.paths.pitch_surprisal_file), str(config.paths.onset_surprisal_file),
        config.ic_clip, str(config.paths.midi_dir))


# ── Stimulus/IDyOM alignment + trial assembly (former Dataset methods) ──────────

def align_stimulus_and_idyom(events, preprocessed_trials, lib, subject, sfreq,
                             trial_to_song_id_table, surprisal_feature_keys,
                             debug=False):
    """Resample the stimulus envelopes to `sfreq`, align each to its EEG trial
    length, derive onsets, and place the IDyOM surprisal/entropy features onto
    the same grid. Mutates `events` in place (adds 'envelope', 'onsets',
    'duration', 'eeg', and each surprisal feature) and returns it.

    Standalone version of the former Dataset._align_stimulus_and_idyom method.
    """
    eeg_trial_lengths = [t.shape[0] for t in preprocessed_trials]

    envelopes = []
    for i in range(len(events['event'])):
        env_raw = lib.raw_envelope(i)
        n_eeg = eeg_trial_lengths[i]

        env_resampled = resample_poly(env_raw, lib.stim_up, lib.stim_down)

        # Trim to the shorter of stim/EEG, matching MATLAB's min(envLen, eegLen).
        # A ~1s mismatch is normal (the EEG recording overruns the audio
        # slightly); only warn on mismatches large enough to indicate an error.
        n_min = min(len(env_resampled), n_eeg)
        diff = len(env_resampled) - n_eeg
        if abs(diff) > 4 * sfreq:
            warnings.warn(
                f"{subject} trial {i}: unusually large stim/EEG mismatch "
                f"(stim={len(env_resampled)}, EEG={n_eeg}, diff={diff} smp, "
                f"{abs(diff)/sfreq:.2f} s). Check padding removal and resampling."
            )
        if debug and diff != 0:
            print(f"  [align] Trial {i}: stim={len(env_resampled)}, EEG={n_eeg}, "
                  f"n_min={n_min} (diff={diff} smp, {diff*1000/sfreq:.1f} ms)")
        env_resampled = env_resampled[:n_min]

        if debug:
            eeg_ch0 = preprocessed_trials[i][:n_min, 0]
            sig_a = (env_resampled - env_resampled.mean()) / (env_resampled.std() + 1e-12)
            sig_b = (eeg_ch0 - eeg_ch0.mean()) / (eeg_ch0.std() + 1e-12)
            xcorr = np.correlate(sig_b, sig_a, mode='full')
            lag_smp = int(np.argmax(xcorr)) - (n_min - 1)
            lag_ms = lag_smp * 1000.0 / sfreq
            ok = -200 <= lag_ms <= 600
            print(f"  [xcorr] Trial {i}: peak lag = {lag_smp} smp "
                  f"({lag_ms:.1f} ms)  {'[OK]' if ok else '[WARNING: implausible]'}")

        time_axis = eelbrain.UTS(0, 1 / sfreq, n_min)
        envelopes.append(eelbrain.NDVar(env_resampled, (time_axis,)))

    events['envelope'] = envelopes
    events['onsets'] = [env.diff('time').clip(0) for env in envelopes]
    events['duration'] = eelbrain.Var([env.time.tstop for env in envelopes])
    events['eeg'] = eelbrain.load.mne.variable_length_epochs(
        events, 0, tstop='duration', decim=1, adjacency='auto')

    per_key_lists = {k: [] for k in surprisal_feature_keys}
    for i, stimulus_id in enumerate(events['event']):
        song_id = song_id_for_marker(stimulus_id, trial_to_song_id_table)
        time = events['envelope'][i].time
        n_times = time.nsamples
        ts = lib.surprisal_timeseries(song_id, n_times)
        for k in surprisal_feature_keys:
            per_key_lists[k].append(eelbrain.NDVar(ts[k], dims=(time,), name=k))
    for k in surprisal_feature_keys:
        events[k] = per_key_lists[k]

    return events


def build_trials(events, feature_names, subject):
    """Convert the aligned eelbrain `events` into a list of per-trial numpy
    dicts (raw, not yet z-scored), one dict per trial with keys 'eeg' plus each
    feature in `feature_names`. Returns (trials, sensor_dim, channel_names,
    n_channels). Standalone version of the former Dataset._build_trials."""
    n_trials = len(events['event'])

    trials = []
    for i in range(n_trials):
        eeg_arr = events['eeg'][i].get_data(('sensor', 'time')).T   # (T, n_ch)
        stim_arrays = {k: events[k][i].x for k in feature_names}
        eeg_arr, stim_arrays = align_trial(
            eeg_arr, stim_arrays, trial_idx=i, subject=subject)
        trials.append({'eeg': eeg_arr, **stim_arrays})

    sensor_dim = events['eeg'][0].sensor
    channel_names = list(sensor_dim.names)
    n_channels = trials[0]['eeg'].shape[1]
    return trials, sensor_dim, channel_names, n_channels


def zscore_trials(trials, feature_keys, subject, debug=False):
    """Per-trial z-score the EEG and the `feature_keys` of each trial dict,
    with boundary checks. Returns a new list of per-trial dicts holding
    {'eeg': (T,n_ch), <feature_key>: (T,), ...}, all z-scored.

    Standalone version of the former Dataset.get_trials body. Keeps all three
    checks exactly: 2-D EEG, per-feature length match, post-zscore finiteness.
    """
    out = []
    for ti, t in enumerate(trials):
        # Shape sanity at the dataset -> model boundary: EEG must be a
        # 2-D (T, n_channels) array and every feature must share its T.
        if t['eeg'].ndim != 2:
            raise ValueError(
                f"{subject} trial {ti}: expected 2-D (T, n_channels) "
                f"EEG, got shape {t['eeg'].shape} (ndim={t['eeg'].ndim})."
            )
        n_time = t['eeg'].shape[0]
        for k in feature_keys:
            if t[k].shape[0] != n_time:
                raise ValueError(
                    f"{subject} trial {ti}: feature {k!r} has "
                    f"{t[k].shape[0]} timepoints but EEG has {n_time} "
                    f"(feature shape {t[k].shape}, EEG shape {t['eeg'].shape})."
                )

        zt = {'eeg': zscore(t['eeg'])}
        for k in feature_keys:
            zt[k] = zscore(t[k])

        # Guard against non-finite values introduced by z-scoring: a
        # constant-variance channel (e.g. a dead electrode) divides by a
        # zero std and becomes all-NaN, which would otherwise propagate
        # silently into pearsonr as nan.
        for k, arr in zt.items():
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"{subject} trial {ti}: {k!r} contains non-finite "
                    f"values after z-scoring (likely a zero-variance / dead "
                    f"channel). Inspect the raw data for this subject/trial."
                )
        out.append(zt)
    return out
