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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from scipy.io import loadmat
from scipy.signal import resample_poly, butter, sosfiltfilt, hilbert

import eelbrain
import mne
from mne.channels import make_dig_montage


# ═══════════════════════════════════════════════════════════════════════════════
# 1. General-purpose utilities (not liberi-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def compute_envelope_from_audio(path, target_fs):
    """Broadband amplitude envelope of an audio file (.wav/.mp3/...),
    resampled to `target_fs`. Used when a dataset ships raw stimulus audio
    instead of a precomputed envelope (e.g. no equivalent of liberi's
    dataStim.mat): loads the mono mixdown at its native sample rate, takes
    the analytic-signal (Hilbert) magnitude as the envelope, then
    polyphase-resamples to `target_fs`.

    Uses soundfile (libsndfile, incl. its mp3 support) rather than
    librosa.load: librosa's audio-loading path unconditionally imports numba,
    which at the time of writing rejects the numpy version this project
    otherwise pins, making librosa.load unusable in that environment."""
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    envelope = np.abs(hilbert(audio.astype(np.float64)))
    g = gcd(target_fs, sr)
    return resample_poly(envelope, target_fs // g, sr // g)


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

def load_subject_raw_eeg(filepath, subject, trial_to_stimulus=None):
    """Load one subject's raw EEG, dispatching on file format.

    Thin dispatcher: `.mat` files go through _load_eeg_from_mat; any other
    extension is handed to _load_eeg_from_other_format. Both return the same
    `eeg_data` dict shape, so everything downstream (preprocess_eeg_trials,
    create_mne_raw_from_preprocessed) is format-agnostic:

        {
          'trials':         list of per-trial (n_time, n_channels) arrays, in
                             raw counts/volts BEFORE any filtering (still
                             padded),
          'fs':              int, original sampling rate in Hz (no resampling
                             done),
          'chanlocs':        iterable of channel objects/dicts exposing a
                             `.labels` name and `.X`/`.Y`/`.Z` position
                             attributes,
          'pad_start':       int, leading-padding sample count expressed at
                             `fs`,
          'stimulus_paths':  optional, list of per-trial stimulus audio paths
                             (or None entries) — only present for formats
                             whose stimulus identity isn't a simple trial
                             index (see _load_eeg_from_edf).
        }

    subject_type is deliberately NOT part of this dict — it's subject metadata,
    not EEG-file content; TRFDataset looks it up from config.subject_type.

    trial_to_stimulus : only used by non-.mat formats that can't determine
        stimulus identity from the EEG file itself (see _load_eeg_from_edf);
        ignored for `.mat`.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == '.mat':
        return _load_eeg_from_mat(filepath, subject)
    return _load_eeg_from_other_format(filepath, subject, trial_to_stimulus)


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


def _load_eeg_from_other_format(filepath, subject, trial_to_stimulus=None):
    """Dispatch non-.mat EEG formats. Each implementation must return the same
    `eeg_data` dict shape that _load_eeg_from_mat produces, so it flows
    through preprocess_eeg_trials unchanged:

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
    suffix = filepath.suffix.lower()
    if suffix == '.edf':
        return _load_eeg_from_edf(filepath, subject, trial_to_stimulus)
    raise NotImplementedError(
        f"EEG loading for '{filepath.suffix}' files is not implemented "
        f"(subject {subject}, path {filepath}). Only '.mat' and '.edf' are "
        "supported today; extend _load_eeg_from_other_format to return the "
        "documented eeg_data dict shape for this format."
    )


# Event-marker code for "trial start, music played to participant" in the
# ds002725 (Daly et al. 2019) BIDS release's *_events.tsv files.
_DALY_TRIAL_START_MARKER = 768


def _load_eeg_from_edf(filepath, subject, trial_to_stimulus):
    """Load raw EEG from a BIDS-style continuous .edf recording (one file per
    subject x task, e.g. ds002725/Daly et al. 2019), segmenting trials from
    the sibling `*_events.tsv`'s trial-start markers rather than receiving
    pre-segmented trials the way _load_eeg_from_mat does.

    trial_to_stimulus : list[str | Path | None] | None
        One entry per detected trial, giving the path to that trial's
        stimulus audio file. There is no reliable way to decode stimulus
        identity from this dataset's released EEG channels — the `music`/
        `trialtype` channels documented as carrying a stimulus code are flat
        at the ADC floor in the public release (confirmed empirically across
        subjects/tasks), and the private "paradigm record" the original
        analysis script reads instead isn't included. So the mapping must be
        supplied by the caller (e.g. from config_daly.yaml's
        `trial_to_stimulus`); a None entry (or trial_to_stimulus=None
        entirely) is allowed and simply means no stimulus/envelope features
        can be computed for that trial — it only becomes an error if
        align_stimulus_and_idyom later needs an envelope for it.

    Returns the eeg_data dict shape documented on _load_eeg_from_other_format,
    plus a 'stimulus_paths' key (list parallel to 'trials').
    """
    events_path = filepath.parent / filepath.name.replace('_eeg.edf', '_events.tsv')
    channels_path = filepath.parent / filepath.name.replace('_eeg.edf', '_channels.tsv')
    events_df = pd.read_csv(events_path, sep='\t')
    channels_df = pd.read_csv(channels_path, sep='\t')
    eeg_channel_names_tsv = channels_df.loc[channels_df['type'] == 'EEG', 'name'].tolist()

    raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR')
    sf = int(raw.info['sfreq'])

    # channels.tsv's declared names can differ in case from the actual EDF
    # channel labels (e.g. this dataset's channels.tsv says 'FP1'/'FP2' but
    # the EDF itself has 'Fp1'/'Fp2') -- resolve case-insensitively against
    # raw.ch_names rather than assuming an exact string match.
    edf_name_by_lower = {name.lower(): name for name in raw.ch_names}
    eeg_channel_names = []
    for tsv_name in eeg_channel_names_tsv:
        edf_name = edf_name_by_lower.get(tsv_name.lower())
        if edf_name is None:
            raise KeyError(
                f"{subject}: channel {tsv_name!r} from {channels_path.name} "
                f"not found among the EDF's channels ({filepath.name})."
            )
        eeg_channel_names.append(edf_name)

    onsets = events_df.loc[
        events_df['trial_type'] == _DALY_TRIAL_START_MARKER, 'onset'].to_numpy()
    if len(onsets) == 0:
        raise ValueError(
            f"{subject} {filepath.name}: no trial-start "
            f"(trial_type == {_DALY_TRIAL_START_MARKER}) markers found in "
            f"{events_path.name}."
        )
    # Some tasks double-fire this marker (rising+falling edge of the same
    # trigger pulse); the original MATLAB pipeline for this dataset takes
    # every other one (`trialInds(1:2:end)`) as the true trial onset. Detect
    # doubling via a short median inter-onset gap rather than assuming it.
    diffs = np.diff(onsets)
    if len(diffs) and np.median(diffs) < 1.0:
        onsets = onsets[0::2]

    total_duration = raw.n_times / sf
    trial_bounds = list(zip(onsets, list(onsets[1:]) + [total_duration]))

    if trial_to_stimulus is not None and len(trial_to_stimulus) != len(trial_bounds):
        raise ValueError(
            f"{subject} {filepath.name}: trial_to_stimulus has "
            f"{len(trial_to_stimulus)} entries but {len(trial_bounds)} trials "
            "were detected from the events file."
        )

    eeg_full = raw.get_data(picks=eeg_channel_names).T  # (n_time, n_channels), volts
    eeg_full = (1e6 * eeg_full).astype(np.float32)      # match the CND pipeline's microvolt convention

    trials = [eeg_full[int(round(s * sf)):int(round(e * sf))] for s, e in trial_bounds]

    montage_pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    chanlocs = []
    for name in eeg_channel_names:
        pos = montage_pos.get(name)
        if pos is None:
            raise KeyError(
                f"{subject}: channel {name!r} has no position in the "
                "standard_1020 montage; add it manually if this dataset uses "
                "a non-standard label for a known electrode."
            )
        # create_mne_raw_from_preprocessed reads positions back out as
        # [ch.Y, ch.X, ch.Z] (matching the EEGLAB chanlocs convention the
        # liberi loader's chanlocs use) — pre-swap X/Y here so that reswap
        # yields this montage's true head-frame position unchanged.
        chanlocs.append(SimpleNamespace(labels=name, X=pos[1], Y=pos[0], Z=pos[2]))

    eeg_data = {
        'trials': trials,
        'fs': sf,
        'chanlocs': chanlocs,
        'pad_start': 0,
        'stimulus_paths': (list(trial_to_stimulus) if trial_to_stimulus is not None
                            else [None] * len(trials)),
    }

    print(f"✓ Loaded raw EEG ({subject}, {filepath.stem}): {len(trials)} trials, "
          f"{len(eeg_channel_names)} channels @ {sf} Hz")

    return eeg_data


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
    """Stimulus envelope (+ optionally IDyOM surprisal) source, parsed/primed
    once and reused across subjects. Two source types, selected by
    `source_type`:

      'mat' (default, the liberi/CNSP case) — dataStim.mat + the IDyOM .mat
        files, identical for every subject. `raw_envelope(trial_idx)` indexes
        the precomputed per-trial envelope array directly.

      'audio_files' — for datasets that ship raw stimulus audio (.mp3/.wav)
        with no precomputed envelope and no symbolic/MIDI surprisal source.
        `raw_envelope(trial_idx, stimulus_path=...)` computes the envelope
        from the given audio file on demand (compute_envelope_from_audio,
        already resampled to `sfreq`) and caches it by path, since the same
        stimulus can recur across trials/subjects. `surprisal_timeseries` is
        not supported in this mode.

    This is a plain class — the old singleton machinery moved out to
    get_stimulus_library() (an lru_cache factory), so sharing is explicit and
    injectable. All the config values it needs are passed in; it reads no
    module globals.
    """

    def __init__(self, sfreq, source_type='mat', eeg_dir=None,
                 pitch_surprisal_file=None, onset_surprisal_file=None,
                 ic_clip=None, midi_dir=None):
        self.sfreq = sfreq
        self.source_type = source_type

        if source_type == 'audio_files':
            self._envelope_cache = {}
            return
        if source_type != 'mat':
            raise ValueError(
                f"Unknown source_type {source_type!r}; expected 'mat' or "
                "'audio_files'.")

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

    def raw_envelope(self, trial_idx, stimulus_path=None):
        if self.source_type == 'audio_files':
            if stimulus_path is None:
                raise ValueError(
                    f"trial {trial_idx}: no stimulus file available for this "
                    "trial (missing/null trial_to_stimulus entry). Fill in "
                    "the real mapping in the dataset config before this "
                    "trial can be used."
                )
            path = Path(stimulus_path)
            if path not in self._envelope_cache:
                self._envelope_cache[path] = compute_envelope_from_audio(path, self.sfreq)
            return self._envelope_cache[path]
        return np.asarray(self.stim_feature[trial_idx], dtype=np.float64)

    def surprisal_timeseries(self, song_id, n_times):
        """The 4 surprisal/entropy arrays for `song_id`, placed onto a
        length-`n_times` grid at self.sfreq. Cached per (song_id, n_times).
        Only available for source_type='mat'."""
        if self.source_type != 'mat':
            raise NotImplementedError(
                "surprisal_timeseries requires source_type='mat' (IDyOM/MIDI "
                "surprisal); a dataset with no symbolic score should leave "
                "feature_keys.surprisal empty so this is never called."
            )
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
def _get_stimulus_library_cached(source_type, eeg_dir, sfreq, pitch_file, onset_file,
                                  ic_clip, midi_dir):
    """lru_cache keyed on hashable (string paths + scalars), so one library is
    built per distinct set of these values and reused thereafter."""
    if source_type == 'audio_files':
        return _StimulusLibrary(sfreq=sfreq, source_type='audio_files')
    return _StimulusLibrary(
        sfreq=sfreq, source_type='mat', eeg_dir=Path(eeg_dir),
        pitch_surprisal_file=Path(pitch_file), onset_surprisal_file=Path(onset_file),
        ic_clip=ic_clip, midi_dir=Path(midi_dir))


def get_stimulus_library(config):
    """Return a shared _StimulusLibrary for `config`, built once and cached.
    Multiple TRFDataset instances across a subject loop automatically share
    one library. Dispatches on config.stimulus_source_type ('mat', the
    default, or 'audio_files')."""
    source_type = getattr(config, 'stimulus_source_type', 'mat')
    if source_type == 'audio_files':
        return _get_stimulus_library_cached(
            'audio_files', None, config.sfreq, None, None, None, None)
    return _get_stimulus_library_cached(
        'mat', str(config.paths.eeg_dir), config.sfreq,
        str(config.paths.pitch_surprisal_file), str(config.paths.onset_surprisal_file),
        config.ic_clip, str(config.paths.midi_dir))


# ── Stimulus/IDyOM alignment + trial assembly (former Dataset methods) ──────────

def align_stimulus_and_idyom(events, preprocessed_trials, lib, subject, sfreq,
                             trial_to_song_id_table, surprisal_feature_keys,
                             stimulus_paths=None, debug=False):
    """Resample the stimulus envelopes to `sfreq`, align each to its EEG trial
    length, derive onsets, and place the IDyOM surprisal/entropy features onto
    the same grid. Mutates `events` in place (adds 'envelope', 'onsets',
    'duration', 'eeg', and each surprisal feature) and returns it.

    stimulus_paths : optional list parallel to the trials, giving each
        trial's stimulus audio path — only needed for an 'audio_files'-mode
        `lib` (see _StimulusLibrary), where trial_idx alone isn't enough to
        find the right file; ignored by a 'mat'-mode `lib`.

    Standalone version of the former Dataset._align_stimulus_and_idyom method.
    """
    eeg_trial_lengths = [t.shape[0] for t in preprocessed_trials]
    already_at_target_fs = getattr(lib, 'source_type', 'mat') == 'audio_files'

    envelopes = []
    for i in range(len(events['event'])):
        stim_path = stimulus_paths[i] if stimulus_paths is not None else None
        env_raw = lib.raw_envelope(i, stimulus_path=stim_path)
        n_eeg = eeg_trial_lengths[i]

        # audio_files-mode envelopes are already resampled to `sfreq` inside
        # compute_envelope_from_audio (each stimulus has its own native
        # sample rate, so there's no single up/down ratio to apply here).
        env_resampled = env_raw if already_at_target_fs else \
            resample_poly(env_raw, lib.stim_up, lib.stim_down)

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

    # Skip entirely for datasets with no surprisal features (e.g. no symbolic
    # score): song_id_for_marker below assumes a per-trial trial_to_song_id
    # entry, which a dataset like ds002725 (single repeating marker code, no
    # such table) doesn't have and doesn't need.
    if surprisal_feature_keys:
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
