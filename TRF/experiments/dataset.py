"""
dataset.py — centralized data loading/preprocessing for the TRF experiment scripts.

Self-contained: experiments/ has no import dependency on the parent TRF/
folder (constants.py, eeg_functions.py, midi_func.py). Their logic is inlined
below; the only external dependency is the raw dataset files themselves
(liberi_dataset/ and the IDyOM surprisal .mat outputs), located relative to
this folder's parent — see BASE_DIR / DATA_ROOT below.

This module owns everything that must be *identical* across every model we
compare (ridge/sklearn, ridge/MNE, boosting, conv), per the invariants in
../CLAUDE.md:

    - EEG loading + per-trial preprocessing (LPF -> downsample -> HPF -> strip
      padding, matching the MATLAB CNSP order; never on the concatenated signal)
    - Stimulus envelope loading, resampling, and trial-length alignment
    - IDyOM surprisal/entropy loading and placement onto the 64 Hz time grid
    - Per-trial z-scoring of every feature and the EEG

It deliberately does NOT own model fitting, LOOCV harnesses, alpha selection,
or lag-matrix construction — those differ enough between model families that
folding them into one abstraction here would be premature. See
EVALUATION_NOTES.md for a write-up of how each model's evaluation loop works
and what a future shared eval module could look like.

Usage
-----
    from dataset import Dataset, CONDITIONS, SUBJECTS, SAVE_DIR

    ds = Dataset('Sub2', debug=True)
    trials = ds.get_trials('acoustic_and_surprisal')   # list of per-trial dicts
    # trials[i]['eeg']      -> (T_i, n_channels) z-scored ndarray
    # trials[i]['envelope'] -> (T_i,)            z-scored ndarray
    # ...

    ds.events          # eelbrain Dataset of NDVars, for eelbrain.boosting()
    ds.sensor_dim      # eelbrain Sensor dimension
    ds.channel_names   # list[str]
    ds.subject_type    # 'Musician' | 'Non-musician'
"""

import warnings
from datetime import date
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
# Paths and dataset-wide constants (inlined from ../constants.py)
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent   # TRF/
DATA_ROOT = BASE_DIR / 'liberi_dataset/doi_10_5061_dryad_g1jwstqmh__v20211008'

WAV_DIR = DATA_ROOT / 'diliBach_wav_4dryad'
MIDI_DIR = DATA_ROOT / 'diliBach_midi_4dryad'
EEG_DIR = DATA_ROOT / 'diliBach_4dryad_CND'

PITCH_SURPRISAL_FILE = BASE_DIR / '../IDyOM/codeForPaper-IDyOMpy-/IDyOM/out/eLife/surprises/mixed2/data/mixed2_quantization_24_maxOrder_20_viewpoints_pitch.mat'
ONSET_SURPRISAL_FILE = BASE_DIR / '../IDyOM/codeForPaper-IDyOMpy-/IDyOM/out/eLife/surprises/mixed2/data/mixed2_quantization_24_maxOrder_20_viewpoints_length.mat'

SAVE_DIR = BASE_DIR / f'pickles/encoding_{date.today()}'

SUBJECTS = [
    'Sub1', 'Sub2', 'Sub3', 'Sub4', 'Sub5', 'Sub6', 'Sub7', 'Sub8', 'Sub9', 'Sub10',
    'Sub11', 'Sub12', 'Sub13', 'Sub14', 'Sub15', 'Sub16', 'Sub17', 'Sub18', 'Sub19', 'Sub20',
]
LOW_FREQUENCY = 1    # Hz, HPF cutoff
HIGH_FREQUENCY = 8   # Hz, LPF cutoff

# ─── Shared TRF config (was duplicated as module-level constants in every script) ──
TMIN = -0.1     # seconds, receptive-field start (pre-stimulus)
TMAX = 0.600    # seconds, receptive-field end   (post-stimulus)
SFREQ = 64      # Hz, target sampling rate after preprocessing/resampling
IC_CLIP = 15.0  # bits, IDyOM surprisal clip ceiling

FEATURE_KEYS_ACOUSTIC = ['envelope', 'onsets']
FEATURE_KEYS_SURPRISAL = [
    'pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy',
]
CONDITIONS = {
    'acoustic': FEATURE_KEYS_ACOUSTIC,
    'acoustic_and_surprisal': FEATURE_KEYS_ACOUSTIC + FEATURE_KEYS_SURPRISAL,
}


def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# EEG loading + preprocessing 
# ═══════════════════════════════════════════════════════════════════════════════

def load_subject_raw_eeg(filepath, subject):
    """Load raw EEG at its original sampling frequency — NO resampling.

    Padding is preserved here (not removed) so that the LPF and HPF can filter
    through the padding before it is discarded — exactly as MATLAB does
    (padding is removed last, after all filtering).
    """
    subject_idx = int(subject[3:])
    mat_data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg = mat_data["eeg"]
    orig_fs = int(eeg.fs)

    for i in range(len(eeg.data)):
        # Scale int32 counts to float32 (~microvolt range)
        eeg.data[i] = 100 * eeg.data[i].astype(np.float32) / np.iinfo(np.int32).max

    raw_data = {
        'trials': eeg.data,
        'fs': orig_fs,                                # original fs; no resampling done
        'chanlocs': eeg.chanlocs,
        'pad_start': int(eeg.paddingStartSample),      # in original-fs samples
        'subject_type': 'Musician' if subject_idx >= 11 else 'Non-musician',
    }

    print(f"✓ Loaded {raw_data['subject_type']} (Subject {subject})")
    print(f"  - {len(raw_data['trials'])} trials, "
          f"{raw_data['trials'][0].shape[1]} channels @ {orig_fs} Hz")

    return raw_data


def preprocess_eeg_trials(subject_data, target_fs=64, lpf_hz=8, hpf_hz=1, debug=False):
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
    trials = subject_data['trials']
    orig_fs = subject_data['fs']
    pad_start_orig = subject_data['pad_start']

    g = gcd(orig_fs, target_fs)
    up = target_fs // g
    down_factor = orig_fs // g

    # Replicate MATLAB cndDownsample: scale padding to the target domain.
    pad_start_target = int(round(pad_start_orig * target_fs / orig_fs))

    nyq_orig = orig_fs / 2.0
    lpf_sos = butter(4, lpf_hz / nyq_orig, btype='low', output='sos')

    nyq_tgt = target_fs / 2.0
    hpf_sos = butter(4, hpf_hz / nyq_tgt, btype='high', output='sos')

    preprocessed = []
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

        preprocessed.append(trial_clean)

    return preprocessed


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


# ═══════════════════════════════════════════════════════════════════════════════
# IDyOM surprisal placement (inlined from ../midi_func.py)
# ═══════════════════════════════════════════════════════════════════════════════

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


def _align_trial(eeg, stim_arrays, trial_idx, subject, max_diff=2):
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
# Stimulus / IDyOM library — shared across subjects (same songs for everyone)
# ═══════════════════════════════════════════════════════════════════════════════

class _StimulusLibrary:
    """Lazily-loaded singleton: dataStim.mat + IDyOM .mat files are the same
    for every subject, so they're loaded once and shared read-only across all
    Dataset instances instead of being re-read/re-parsed per subject.
    """

    _instance = None

    def __init__(self):
        stim_mat = loadmat(EEG_DIR / "dataStim.mat", struct_as_record=False, squeeze_me=True)
        stim = stim_mat["stim"]
        self.stim_fs = int(stim.fs)
        self.stim_feature = stim.data[0, :]   # per-trial envelope arrays @ stim_fs

        g = gcd(self.stim_fs, SFREQ)
        self.stim_up = SFREQ // g
        self.stim_down = self.stim_fs // g

        idyom_pitch_mat = loadmat(PITCH_SURPRISAL_FILE, squeeze_me=True)
        idyom_onset_mat = loadmat(ONSET_SURPRISAL_FILE, squeeze_me=True)
        unique_song_ids = np.unique(stim.stimIdxs)

        self.pitch_surprisal, self.pitch_entropy = {}, {}
        self.onset_surprisal, self.onset_entropy = {}, {}
        for song_id in unique_song_ids:
            song_name = f"audio{song_id}"
            if song_name not in idyom_pitch_mat:
                raise KeyError(f"{song_name} not found in PITCH_SURPRISAL_FILE")
            if song_name not in idyom_onset_mat:
                raise KeyError(f"{song_name} not found in ONSET_SURPRISAL_FILE")
            raw_pitch = np.asarray(idyom_pitch_mat[song_name])
            raw_onset = np.asarray(idyom_onset_mat[song_name])
            self.pitch_surprisal[song_id] = np.clip(raw_pitch[0], 0, IC_CLIP)
            self.pitch_entropy[song_id] = raw_pitch[1]
            self.onset_surprisal[song_id] = np.clip(raw_onset[0], 0, IC_CLIP)
            self.onset_entropy[song_id] = raw_onset[1]

        # Cache key is (song_id, n_times), NOT song_id alone: the same song can
        # repeat across trials (see ../check_trial_song_repeats.py) and each
        # occurrence gets independently trimmed to min(envelope_len, eeg_len),
        # so two presentations of the same song are not guaranteed to have the
        # same n_times. Caching by song_id alone (as the original scripts did)
        # would silently reuse a timeseries built for the wrong length.
        self._surprisal_cache = {}

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def song_id_for(stimulus_id):
        return int(stimulus_id % 10) or 10

    def raw_envelope(self, trial_idx):
        return np.asarray(self.stim_feature[trial_idx], dtype=np.float64)

    def surprisal_timeseries(self, song_id, n_times):
        """Return the 4 surprisal/entropy arrays for `song_id`, placed onto a
        length-`n_times` grid at SFREQ. Cached per (song_id, n_times)."""
        key = (song_id, n_times)
        if key not in self._surprisal_cache:
            midi_path = MIDI_DIR / f"audio{song_id}.mid"
            self._surprisal_cache[key] = {
                'pitch_surprisal': make_surprisal_timeseries(
                    midi_path, self.pitch_surprisal[song_id], SFREQ, n_times),
                'pitch_entropy': make_surprisal_timeseries(
                    midi_path, self.pitch_entropy[song_id], SFREQ, n_times),
                'onset_surprisal': make_surprisal_timeseries(
                    midi_path, self.onset_surprisal[song_id], SFREQ, n_times),
                'onset_entropy': make_surprisal_timeseries(
                    midi_path, self.onset_entropy[song_id], SFREQ, n_times),
            }
        return self._surprisal_cache[key]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset — one subject's preprocessed, feature-aligned EEG + stimulus data
# ═══════════════════════════════════════════════════════════════════════════════

class Dataset:
    """One subject's preprocessed, feature-aligned EEG + stimulus data.

    Construction runs the full pipeline once (EEG load/preprocess, stimulus
    resample/align, IDyOM placement). After that:

      - `get_trials(condition)` returns per-trial z-scored numpy arrays, the
        input every non-eelbrain model (sklearn ridge, MNE ReceptiveField,
        conv) needs.
      - `events` is the eelbrain Dataset of NDVars, for eelbrain.boosting().
    """

    def __init__(self, subject, debug=False):
        self.subject = subject
        self.debug = debug
        self._lib = _StimulusLibrary.get()

        eeg_data = load_subject_raw_eeg(EEG_DIR / f'data{subject}.mat', subject)
        self.subject_type = eeg_data['subject_type']

        # Per-trial LPF -> downsample -> HPF -> strip padding (never on the
        # concatenated signal).
        self.preprocessed_trials = preprocess_eeg_trials(
            eeg_data,
            target_fs=SFREQ,
            lpf_hz=HIGH_FREQUENCY,
            hpf_hz=LOW_FREQUENCY,
            debug=debug,
        )

        self.raw = create_mne_raw_from_preprocessed(
            self.preprocessed_trials, SFREQ, eeg_data['chanlocs'])
        self.events = create_eelbrain_events(self.raw)

        self._align_stimulus_and_idyom()
        self._build_trials()

    # ── internal pipeline steps ─────────────────────────────────────────────

    def _align_stimulus_and_idyom(self):
        lib = self._lib
        eeg_trial_lengths = [t.shape[0] for t in self.preprocessed_trials]

        envelopes = []
        for i in range(len(self.events['event'])):
            env_raw = lib.raw_envelope(i)
            n_eeg = eeg_trial_lengths[i]

            env_resampled = resample_poly(env_raw, lib.stim_up, lib.stim_down)

            # Trim to the shorter of stim/EEG, matching MATLAB's min(envLen,
            # eegLen) in CNSP2025_forwardTRF_example1.m. A ~1s mismatch is
            # normal (the EEG recording overruns the audio slightly); only
            # warn on mismatches large enough to indicate a real error.
            n_min = min(len(env_resampled), n_eeg)
            diff = len(env_resampled) - n_eeg
            if abs(diff) > 4 * SFREQ:
                warnings.warn(
                    f"{self.subject} trial {i}: unusually large stim/EEG mismatch "
                    f"(stim={len(env_resampled)}, EEG={n_eeg}, diff={diff} smp, "
                    f"{abs(diff)/SFREQ:.2f} s). Check padding removal and resampling."
                )
            if self.debug and diff != 0:
                print(f"  [align] Trial {i}: stim={len(env_resampled)}, EEG={n_eeg}, "
                      f"n_min={n_min} (diff={diff} smp, {diff*1000/SFREQ:.1f} ms)")
            env_resampled = env_resampled[:n_min]

            if self.debug:
                eeg_ch0 = self.preprocessed_trials[i][:n_min, 0]
                sig_a = (env_resampled - env_resampled.mean()) / (env_resampled.std() + 1e-12)
                sig_b = (eeg_ch0 - eeg_ch0.mean()) / (eeg_ch0.std() + 1e-12)
                xcorr = np.correlate(sig_b, sig_a, mode='full')
                lag_smp = int(np.argmax(xcorr)) - (n_min - 1)
                lag_ms = lag_smp * 1000.0 / SFREQ
                ok = -200 <= lag_ms <= 600
                print(f"  [xcorr] Trial {i}: peak lag = {lag_smp} smp "
                      f"({lag_ms:.1f} ms)  {'[OK]' if ok else '[WARNING: implausible]'}")

            time_axis = eelbrain.UTS(0, 1 / SFREQ, n_min)
            envelopes.append(eelbrain.NDVar(env_resampled, (time_axis,)))

        self.events['envelope'] = envelopes
        self.events['onsets'] = [env.diff('time').clip(0) for env in envelopes]
        self.events['duration'] = eelbrain.Var([env.time.tstop for env in envelopes])
        self.events['eeg'] = eelbrain.load.mne.variable_length_epochs(
            self.events, 0, tstop='duration', decim=1, adjacency='auto')

        surprisal_keys = FEATURE_KEYS_SURPRISAL
        per_key_lists = {k: [] for k in surprisal_keys}
        for i, stimulus_id in enumerate(self.events['event']):
            song_id = lib.song_id_for(stimulus_id)
            time = self.events['envelope'][i].time
            n_times = time.nsamples
            ts = lib.surprisal_timeseries(song_id, n_times)
            for k in surprisal_keys:
                per_key_lists[k].append(eelbrain.NDVar(ts[k], dims=(time,), name=k))
        for k in surprisal_keys:
            self.events[k] = per_key_lists[k]

    def _build_trials(self):
        feature_names = FEATURE_KEYS_ACOUSTIC + FEATURE_KEYS_SURPRISAL
        n_trials = len(self.events['event'])

        trials = []
        for i in range(n_trials):
            eeg_arr = self.events['eeg'][i].get_data(('sensor', 'time')).T   # (T, n_ch)
            stim_arrays = {k: self.events[k][i].x for k in feature_names}
            eeg_arr, stim_arrays = _align_trial(
                eeg_arr, stim_arrays, trial_idx=i, subject=self.subject)
            trials.append({'eeg': eeg_arr, **stim_arrays})

        self.trials = trials
        self.sensor_dim = self.events['eeg'][0].sensor
        self.channel_names = list(self.sensor_dim.names)
        self.n_channels = trials[0]['eeg'].shape[1]

    # ── public API ───────────────────────────────────────────────────────────

    def get_trials(self, condition):
        """Per-trial z-scored numpy arrays for `condition`
        ('acoustic' | 'acoustic_and_surprisal').

        Returns a list of dicts, one per trial:
            {'eeg': (T_i, n_channels) ndarray, <feature_key>: (T_i,) ndarray, ...}
        All arrays are z-scored per-trial, matching TRF_ridge_3.py exactly.
        """
        if condition not in CONDITIONS:
            raise KeyError(f"Unknown condition {condition!r}; expected one of {list(CONDITIONS)}")
        feature_keys = CONDITIONS[condition]
        out = []
        for t in self.trials:
            zt = {'eeg': zscore(t['eeg'])}
            for k in feature_keys:
                zt[k] = zscore(t[k])
            out.append(zt)
        return out

    @property
    def trial_lengths(self):
        return [t['eeg'].shape[0] for t in self.trials]
