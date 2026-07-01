"""
Gradient Frequency Neural Network (GFNN) Oscillator — 8 oscillator types
=========================================================================
  1. Damped DLC, weak forcing (α= -1.0, β₁=2.5, β₂= -1, ε=1, F=0.1)
  2. Damped DLC, intermediate forcing (α= -1.0, β₁=2.5, β₂= -1, ε=1, F=0.2)
  3. Critical           (α= 0.0, β₁=-100, β₂= 0, ε=1, F=0.2)
  4. Limit cycle, weak forcing (α= 1.0, β₁=-100, β₂= 0, ε=1, F=0.02)
  5. Limit cycle, strong forcing (α= 1.0, β₁=-100, β₂= 0, ε=1, F=0.2)
  6. Double limit cycle, weak forcing (α= -1.0, β₁=4, β₂=-1, ε=1, F=0.1)
  7. Double limit cycle, intermediate forcing (α= -1.0, β₁=4, β₂=-1, ε=1, F=0.3)
  8. Double limit cycle, strong forcing (α= -1.0, β₁=4, β₂=-1, ε=1, F=1.5)
"""

import os
from pathlib import Path
from datetime import date

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
from joblib import Parallel, delayed
import mne
from mne.channels import make_dig_montage
import eelbrain

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path('NRT').resolve().parent
DATA_ROOT = BASE_DIR / '../TRF/liberi_dataset/doi_10_5061_dryad_g1jwstqmh__v20211008'
WAV_DIR   = DATA_ROOT / 'diliBach_wav_4dryad'
EEG_DIR   = DATA_ROOT / 'diliBach_4dryad_CND'

LOW_FREQUENCY  = 1
HIGH_FREQUENCY = 8
N_JOBS         = -1   # use all available cores for GFNN trial parallelism

SUBJECTS = [f'Sub{i}' for i in range(1, 2)]

OSCILLATOR_CONFIGS = {
    # 'double_limit_cycle_strong':       dict(alpha=-1.0,  r=1.0, beta1=4.0,  beta2=-1.0, epsilon=1.0, F=1.5),
    # 'damped_dlc_weak':                 dict(alpha=-1.0, r=1.0, beta1=2.5, beta2=-1.0,  epsilon=1.0, F=0.1),
    # 'damped_dlc_intermediate':         dict(alpha=-1.0, r=1.0, beta1=2.5, beta2=-1.0,  epsilon=1.0, F=0.2),
    'critical':                        dict(alpha=0.0,  r=1.0, beta1=-100.0, beta2=0.0,  epsilon=1.0, F=0.2),
    'limit_cycle_weak':                dict(alpha=1.0,  r=1.0, beta1=-100.0, beta2=0.0,  epsilon=1.0, F=0.02),
    'limit_cycle_strong':              dict(alpha=1.0,  r=1.0, beta1=-100.0, beta2=0.0,  epsilon=1.0, F=0.2),
    'double_limit_cycle_weak':         dict(alpha=-1.0,  r=1.0, beta1=4.0,  beta2=-1.0, epsilon=1.0, F=0.1),
    'double_limit_cycle_intermediate': dict(alpha=-1.0,  r=1.0, beta1=4.0,  beta2=-1.0, epsilon=1.0, F=0.3),
}

# ─── EEG loading ──────────────────────────────────────────────────────────────
def load_subject_raw_eeg(filepath, subject):
    subject_idx = int(subject[3:])
    mat_data    = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg         = mat_data["eeg"]
    target_fs   = 100
    orig_fs     = int(eeg.fs)
    resample_needed = orig_fs != target_fs

    for i in range(len(eeg.data)):
        trial_data = 100 * eeg.data[i].astype(np.float32) / np.iinfo(np.int32).max
        if resample_needed:
            n_samples  = int(trial_data.shape[0] * target_fs / orig_fs)
            trial_data = resample(trial_data, n_samples, axis=0)
        eeg.data[i] = trial_data

    raw_data = {
        'trials':       eeg.data,
        'fs':           target_fs,
        'chanlocs':     eeg.chanlocs,
        'pad_start':    int(eeg.paddingStartSample * target_fs / orig_fs)
                        if resample_needed else int(eeg.paddingStartSample),
        'subject_type': 'Musician' if subject_idx >= 11 else 'Non-musician',
    }
    print(f"✓ Loaded {raw_data['subject_type']} ({subject}): "
          f"{len(raw_data['trials'])} trials, "
          f"{raw_data['trials'][0].shape[1]} channels")
    return raw_data


def create_mne_raw_from_loaded(subject_data):
    trials    = subject_data['trials']
    sfreq     = subject_data['fs']
    pad_start = subject_data['pad_start']
    chanlocs  = subject_data['chanlocs']

    ch_names, positions = [], []
    for ch in chanlocs:
        ch_names.append(ch.labels)
        if hasattr(ch, 'X') and hasattr(ch, 'Y') and hasattr(ch, 'Z'):
            positions.append([ch.Y, ch.X, ch.Z])

    all_trials, trial_lengths = [], []
    for trial in trials:
        trial_clean = trial[pad_start:, :].T
        all_trials.append(trial_clean)
        trial_lengths.append(trial_clean.shape[1])

    eeg_continuous        = np.hstack(all_trials)
    n_channels, n_samples = eeg_continuous.shape

    stim_data      = np.zeros((1, n_samples))
    current_sample = 0
    for i in range(30):
        marker_sample               = 1 if current_sample == 0 else current_sample
        stim_data[0, marker_sample] = i + 1
        current_sample             += trial_lengths[i]

    data_with_stim = np.vstack([eeg_continuous, stim_data])
    all_ch_names   = ch_names + ['STI']
    ch_types       = ['eeg'] * n_channels + ['stim']

    info = mne.create_info(ch_names=all_ch_names, sfreq=sfreq, ch_types=ch_types)
    raw  = mne.io.RawArray(data_with_stim, info)
    raw.set_montage(make_dig_montage(
        ch_pos=dict(zip(ch_names, positions)), coord_frame='head'
    ))
    return raw


def create_eelbrain_events(raw):
    mne_events = mne.find_events(raw, stim_channel='STI', verbose=False)
    events = eelbrain.Dataset({
        'i_start': mne_events[:, 0],
        'trigger':  mne_events[:, 2],
        'event':    mne_events[:, 2],
    })
    events.info['raw'] = raw
    return events

# ─── GFNN ODE ─────────────────────────────────────────────────────────────────
def _rhs(z, x_t, freqs, alpha, r, beta1, beta2, epsilon):
    z2    = np.abs(z) ** 2
    z4    = z2 ** 2
    denom = 1.0 - epsilon * z2
    denom = np.where(
        np.abs(denom) < 1e-12,
        np.sign(denom.real + 1e-30) * 1e-12,
        denom,
    )
    nl = alpha + 1j * 2.0 * np.pi * r + beta1 * z2 + (epsilon * beta2 * z4) / denom
    return freqs * (z * nl + x_t)


def _rk4_step(z, x_n, x_half, x_n1, freqs, alpha, r, beta1, beta2, epsilon, dt):
    k1 = _rhs(z,              x_n,    freqs, alpha, r, beta1, beta2, epsilon)
    k2 = _rhs(z + 0.5*dt*k1, x_half, freqs, alpha, r, beta1, beta2, epsilon)
    k3 = _rhs(z + 0.5*dt*k2, x_half, freqs, alpha, r, beta1, beta2, epsilon)
    k4 = _rhs(z +     dt*k3, x_n1,   freqs, alpha, r, beta1, beta2, epsilon)
    return z + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def integrate_gfnn_amplitude(x, freqs, integration_fs=200.0, alpha=-0.1, r=1.0,
                              beta1=-1.0, beta2=0.0, epsilon=1.0, z0=None):
    """
    Like integrate_gfnn but returns only |z| (float32) instead of the full
    complex trajectory — halves peak memory per trial. Has its own sampling rate
    """
    x     = np.asarray(x,     dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    N     = len(x)
    dt    = 1.0 / integration_fs

    N_orig = len(x)
    if integration_fs != 100.0:
        x = resample(x, int(N_orig * integration_fs / 100.0))
    N = len(x)
    
    if z0 is None:
        rng = np.random.default_rng(0)
        z0  = (rng.standard_normal(len(freqs)) * 1e-6
               + 1j * rng.standard_normal(len(freqs)) * 1e-6)

    # Store amplitude only — no complex array kept in memory
    Z_amp = np.zeros((N, len(freqs)), dtype=np.float32)
    z     = np.asarray(z0, dtype=complex)
    Z_amp[0] = np.abs(z).astype(np.float32)

    for n in range(N - 1):
        x_n, x_n1 = x[n], x[n + 1]
        z = _rk4_step(z, x_n, 0.5 * (x_n + x_n1), x_n1,
                      freqs, alpha, r, beta1, beta2, epsilon, dt)
        Z_amp[n + 1] = np.abs(z).astype(np.float32)

    return Z_amp


def _integrate_trial(trial_idx, envelope_array, time_dim, freqs,
                     freq_dim, osc_params, integration_fs=200.0):
    """Worker function: integrates one trial, returns an NDVar. Pickle-safe."""
    
    Z_amp_hires = integrate_gfnn_amplitude(
        envelope_array, freqs, integration_fs=integration_fs, **osc_params
    )

    # Downsample from integration_fs back to 100 Hz
    n_samples_100hz = time_dim.nsamples
    Z_amp = resample(Z_amp_hires, n_samples_100hz, axis=0).astype(np.float32)

    if Z_amp.shape[0] != time_dim.nsamples:
        raise ValueError(
            f"Trial {trial_idx+1}: Z n_times={Z_amp.shape[0]} "
            f"!= envelope n_times={time_dim.nsamples}."
        )

    # Guard: catch any NaN/Inf before they reach boosting
    n_bad = np.sum(~np.isfinite(Z_amp))
    if n_bad > 0:
        print(f"    ⚠ Trial {trial_idx+1}: {n_bad} non-finite values — replacing with 0")
        Z_amp = np.where(np.isfinite(Z_amp), Z_amp, np.float32(0.0))

    return eelbrain.NDVar(Z_amp, dims=(time_dim, freq_dim), name='gfnn_amplitude')

def build_gfnn_amplitude_column(events, osc_params, osc_name, integration_fs=200.0, n_jobs=N_JOBS):
    """
    Returns a list of amplitude NDVars (one per trial) without touching
    the events table — caller assigns the column directly.
    Trials are integrated in parallel across available cores.
    """
    osc_params = dict(osc_params)
    F = osc_params.pop('F', 1.0)
    
    n_trials = 30
    freqs    = np.logspace(np.log10(0.1), np.log10(8.0), 40)
    freq_dim = eelbrain.Scalar('frequency', freqs, unit='Hz')

    # Pre-extract plain arrays so joblib workers don't serialise NDVars
    envelope_arrays = [np.asarray(events['envelope'][i]) * F for i in range(n_trials)]
    time_dims       = [events['envelope'][i].time         for i in range(n_trials)]

    print(f"  [{osc_name}] integrating {n_trials} trials F={F}"
          f"integration_fs={integration_fs}Hz (n_jobs={n_jobs}) …")

    ndvars = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_integrate_trial)(
            i, envelope_arrays[i], time_dims[i], freqs, freq_dim, osc_params, integration_fs
        )
        for i in range(n_trials)
    )

    print(f"  [{osc_name}] ✓  GFNN done")
    return ndvars

def compute_envelope_rms(wav_dir, n_songs=10, fs=100, target_rms=0.7):
    """
    Compute RMS energy of the acoustic envelope for each stimulus and
    derive a global scaling factor to bring mean RMS to target_rms.

    Parameters
    ----------
    wav_dir    : Path
    n_songs    : int
    fs         : int    Target sampling rate (default 100 Hz).
    target_rms : float  Desired mean RMS after scaling (default 0.7).

    Returns
    -------
    rms_per_song : dict  {song_id -> rms}
    rms_mean     : float
    rms_std      : float
    scale        : float  Multiply every envelope by this before GFNN.
    """
    rms_per_song = {}

    print(f"Envelope RMS energy per stimulus  (target RMS = {target_rms})")
    print("-" * 52)

    for song_id in range(1, n_songs + 1):
        wav      = eelbrain.load.wav(Path(wav_dir) / f'{song_id}.wav')
        envelope = eelbrain.resample(wav.envelope(), fs)
        x        = np.asarray(envelope)

        rms = np.sqrt(np.mean(x ** 2))
        rms_per_song[song_id] = rms
        print(f"  Song {song_id:2d}:  RMS = {rms:.4f}  "
              f"min = {x.min():.4f}  max = {x.max():.4f}")

    rms_values = np.array(list(rms_per_song.values()))
    rms_mean   = rms_values.mean()
    rms_std    = rms_values.std()

    # Scale so that mean RMS across songs lands at target_rms
    scale = target_rms / rms_mean

    print("-" * 52)
    print(f"  Mean RMS : {rms_mean:.4f}")
    print(f"  Std  RMS : {rms_std:.4f}")
    print(f"  Scale    : {scale:.8f}  (× envelope → mean RMS ≈ {target_rms})")

    # Verify
    scaled_rms = rms_values * scale
    print(f"\n  RMS after scaling:")
    for song_id, srms in zip(rms_per_song.keys(), scaled_rms):
        print(f"    Song {song_id:2d}: {srms:.4f}")
    print(f"  Mean: {scaled_rms.mean():.4f}  Std: {scaled_rms.std():.4f}")

    return rms_per_song, rms_mean, rms_std, scale

# ─── Main loop ────────────────────────────────────────────────────────────────
VARIABLES = ['envelope', 'onsets', 'gfnn_amplitude']

SAVE_DIR  = f'pickles_{date.today()}_{VARIABLES}'
os.makedirs(SAVE_DIR, exist_ok=True)

TRF_START = -0.150
TRF_END   =  0.750

rms_per_song, rms_mean, rms_std, envelope_scale = compute_envelope_rms(
    WAV_DIR, target_rms=0.7
)

for SUBJECT in SUBJECTS:
    print(f"\n{'='*60}\nSubject: {SUBJECT}\n{'='*60}")

    # ── Load EEG (once per subject) ───────────────────────────────────────────
    eeg_data = load_subject_raw_eeg(EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)
    raw      = create_mne_raw_from_loaded(eeg_data)
    raw.filter(LOW_FREQUENCY, HIGH_FREQUENCY, n_jobs=1)
    events   = create_eelbrain_events(raw)

    # ── Build stimulus features (once per subject) ────────────────────────────
    # FIX 4: cache the 10 unique wavs — don't reload the same file 3× per subject
    wav_cache = {}
    envelopes = []
    for stimulus_id in events['event']:
        song_id = stimulus_id % 10 or 10
        if song_id not in wav_cache:
            wav              = eelbrain.load.wav(WAV_DIR / f'{song_id}.wav')
            envelope = eelbrain.resample(wav.envelope(), 100)
            # Apply global scale so mean RMS ≈ 0.7 across songs
            wav_cache[song_id] = envelope * envelope_scale
        envelopes.append(wav_cache[song_id])

    events['envelope'] = envelopes
    events['onsets']   = [env.diff('time').clip(0) for env in envelopes]
    events['duration'] = eelbrain.Var([env.time.tstop for env in envelopes])
    events['eeg']      = eelbrain.load.mne.variable_length_epochs(
                             events, 0, tstop='duration', decim=1, adjacency='auto')

    # ── Inner loop: one TRF per oscillator type ───────────────────────────────
    for osc_name, osc_params in OSCILLATOR_CONFIGS.items():

        filename = os.path.join(
            SAVE_DIR,
            f'{SUBJECT}_{osc_name}_encoding_window_({TRF_START},{TRF_END}).pkl'
        )
        if os.path.exists(filename):
            print(f"  [{osc_name}] already exists — skipping.")
            continue

        # FIX 1: assign only the gfnn_amplitude column — no deepcopy of events
        # FIX 2: trials integrated in parallel across cores
        events['gfnn_amplitude'] = build_gfnn_amplitude_column(
            events, osc_params, osc_name
        )

        print(f"  [{osc_name}] fitting TRF …")
        trf_cv = eelbrain.boosting(
            'eeg', VARIABLES,
            TRF_START, TRF_END,
            data=events,
            basis=0.050,
            partitions=4,
            test=True,
            error='l1',
        )

        eelbrain.save.pickle(
            {'trf_cv': trf_cv, 'osc_params': osc_params, 'subject': SUBJECT},
            filename,
        )
        print(f"  [{osc_name}] ✓  saved → {filename}")