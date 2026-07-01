"""
Gradient Frequency Neural Network (GFNN) Oscillator
====================================================
Implements the nonlinear oscillator:

    (1/f_i) ż_i = z_i (α + i2πr + β₁|z_i|² + εβ₂|z_i|⁴ / (1 − ε|z_i|²)) + x

Gradient Frequency Neural Network (GFNN) Oscillator
====================================================
Implements the nonlinear oscillator:

    (1/f_i) ż_i = z_i (α + i2πr + β₁|z_i|² + εβ₂|z_i|⁴ / (1 − ε|z_i|²)) + x

Integrated via RK4 over a bank of oscillators at user-defined frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
from pathlib import Path 
from scipy.io import loadmat
from scipy.signal import resample

import mne
from mne import find_events
from mne.channels import make_dig_montage
import eelbrain
import pickle
from datetime import date


BASE_DIR = Path('NRT').resolve().parent  # TRF/

# Define the dataset root; replace it with the proper path if you downloaded the dataset in a different location
DATA_ROOT = BASE_DIR / '../TRF/liberi_dataset/doi_10_5061_dryad_g1jwstqmh__v20211008'

# Define paths that will be used throughout
WAV_DIR = DATA_ROOT / 'diliBach_wav_4dryad'

STIMULUS_DIR = DATA_ROOT / 'diliBach_wav_4dryad'
EEG_DIR = DATA_ROOT / 'diliBach_4dryad_CND'
print(EEG_DIR)

LOW_FREQUENCY = 1
HIGH_FREQUENCY = 8

SUBJECTS = [
    'Sub1',
    'Sub2',
    'Sub3',
    'Sub4',
    'Sub5',
    'Sub6',
    'Sub7',
    'Sub8',
    'Sub9',
    'Sub10',
    'Sub11',
    'Sub12',
    'Sub13',
    'Sub14',
    'Sub15',
    'Sub16', 
    'Sub17', 
    'Sub18', 
    'Sub19', 
    'Sub20'
    ]

# SUBJECT = 'Sub19'

# Load EEG data from one subject
def load_subject_raw_eeg(filepath, subject):
    
    # Extract subject index from string (e.g., 'S18' -> 18)
    subject_idx = int(subject[3:])
        
    # Load the .mat file
    mat_data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg = mat_data["eeg"]
    target_fs = 100  # Target sampling frequency
    orig_fs = int(eeg.fs)
    resample_needed = orig_fs != target_fs
    for i in range(len(eeg.data)):
        
        # Scale data 
        trial_data = 100 * eeg.data[i].astype(np.float32) / np.iinfo(np.int32).max
        
        # Resample to 100Hz, if needed 
        if resample_needed:
            n_samples = int(trial_data.shape[0] * target_fs / orig_fs)
            trial_data = resample(trial_data, n_samples, axis=0)
        
        eeg.data[i] = trial_data
        
    # Extract key information into a dictionary
    raw_data = {
        'trials': eeg.data,
        'fs': target_fs,
        'chanlocs': eeg.chanlocs,
        'pad_start': int(eeg.paddingStartSample * target_fs / orig_fs) if resample_needed else int(eeg.paddingStartSample),
        'subject_type': 'Musician' if subject_idx >= 11 else 'Non-musician'
    }
    
    print(f"✓ Loaded {raw_data['subject_type']} (Subject {subject})")
    print(f"  - {len(raw_data['trials'])} trials, {raw_data['trials'][0].shape[1]} channels")
    
    return raw_data


def create_mne_raw_from_loaded(subject_data):
    """Convert already-loaded Bach data to MNE Raw object with channel positions."""
    
    trials = subject_data['trials']
    sfreq = subject_data['fs']
    pad_start = subject_data['pad_start']    
    chanlocs = subject_data['chanlocs']
    
    # Get channel names and positions
    ch_names = []
    positions = []
    
    for ch in chanlocs:
        
        # Get channel label        
        ch_names.append(ch.labels)
        
        # Get channel positions if available
        if hasattr(ch, 'X') and hasattr(ch, 'Y') and hasattr(ch, 'Z'):
            positions.append([ch.Y, ch.X, ch.Z])
    
    # Concatenate all trials
    all_trials = []
    trial_lengths = []
    
    for trial in trials:
        # Remove padding and transpose to channels x time
        trial_clean = trial[pad_start:, :].T
        all_trials.append(trial_clean)
        trial_lengths.append(trial_clean.shape[1])
    
    # Concatenate
    eeg_continuous = np.hstack(all_trials)
    n_channels, n_samples = eeg_continuous.shape
    
    # Create stimulus channel with trial markers
    stim_data = np.zeros((1, n_samples))
    
    # Mark all 30 trial onsets
    current_sample = 0
    marker_positions = []
    for i in range(30):
        # Place marker at current position (offset by 1 if at sample 0)
        marker_sample = 1 if current_sample == 0 else current_sample
        stim_data[0, marker_sample] = i + 1  # Use 1-30 as event IDs
        marker_positions.append((i+1, marker_sample))
        current_sample += trial_lengths[i]  # Move to next trial start
        
    # Combine EEG and stim
    data_with_stim = np.vstack([eeg_continuous, stim_data])
    
    # Channel setup
    ch_names = ch_names + ['STI']
    ch_types = ['eeg'] * n_channels + ['stim']
    
    # Create Raw
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_with_stim, info)
        
    montage = make_dig_montage(
        ch_pos=dict(zip(ch_names[:n_channels], positions)),
        coord_frame='head'
    )
    raw.set_montage(montage)
    return raw


def create_eelbrain_events(raw):
    """Create eelbrain events with correct column structure."""
    
    # Find events in the MNE raw object
    mne_events = mne.find_events(raw, stim_channel='STI', verbose=False)
    
    # Create eelbrain Dataset with the required columns
    events_data = {
        'i_start': mne_events[:, 0],  # Sample indices
        'trigger': mne_events[:, 2],  # Event IDs (1-30)
        'event': mne_events[:, 2]     # Same as trigger (1-30)
    }
    
    events = eelbrain.Dataset(events_data)

    # Link raw data to events for use in variable_length_epochs
    events.info['raw'] = raw
    return events

# ──────────────────────────────────────────────────────────────────────────────
# Core ODE and RK4
# ──────────────────────────────────────────────────────────────────────────────

def _rhs(z, x_t, freqs, alpha, r, beta1, beta2, epsilon):
    """
    Vectorised right-hand side of the GFNN ODE.

    z      : complex ndarray (n_freqs,)
    x_t    : float  — forcing value at current time step
    freqs  : float ndarray (n_freqs,)  — natural frequencies [Hz]
    """
    z2    = np.abs(z) ** 2
    z4    = z2 ** 2
    denom = 1.0 - epsilon * z2
    # Protect against singularity (|z|² → 1/ε)
    denom = np.where(np.abs(denom) < 1e-12,
                     np.sign(denom.real + 1e-30) * 1e-12, denom)

    nl = alpha + 1j * 2.0 * np.pi * r + beta1 * z2 + (epsilon * beta2 * z4) / denom
    return freqs * (z * nl + x_t)


def _rk4_step(z, x_n, x_half, x_n1, freqs, alpha, r, beta1, beta2, epsilon, dt):
    """Single RK4 step. x_half is the forcing interpolated at t + dt/2."""
    k1 = _rhs(z,              x_n,    freqs, alpha, r, beta1, beta2, epsilon)
    k2 = _rhs(z + 0.5*dt*k1, x_half, freqs, alpha, r, beta1, beta2, epsilon)
    k3 = _rhs(z + 0.5*dt*k2, x_half, freqs, alpha, r, beta1, beta2, epsilon)
    k4 = _rhs(z +     dt*k3, x_n1,   freqs, alpha, r, beta1, beta2, epsilon)
    return z + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def integrate_gfnn(
    x,
    freqs,
    fs=100.0,
    alpha=-0.1,
    r=1.0,
    beta1=-1.0,
    beta2=-1.0,
    epsilon=1.0,
    z0=None,
):
    """
    Integrate the GFNN ODE for a bank of oscillators driven by signal x.

    Parameters
    ----------
    x       : array-like (N,)
        Forcing signal (acoustic envelope) sampled at `fs` Hz.
    freqs   : array-like (n_freqs,)
        Natural frequencies of each oscillator in Hz (e.g. 0.1 – 30).
    fs      : float
        Sampling rate of x in Hz.  Default 100 Hz.
    alpha   : float
        Linear growth/damping term.
          α > 0 → limit-cycle oscillator (supercritical Hopf)
          α < 0 → damped, only responds while driven
    r       : float
        Sets the intrinsic rotation rate; usually 1.0.
    beta1   : float
        Cubic nonlinear coefficient (typically < 0 → saturation).
    beta2   : float
        Quintic nonlinear coefficient.
    epsilon : float
        Scaling of the higher-order rational term.
    z0      : array-like (n_freqs,) complex, optional
        Initial oscillator states.  Defaults to tiny random perturbations.

    Returns
    -------
    Z : ndarray (N, n_freqs), complex
        Oscillator responses z_i(t).
    t : ndarray (N,)
        Time axis in seconds.
    """
    x     = np.asarray(x,     dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    N       = len(x)
    n_freqs = len(freqs)
    dt      = 1.0 / fs

    if z0 is None:
        rng = np.random.default_rng(0)
        z0  = (rng.standard_normal(n_freqs) * 1e-6
               + 1j * rng.standard_normal(n_freqs) * 1e-6)

    Z    = np.zeros((N, n_freqs), dtype=complex)
    Z[0] = np.asarray(z0, dtype=complex)
    z    = Z[0].copy()

    for n in range(N - 1):
        x_n    = x[n]
        x_n1   = x[n + 1]
        x_half = 0.5 * (x_n + x_n1)   # midpoint interpolation

        z = _rk4_step(z, x_n, x_half, x_n1,
                      freqs, alpha, r, beta1, beta2, epsilon, dt)
        Z[n + 1] = z

    t = np.arange(N) * dt
    return Z, t

def run_gfnn_per_trial(events, output_dir=None, save_figs=False):
    """
    Run GFNN on each trial independently and add results to events table
    as eelbrain NDVars with (time, freq) dimensions.
    Four NDVar columns: amplitude, phase, real, imaginary.
    """
    fs = 100.0
    n_trials = 30
    
    print(f"Processing {n_trials} trials independently …")
    
    # Oscillator bank: 0.1 – 8 Hz, 40 channels, log-spaced
    freqs = np.logspace(np.log10(0.1), np.log10(8.0), 40)
    freq_dim = eelbrain.Scalar('frequency', freqs, unit='Hz')
    
    gfnn_params = dict(
        fs      = fs,
        alpha   =  0.1,
        r       =  1.0,
        beta1   = -1.0,
        beta2   = -1.0,
        epsilon =  1.0,
    )
    
    gfnn_amp_ndvars  = []
    
    for trial_idx in range(n_trials):
        envelope = np.asarray(events['envelope'][trial_idx])
        onsets   = np.asarray(events['onsets'][trial_idx])
        
        # Run GFNN — Z shape is (n_times, n_freqs)
        Z, t = integrate_gfnn(envelope, freqs, **gfnn_params)
        
        # Reuse the same time axis as the envelope for this trial
        time_dim = events['envelope'][trial_idx].time
        
        # Sanity check
        if Z.shape[0] != time_dim.nsamples:
            raise ValueError(
                f"Trial {trial_idx+1}: Z n_times={Z.shape[0]} "
                f"!= envelope n_times={time_dim.nsamples}. "
                f"Check GFNN output length."
            )
        
        # Decompose complex Z into 4 representations
        Z_amp   = np.abs(Z)       # instantaneous amplitude, shape (n_times, n_freqs)
        
        def make_ndvar(data, name):
            return eelbrain.NDVar(data, dims=(time_dim, freq_dim), name=name)
        
        gfnn_amp_ndvars.append(make_ndvar(Z_amp,   'gfnn_amplitude'))
        
        print(f"  Trial {trial_idx+1:2d}/{n_trials}  |  "
              f"Z shape: {Z.shape}  |  "
              f"amp range: [{Z_amp.min():.3f}, {Z_amp.max():.3f}]  |  "
    
    events['gfnn_amplitude']  = gfnn_amp_ndvars
    
    print(f"\n✓ Added GFNN column(s) to events table"
          f"(each: n_times × {len(freqs)} freqs per trial)")
    
    return events, freqs


for SUBJECT in SUBJECTS:
    
    # Main execution
    eeg_data = load_subject_raw_eeg(EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)
    raw = create_mne_raw_from_loaded(eeg_data)

    # Filter the raw data to the desired band
    raw.filter(LOW_FREQUENCY, HIGH_FREQUENCY, n_jobs=1)

    # Create events with correct structure
    events = create_eelbrain_events(raw)
    
    # Extract envelopes and add to the event table to use for time aligning all features
    envelopes = []

    for stimulus_id in events['event']:
        song_id = stimulus_id % 10
        song_id = song_id if song_id != 0 else 10
        wav = eelbrain.load.wav(WAV_DIR / f'{song_id}.wav')
        envelope = wav.envelope()
        envelope = eelbrain.resample(envelope, 100)
        envelopes.append(envelope)

    events['envelope'] = envelopes

    # Add a second predictor to events table corresponding to acoustic onsets
    events['onsets'] = [envelope.diff('time').clip(0) for envelope in envelopes]

    # Add duration to the events table corresponding to envelope events
    events['duration'] = eelbrain.Var([env.time.tstop for env in events['envelope']])

    # Add the eeg data itself as NDVars
    events['eeg'] = eelbrain.load.mne.variable_length_epochs(events, 0, tstop='duration', decim=1, adjacency='auto')

    events, freqs = run_gfnn_per_trial(events, save_figs=False)
    
    variables = [
    'envelope',
    'onsets',
    'gfnn_amplitude'
    ]
    
    trf_window_start = -0.150
    trf_window_end = 0.750
    
    # Estimate the predicted power
    trf_cv = eelbrain.boosting('eeg', variables, trf_window_start, trf_window_end, data=events, basis=0.050, partitions=4, test=True, error='l1')

    all_data = {
        'trf_cv': trf_cv
    }   
    
    SAVE_DIR = f'pickles_{date.today()}'

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    filename = os.path.join(SAVE_DIR, f'{SUBJECT}_{variables}_encoding_window_({trf_window_start},{trf_window_end}).pkl')
            
    eelbrain.save.pickle(all_data, filename)
