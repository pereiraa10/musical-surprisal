import constants as constants

from collections import defaultdict
import eelbrain
from math import gcd
import mne
from mne import find_events
from mne.channels import make_dig_montage
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pretty_midi
from scipy.io import loadmat
from scipy.signal import resample_poly, butter, sosfiltfilt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# ================================
# Load EEG data
# ================================
def load_subject_raw_eeg(filepath, subject):
    """
    Load raw EEG at its original sampling frequency — NO resampling.

    The previous implementation resampled to 100 Hz inside this function,
    then resampled again to 64 Hz in TRF_ridge_3.py (double resampling).
    The new implementation returns data at the original fs so that
    preprocess_eeg_trials() can handle the full LPF → downsample → HPF
    chain in one pass, matching the MATLAB CNSP workflow.

    Padding is preserved here (not removed) so that the LPF and HPF can
    filter through the padding before it is discarded — exactly as MATLAB
    does (padding is removed last, after all filtering).
    """
    subject_idx = int(subject[3:])
    mat_data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg = mat_data["eeg"]
    orig_fs = int(eeg.fs)

    for i in range(len(eeg.data)):
        # Scale int32 counts to float32 (~microvolt range)
        eeg.data[i] = 100 * eeg.data[i].astype(np.float32) / np.iinfo(np.int32).max

    raw_data = {
        'trials':       eeg.data,
        'fs':           orig_fs,                     # original fs; no resampling done
        'chanlocs':     eeg.chanlocs,
        'pad_start':    int(eeg.paddingStartSample),  # in original-fs samples
        'subject_type': 'Musician' if subject_idx >= 11 else 'Non-musician'
    }

    print(f"✓ Loaded {raw_data['subject_type']} (Subject {subject})")
    print(f"  - {len(raw_data['trials'])} trials, "
          f"{raw_data['trials'][0].shape[1]} channels @ {orig_fs} Hz")

    return raw_data


def preprocess_eeg_trials(subject_data, target_fs=64, lpf_hz=8, hpf_hz=1, debug=False):
    """
    Preprocess EEG trials independently, faithfully replicating the MATLAB
    CNSP workflow from CNSP2025_EEGpreprocessing.m:

      Step 1 — LPF  each trial at original fs   (cellfun per trial in MATLAB)
      Step 2 — Downsample to target_fs           (cndDownsample in MATLAB)
      Step 3 — HPF  each trial at target_fs      (cellfun per trial in MATLAB)
      Step 4 — Remove leading padding            (paddingStartSample removal)

    Why this order matters
    ----------------------
    * Filtering before downsampling prevents aliasing artefacts from LPF
      energy that would fold back into the band of interest.
    * Filtering PER TRIAL (not on the concatenated signal) prevents filter
      transients / ringing at one trial from bleeding into the next.
    * Padding removal AFTER all preprocessing lets the filters settle through
      the padding rather than cold-starting at the real signal boundary.

    MATLAB detail: cndDownsample uses plain decimation (downsample()) because
    the explicit LPF beforehand already removes everything above the new
    Nyquist.  We use resample_poly (with its internal Kaiser AA filter) which
    is strictly better — the pre-applied LPF makes the two approaches
    equivalent in practice.

    MATLAB detail: cndDownsample rescales paddingStartSample:
        paddingStartSample = round(paddingStartSample / (orig_fs / downFs))
    We replicate this with int(round(pad_start_orig * target_fs / orig_fs)).

    Parameters
    ----------
    subject_data : dict   from load_subject_raw_eeg
    target_fs    : int    target sampling rate in Hz (default 64)
    lpf_hz       : float  low-pass cut-off in Hz     (default 8)
    hpf_hz       : float  high-pass cut-off in Hz    (default 1)
    debug        : bool   print per-trial length info when True

    Returns
    -------
    list of np.ndarray, shape (n_time_i, n_channels) at target_fs,
    with leading padding already removed
    """
    trials         = subject_data['trials']
    orig_fs        = subject_data['fs']
    pad_start_orig = subject_data['pad_start']

    # Integer up/down factors for resample_poly — gcd avoids any floating-point
    # drift that would accumulate over 30 trials with a naive ratio approach.
    g           = gcd(orig_fs, target_fs)
    up          = target_fs // g
    down_factor = orig_fs   // g

    # Replicate MATLAB cndDownsample: scale padding to the target domain.
    pad_start_target = int(round(pad_start_orig * target_fs / orig_fs))

    # LPF at original fs — Butterworth SOS, zero-phase sosfiltfilt.
    # SOS form avoids the numerical instability of ba coefficients at low
    # normalised cutoffs (e.g. 8 Hz at 500 Hz orig_fs → normalised = 0.032).
    nyq_orig = orig_fs / 2.0
    lpf_sos  = butter(4, lpf_hz / nyq_orig, btype='low',  output='sos')

    # HPF at target_fs — Butterworth SOS, zero-phase sosfiltfilt.
    nyq_tgt  = target_fs / 2.0
    hpf_sos  = butter(4, hpf_hz / nyq_tgt,  btype='high', output='sos')

    preprocessed = []
    for i, trial in enumerate(trials):
        trial_f64 = trial.astype(np.float64)   # (n_time, n_channels)

        if debug:
            print(f"  [pre] Trial {i+1}: raw = {trial_f64.shape[0]} samples "
                  f"@ {orig_fs} Hz  (pad = {pad_start_orig} samples)")

        # Step 1 — LPF per trial, no cross-trial bleed
        trial_lpf = sosfiltfilt(lpf_sos, trial_f64, axis=0)

        # Step 2 — downsample to target_fs using resample_poly
        trial_down = resample_poly(trial_lpf, up, down_factor, axis=0)

        # Step 3 — HPF per trial at target_fs, no cross-trial bleed
        trial_hpf = sosfiltfilt(hpf_sos, trial_down, axis=0)

        # Step 4 — remove leading padding (now expressed in target_fs samples)
        trial_clean = trial_hpf[pad_start_target:, :]

        if debug:
            n_out = trial_clean.shape[0]
            print(f"  [pre] Trial {i+1}: post-preproc = {n_out} samples "
                  f"@ {target_fs} Hz")

        preprocessed.append(trial_clean)

    return preprocessed


def create_mne_raw_from_preprocessed(preprocessed_trials, target_fs, chanlocs):
    """
    Build an MNE RawArray from already-preprocessed trials.

    Trials must already be filtered, downsampled, and padding-removed
    (i.e. output of preprocess_eeg_trials).  This function only concatenates
    them end-to-end and inserts trial-onset stim-channel markers.

    Unlike the legacy create_mne_raw_from_loaded, NO padding removal or
    filtering happens here — all preprocessing was done upstream per trial.
    """
    ch_names  = []
    positions = []
    for ch in chanlocs:
        ch_names.append(ch.labels)
        if hasattr(ch, 'X') and hasattr(ch, 'Y') and hasattr(ch, 'Z'):
            positions.append([ch.Y, ch.X, ch.Z])

    all_trials    = []
    trial_lengths = []
    for trial in preprocessed_trials:
        t = trial.T                      # → (n_channels, n_time)
        all_trials.append(t)
        trial_lengths.append(t.shape[1])

    eeg_continuous        = np.hstack(all_trials)   # (n_channels, total_time)
    n_channels, n_samples = eeg_continuous.shape

    # Stim channel: mark trial onsets with IDs 1..N
    stim_data      = np.zeros((1, n_samples))
    current_sample = 0
    for i, length in enumerate(trial_lengths):
        # Keep first-trial marker off sample 0 (MNE treats sample-0 events
        # as ambiguous on some pipelines).
        marker_sample = max(1, current_sample)
        stim_data[0, marker_sample] = i + 1
        current_sample += length

    data_with_stim = np.vstack([eeg_continuous, stim_data])
    ch_names_full  = ch_names + ['STI']
    ch_types       = ['eeg'] * n_channels + ['stim']

    info = mne.create_info(ch_names=ch_names_full,
                           sfreq=float(target_fs),
                           ch_types=ch_types)
    raw = mne.io.RawArray(data_with_stim, info)

    montage = make_dig_montage(
        ch_pos=dict(zip(ch_names[:n_channels], positions)),
        coord_frame='head'
    )
    raw.set_montage(montage)
    return raw


def create_mne_raw_from_loaded(subject_data):
    """
    Legacy function — kept for backward compatibility only.

    Original approach: remove padding, concatenate raw trials, then pass the
    concatenated signal through external filtering.  Superseded by the
    combination of preprocess_eeg_trials() + create_mne_raw_from_preprocessed()
    which matches the MATLAB CNSP per-trial preprocessing order and removes
    the double-resampling bug.
    """
    trials    = subject_data['trials']
    sfreq     = subject_data['fs']
    pad_start = subject_data['pad_start']
    chanlocs  = subject_data['chanlocs']

    ch_names  = []
    positions = []
    for ch in chanlocs:
        ch_names.append(ch.labels)
        if hasattr(ch, 'X') and hasattr(ch, 'Y') and hasattr(ch, 'Z'):
            positions.append([ch.Y, ch.X, ch.Z])

    all_trials    = []
    trial_lengths = []
    for trial in trials:
        trial_clean = trial[pad_start:, :].T
        all_trials.append(trial_clean)
        trial_lengths.append(trial_clean.shape[1])

    eeg_continuous        = np.hstack(all_trials)
    n_channels, n_samples = eeg_continuous.shape

    stim_data      = np.zeros((1, n_samples))
    current_sample = 0
    marker_positions = []
    for i in range(30):
        marker_sample = 1 if current_sample == 0 else current_sample
        stim_data[0, marker_sample] = i + 1
        marker_positions.append((i+1, marker_sample))
        current_sample += trial_lengths[i]

    data_with_stim = np.vstack([eeg_continuous, stim_data])
    ch_names       = ch_names + ['STI']
    ch_types       = ['eeg'] * n_channels + ['stim']

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw  = mne.io.RawArray(data_with_stim, info)

    montage = make_dig_montage(
        ch_pos=dict(zip(ch_names[:n_channels], positions)),
        coord_frame='head'
    )
    raw.set_montage(montage)
    return raw


def create_eelbrain_events(raw):
    """Create eelbrain events with correct column structure."""

    mne_events = mne.find_events(raw, stim_channel='STI', verbose=False)

    events_data = {
        'i_start': mne_events[:, 0],
        'trigger':  mne_events[:, 2],
        'event':    mne_events[:, 2]
    }

    events = eelbrain.Dataset(events_data)
    events.info['raw'] = raw
    return events
