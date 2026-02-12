import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func

from collections import defaultdict
import eelbrain
import librosa
import mne
from mne import find_events
from mne.channels import make_dig_montage
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pretty_midi
from scipy.io import loadmat
from scipy.signal import resample, butter, filtfilt, decimate, hilbert
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# Main execution - run this file from inside the TRF Folder
all_events = []

# Load the EEG data and provided envelope from dataStim.mat
stim_mat = loadmat(constants.EEG_DIR / "dataStim.mat", struct_as_record=False, squeeze_me=True)
stim = stim_mat["stim"]
stim_fs = int(stim.fs)

# Get list of song_IDs in the order they were played as stimulus
unique_song_ids = np.unique(stim.stimIdxs)

# Load the surprisal data the IDyOM output files
idyom_mat = loadmat(constants.SURPRISAL_FILE, squeeze_me=True)

# Extracting surprisal and entropy values for each song from the IDyOM MatLab output files
surprisal_data = {}
entropy_data = {}

for song_id in unique_song_ids:

    song_name = f"audio{song_id}"

    if song_name not in idyom_mat:
        raise KeyError(f"{song_name} not found in SURPRISAL_FILE")

    surprisal_data[song_id] = np.asarray(idyom_mat[song_name])[0]

    entropy_data[song_id] = np.asarray(idyom_mat[song_name])[1]


all_events = []
stim_surprisal_ndvars = {}
stim_entropy_ndvars = {}

def load_envelope(wav_path, sfreq):
    x, sr = librosa.load(wav_path, sr=None, mono=True)
    env = np.abs(hilbert(x))
    env = librosa.resample(env, orig_sr=sr, target_sr=sfreq)
    return env

# Load in each subject's EEG Data and form events table
for SUBJECT in constants.SUBJECTS:
    eeg_data = eeg_func.load_subject_raw_eeg(
        constants.EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)
    raw = eeg_func.create_mne_raw_from_loaded(eeg_data)

    # Filter the raw data to the desired band
    raw.filter(constants.LOW_FREQUENCY, constants.HIGH_FREQUENCY, n_jobs=1)

    # Create events with correct structure
    events = eeg_func.create_eelbrain_events(raw)

    # Extract envelopes and add to the event table to use for time aligning all features
    envelopes = []

    for stimulus_id in events['event']:
        song_id = stimulus_id % 10
        song_id = song_id if song_id != 0 else 10
        wav = eelbrain.load.wav(constants.WAV_DIR / f'{song_id}.wav')
        envelope = wav.envelope()
        envelope = eelbrain.resample(envelope, 100)
        envelopes.append(envelope)

    events['envelope'] = envelopes

    # Add a second predictor to events table corresponding to acoustic onsets
    events['onsets'] = [envelope.diff('time').clip(0) for envelope in envelopes]

    # Add duration to the events table corresponding to envelope events
    events['duration'] = eelbrain.Var(
        [env.time.tstop for env in events['envelope']])

    # CAUTION: the decim factor has been changed from 5 to 1 to see if this will fix the shape error. This seems to change the sampling rate
    # Add the eeg data itself as NDVars
    events['eeg'] = eelbrain.load.mne.variable_length_epochs(events, 0, tstop='duration', decim=1, adjacency='auto')

    # Setting the sampling rate
    sfreq = raw.info['sfreq']
    dt = 1 / sfreq

    # Make the Surprisal NDVar
    for stimulus_id in events['event']:

        # make the song_IDs
        song_id = stimulus_id % 10
        song_id = song_id if song_id != 0 else 10

        # find the number of eeg samples for each song
        n_times = events['eeg'][stimulus_id - 1].time.nsamples

        # Pull the surprisal data and entropy data for the current song
        midi_path = constants.MIDI_DIR / f"audio{song_id}.mid"
        surprisal_vec = surprisal_data[song_id]
        entropy_vec = entropy_data[song_id]

        # Make the surprisal data and entropy data into a time-based vector
        surprisal_ts = midi_func.make_surprisal_timeseries(
            midi_path=midi_path,
            surprisal_vec=surprisal_vec,
            sfreq=sfreq,
            n_times=n_times)

        entropy_ts = midi_func.make_surprisal_timeseries(
            midi_path=midi_path,
            surprisal_vec=entropy_vec,
            sfreq=sfreq,
            n_times=n_times)

        time = events['eeg'][stimulus_id - 1].time  #  reuse EEG time axis directly

        surprisal_ndvar = eelbrain.NDVar(surprisal_ts, dims=(time, ), name="surprisal")

        entropy_ndvar = eelbrain.NDVar(entropy_ts, dims=(time, ), name="entropy")

        stim_surprisal_ndvars[stimulus_id - 1] = surprisal_ndvar
        stim_entropy_ndvars[stimulus_id - 1] = entropy_ndvar

    surprisal_per_event = []
    entropy_per_event = []

    for stimulus_id in events['event']:
        song_id = stim.stimIdxs[stimulus_id - 1]
        surprisal_per_event.append(stim_surprisal_ndvars[int(song_id)])
        entropy_per_event.append(stim_entropy_ndvars[int(song_id)])

    events['surprisal'] = surprisal_per_event
    events['entropy'] = entropy_per_event

    # Flipping the order to match the envelope TODO: figure out why this sucks 
    events['surprisal'] = events['surprisal'][-1:] + events['surprisal'][:-1]
    events['entropy'] = events['entropy'][-1:] + events['entropy'][:-1]
    
    x = [
        'envelope',
        'onsets',
        'surprisal',
        'entropy',
    ]

    # Estimate the TRF: boosting(y, x, tstart, tstop[, scale_data, ...])
    trf = eelbrain.boosting('eeg', x, -0.150, 0.750, data=events, basis=0.050, partitions=4, error='l1')

    # ================================================================
    # Plot the TRF
    # ================================================================

    # Computing the standard deviation based on the max time
    t = trf.h[0].std('sensor').argmax('time')
    p = eelbrain.plot.TopoButterfly(trf.h, t=t, w=10, h=4, clip='circle')
    p.save(f'{SUBJECT}_topo_butterfly_plot.png')
    p.close()

    # Alternative visualization as array image
    p = eelbrain.plot.TopoArray(trf.h, t=[0.150, 0.350, 0.500], w=6, h=4, clip='circle')
    p.save(f'{SUBJECT}_topo_array_plot.png')
    p.close()

    break
