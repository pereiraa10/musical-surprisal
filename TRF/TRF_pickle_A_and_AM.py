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
import pickle
import pretty_midi
from scipy.io import loadmat
from scipy.signal import resample, butter, filtfilt, decimate, hilbert
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# Main execution - run this file from inside the TRF Folder

# Load the EEG data and provided envelope from dataStim.mat
stim_mat = loadmat(constants.EEG_DIR / "dataStim.mat", struct_as_record=False, squeeze_me=True)
stim = stim_mat["stim"]
stim_fs = int(stim.fs)

# Get list of song_IDs in the order they were played as stimulus
unique_song_ids = np.unique(stim.stimIdxs)

# Load the surprisal data the IDyOM output files
idyom_pitch_mat = loadmat(constants.PITCH_SURPRISAL_FILE, squeeze_me=True)
idyom_onset_mat = loadmat(constants.ONSET_SURPRISAL_FILE, squeeze_me=True)

# Extracting surprisal and entropy values for each song from the IDyOM MatLab output files
pitch_surprisal_data = {}
pitch_entropy_data = {}
onset_surprisal_data = {}
onset_entropy_data = {}

os.makedirs(constants.SAVE_DIR, exist_ok=True)

IC_CLIP = 15.0 # bits

for song_id in unique_song_ids:

    song_name = f"audio{song_id}"

    if song_name not in idyom_pitch_mat:
        raise KeyError(f"{song_name} not found in PITCH_SURPRISAL_FILE")
    if song_name not in idyom_onset_mat:
        raise KeyError(f"{song_name} not found in ONSET_SURPRISAL_FILE")

    raw_pitch = np.asarray(idyom_pitch_mat[song_name])
    raw_onset = np.asarray(idyom_onset_mat[song_name])

    # Clip IC values (row 0), leave entropy (row 1) unclipped
    pitch_surprisal_data[song_id] = np.clip(raw_pitch[0], 0, IC_CLIP)
    pitch_entropy_data[song_id]   = raw_pitch[1]
    onset_surprisal_data[song_id] = np.clip(raw_onset[0], 0, IC_CLIP)
    onset_entropy_data[song_id]   = raw_onset[1]
    
all_events = []
stim_pitch_surprisal_ndvars = {}
stim_pitch_entropy_ndvars = {}
stim_onset_surprisal_ndvars = {}
stim_onset_entropy_ndvars = {}

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
    events['duration'] = eelbrain.Var([env.time.tstop for env in events['envelope']])

    # CAUTION: the decim factor has been changed from 5 to 1 to see if this will fix the shape error. This seems to change the sampling rate
    # Add the eeg data itself as NDVars
    events['eeg'] = eelbrain.load.mne.variable_length_epochs(events, 0, tstop='duration', decim=1, adjacency='auto')

    # Setting the sampling rate
    sfreq = raw.info['sfreq']
    dt = 1 / sfreq

    # Make the Surprisal NDVar and Entropy NDVar
    for stimulus_id in events['event']:

        # make the song_IDs
        song_id = stimulus_id % 10
        song_id = song_id if song_id != 0 else 10
        
        # Skip if already built for this song_id
        if song_id in stim_pitch_surprisal_ndvars:
            continue
    
        midi_path = constants.MIDI_DIR / f"audio{song_id}.mid"
        
        # find the number of envelope samples for each song
        time = events['envelope'][stimulus_id - 1].time
        n_times = time.nsamples

        pitch_surprisal_ndvar = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, pitch_surprisal_data[song_id], 100, n_times),
            dims=(time,), name="pitch_surprisal")

        pitch_entropy_ndvar = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, pitch_entropy_data[song_id], 100, n_times),
            dims=(time,), name="pitch_entropy")

        onset_surprisal_ndvar = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, onset_surprisal_data[song_id], 100, n_times),
            dims=(time,), name="onset_surprisal")

        onset_entropy_ndvar = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, onset_entropy_data[song_id], 100, n_times),
            dims=(time,), name="onset_entropy")

        stim_pitch_surprisal_ndvars[song_id] = pitch_surprisal_ndvar
        stim_pitch_entropy_ndvars[song_id]   = pitch_entropy_ndvar
        stim_onset_surprisal_ndvars[song_id] = onset_surprisal_ndvar
        stim_onset_entropy_ndvars[song_id]   = onset_entropy_ndvar

    pitch_surprisal_per_event = []
    pitch_entropy_per_event = []
    onset_surprisal_per_event = []
    onset_entropy_per_event = []

    for stimulus_id in events['event']:
        song_id = stimulus_id % 10
        song_id = song_id if song_id != 0 else 10  
        
        pitch_surprisal_per_event.append(stim_pitch_surprisal_ndvars[int(song_id)])
        pitch_entropy_per_event.append(stim_pitch_entropy_ndvars[int(song_id)])
        onset_surprisal_per_event.append(stim_onset_surprisal_ndvars[int(song_id)])
        onset_entropy_per_event.append(stim_onset_entropy_ndvars[int(song_id)])

    events['pitch_surprisal'] = pitch_surprisal_per_event
    events['pitch_entropy'] = pitch_entropy_per_event
    events['onset_surprisal'] = onset_surprisal_per_event
    events['onset_entropy'] = onset_entropy_per_event
    
    x_a = [
        'envelope',
        'onsets'
    ]

    # Estimate the predicted power
    trf_cv = eelbrain.boosting('eeg', x_a, -0.05, 0.550, data=events, basis=0.020, partitions=10, test=True, error='l1')
   
    all_data = {
        'trf_cv': trf_cv,
    }
       
    # ================================================================
    # Save TRF data as pickle file 
    # ================================================================      
    
    if not os.path.exists(constants.SAVE_DIR):
        os.makedirs(constants.SAVE_DIR)
    
    filename = os.path.join(constants.SAVE_DIR, f'{SUBJECT}_{x_a}_acoustic_data.pkl')
            
    eelbrain.save.pickle(all_data, filename)
    
    x_am = [
        'envelope',
        'onsets',
        'pitch_surprisal',
        'pitch_entropy',
        'onset_surprisal',
        'onset_entropy'
    ]

    # Estimate the predicted power
    trf_cv = eelbrain.boosting('eeg', x_am, -0.05, 0.550, data=events, basis=0.020, partitions=10, test=True, error='l1')
    
    all_data = {
        'trf_cv': trf_cv,
    }
       
    # ================================================================
    # Save TRF data as pickle file 
    # ================================================================      
    
    if not os.path.exists(constants.SAVE_DIR):
        os.makedirs(constants.SAVE_DIR)
    
    filename = os.path.join(constants.SAVE_DIR, f'{SUBJECT}_{x_am}_acoustic_and_surprisal_data.pkl')
            
    eelbrain.save.pickle(all_data, filename)