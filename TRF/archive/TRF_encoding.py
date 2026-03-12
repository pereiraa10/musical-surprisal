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
idyom_mat = loadmat(constants.SURPRISAL_FILE, squeeze_me=True)

# Extracting surprisal and entropy values for each song from the IDyOM MatLab output files
surprisal_data = {}
entropy_data = {}

os.makedirs(constants.SAVE_DIR, exist_ok=True)

for song_id in unique_song_ids:

    song_name = f"audio{song_id}"

    if song_name not in idyom_mat:
        raise KeyError(f"{song_name} not found in SURPRISAL_FILE")

    surprisal_data[song_id] = np.asarray(idyom_mat[song_name])[0]

    entropy_data[song_id] = np.asarray(idyom_mat[song_name])[1]


all_events = []
stim_surprisal_ndvars = {}
stim_entropy_ndvars = {}
stim_surprisal_entropy_ndvars = {}

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
        # Create a feature dimension
        feature_dim = eelbrain.Scalar('feature', [0, 1])  # 0=surprisal, 1=entropy

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

    # # Flipping the order to match the envelope TODO: figure out why this sucks 
    events['surprisal'] = events['surprisal'][-1:] + events['surprisal'][:-1]
    events['entropy'] = events['entropy'][-1:] + events['entropy'][:-1]
    
    x = [
        'envelope',
        'onsets',
        'surprisal',
        'entropy'
    ]

    # Estimate the TRF: boosting(y, x, tstart, tstop[, scale_data, ...])
    trf = eelbrain.boosting('eeg', x, -0.150, 0.750, data=events, basis=0.050, partitions=4, error='l1')
   
    # Estimate the predicted power
    trf_cv = eelbrain.boosting('eeg', x, -0.150, 0.750, data=events, basis=0.050, partitions=4, test=True, error='l1')
   
    # Train envelope decoder
    decoder = eelbrain.boosting('envelope', 'eeg', -0.600, 0.200, data=events, partitions=4, basis=0.05, error='l1')
    
    # Train onsets decoder
    decoder_onsets = eelbrain.boosting('onsets', 'eeg', -0.600, 0.200, data=events, partitions=4, basis=0.05, error='l1')
    
    # ── Encoding: extract TRF weights per predictor ───────────────────────────────
    # trf.h is an NDVar with dims (predictor × time) or (predictor × sensor × time)
    # We store each predictor's kernel separately for easy analysis later
    trf_weights = {}
    for i, predictor in enumerate(x):
        trf_weights[predictor] = trf.h[i]  # NDVar: (sensor × time) or (time,)

    # ── Encoding: extract cross-validated prediction quality ─────────────────────
    # trf_cv.r         → correlation between predicted and actual EEG (per channel)
    # trf_cv.r_rank    → rank correlation (more robust)
    # trf_cv.y_pred    → the actual predicted EEG timeseries (if available)
    encoding_stats = {
        'r':               trf_cv.r,           # NDVar: correlation per EEG channel
        'r_rank':          trf_cv.r_rank,      # NDVar: rank correlation per channel
        'proportion_explained': trf_cv.proportion_explained  if hasattr(trf_cv, 'proportion_explained') else None,
    }
    
    # ── Encoding: per-trial encoding correlation ──────────────────────────────────
    # Convolve the TRF forward (stimulus → predicted EEG) and correlate with actual EEG
    encoding_trials = {}
    for trial_num in range(30):

        trial_encoding = {}
        for predictor in x:
            # Get the stimulus feature for this trial
            stim = events[trial_num, predictor]

            # Convolve stimulus feature with the TRF kernel to get predicted EEG
            eeg_pred = eelbrain.convolve(trf.h[x.index(predictor)], stim, name=f'predicted_eeg_{predictor}')

            # Get actual EEG for this trial
            eeg_actual = events[trial_num, 'eeg']

            # Correlation between predicted and actual EEG
            r_encoding = eelbrain.correlation_coefficient(eeg_actual, eeg_pred)

            trial_encoding[predictor] = {
                'eeg_pred':  eeg_pred,   # Predicted EEG timeseries from this feature
                'eeg_actual': eeg_actual, # Actual EEG
                'r':          r_encoding  # Correlation: scalar or per-channel NDVar
            }

        encoding_trials[f'trial{trial_num}'] = trial_encoding

    # ── Decoding: per-trial ──────────────────────────
    decoding_trials = {}
    for trial_num in range(30):
        
        # --- Envelope ---
        # Normalize the EEG      
        eeg_one_event = events[trial_num, 'eeg'] / decoder.x_scale
        # Predict the envelope by convolving the decoder with the EEG
        y_pred = eelbrain.convolve(decoder.h, eeg_one_event, name='predicted envelope')

        # Normalize trial envelope and compute correlation with prediction
        y = events[trial_num, 'envelope']
        y = y - decoder.y_mean
        y /= decoder.y_scale / y_pred.std()
        y.name = 'envelope'
        r = eelbrain.correlation_coefficient(y, y_pred)
        
        # --- Onsets ---
        # Normalize the EEG      
        eeg_one_event_onsets = events[trial_num, 'eeg'] / decoder_onsets.x_scale
        # Predict the onsets by convolving the decoder with the EEG
        y_pred_onsets = eelbrain.convolve(decoder_onsets.h, eeg_one_event_onsets, name='predicted onsets')

        # Normalize trial onsets and compute correlation with prediction
        y_onsets = events[trial_num, 'onsets']
        y_onsets = y_onsets - decoder_onsets.y_mean
        y_onsets /= decoder_onsets.y_scale / y_pred_onsets.std()
        y_onsets.name = 'onsets'
        r_onsets = eelbrain.correlation_coefficient(y_onsets, y_pred_onsets)
        
        # Store decoding trial data
        decoding_trials[f'trial{trial_num}'] = {
            'y_pred': y_pred,
            'y': y,
            'r': r,
            'y_pred_onsets': y_pred_onsets,
            'y_onsets': y_onsets,
            'r_onsets': r_onsets
        }
        # print(f"Processed trial {trial_num}: envelope r={r}, onsets r={r_onsets}")
        
    # Create the data structure for one subject and one band
    all_data = {
        'trf': trf,
        'trf_cv': trf_cv,
        'decoder': decoder,
        'decoder_onsets': decoder_onsets,
        # Encoding-specific extracted data
        'encoding': {
            'predictors':       x,
            'trf_weights':      trf_weights,      # Per-predictor TRF kernels
            'stats':            encoding_stats,   # CV r, r_rank, proportion explained
            'trials':           encoding_trials,  # Per-trial predicted vs actual EEG
        },

        # Decoding-specific extracted data (renamed from 'trials' for clarity)
        'decoding': {
            'trials': decoding_trials
        }
    }
    
    # ================================================================
    # Save TRF data as pickle file 
    # ================================================================      
    
    if not os.path.exists(constants.SAVE_DIR):
        os.makedirs(constants.SAVE_DIR)
    
    filename = os.path.join(constants.SAVE_DIR, f'{SUBJECT}_{x}_encoding_decoding_data.pkl')
            
    eelbrain.save.pickle(all_data, filename)
    
    # with open(filename, 'wb') as f:
    #     pickle.dump(trf, f)

    # ================================================================
    # Plot the TRF
    # ================================================================

    # Computing the standard deviation based on the max time
    # t = trf.h[0].std('sensor').argmax('time')
    # p = eelbrain.plot.TopoButterfly(trf.h, t=t, w=10, h=4, clip='circle')
    # p.save(f'{SUBJECT}_topo_butterfly_plot_{x}.png')
    # p.close()

    # Alternative visualization as array image
    # p = eelbrain.plot.TopoArray(trf.h, t=[0.150, 0.350, 0.500], w=6, h=4, clip='circle')
    # p.save(f'{SUBJECT}_topo_array_plot_{x}.png')
    # p.close()

    
