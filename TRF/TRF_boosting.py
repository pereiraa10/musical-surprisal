import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func

import eelbrain
import numpy as np
import os
from scipy.io import loadmat
import mne
from math import gcd
from scipy.signal import resample_poly as sp_resample_poly

from scipy.stats import pearsonr


# ─── TRF config ───────────────────────────────────────────────────────────────
TMIN  = -0.1   # seconds
TMAX  =  0.600
SFREQ =  64    # Hz after resampling

IC_CLIP = 15.0

# Set True to print per-trial lengths, alignment diffs, and xcorr lag diagnostics
DEBUG = True


# ─── Helpers ──────────────────────────────────────────────────────────────────

def zscore(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def align_trial(eeg, stim_arrays, trial_idx, max_diff=2):
    """Trim EEG and all stimulus arrays to the same length.

    Rounding during resampling can cause ±1 sample discrepancies; anything
    larger than max_diff samples is treated as a real alignment error.
    """
    n_eeg  = eeg.shape[0]
    n_stim = len(next(iter(stim_arrays.values())))
    diff   = abs(n_eeg - n_stim)
    if diff > max_diff:
        raise ValueError(
            f"Trial {trial_idx}: EEG/stimulus length mismatch "
            f"(EEG={n_eeg}, stim={n_stim}, diff={diff} samples). "
            "Check padding removal and resampling."
        )
    if diff > 0:
        print(f"  [align] Trial {trial_idx}: trimming {diff} sample(s) "
              f"(EEG={n_eeg}, stim={n_stim})")
    n = min(n_eeg, n_stim)
    return eeg[:n], {k: v[:n] for k, v in stim_arrays.items()}


def loocv_boosting(y_all, x_all_trials, tmin, tmax):
    """
    Leave-one-trial-out CV using eelbrain.boosting.

    Mirrors the loocv_ridge structure: fits on n-1 trials, predicts on the
    held-out trial via convolution with the learned kernels, then concatenates
    held-out predictions to compute r.

    y_all         : list of (sensor, time) NDVars — z-scored EEG per trial
    x_all_trials  : list of lists — x_all_trials[trial][feat_idx] is a
                    (time,) NDVar, z-scored
    Returns (y_pred_concat, y_true_concat) as numpy (n_total, n_ch).
    """
    n_trials = len(y_all)
    y_pred_all = []
    y_true_all = []

    for i in range(n_trials):
        train_idx = [j for j in range(n_trials) if j != i]

        y_train = eelbrain.concatenate([y_all[j] for j in train_idx])
        x_train = [
            eelbrain.concatenate([x_all_trials[j][k] for j in train_idx])
            for k in range(len(x_all_trials[0]))
        ]

        # partitions=None (default): eelbrain auto-splits the training segment for
        # early stopping.  partitions=0 is invalid when validate=1 (the default).
        result = eelbrain.boosting(y_train, x_train, tmin, tmax,
                                   scale_data=False, test=0)

        # result.h is a list of NDVars (one per predictor) when x_train is a list.
        # Predict by summing convolutions: y = Σ_k h_k * x_k
        x_test = x_all_trials[i]
        y_pred_ndvar = sum(
            eelbrain.convolve(h_k, x_k)
            for h_k, x_k in zip(result.h, x_test)
        )

        y_pred_np = y_pred_ndvar.get_data(('sensor', 'time')).T   # (n_t, n_ch)
        y_true_np = y_all[i].get_data(('sensor', 'time')).T

        # Trim to the same length — convolution can add/remove ±1 edge sample
        n = min(len(y_pred_np), len(y_true_np))
        y_pred_all.append(y_pred_np[:n])
        y_true_all.append(y_true_np[:n])

        print(f"  [boosting LOOCV] Fold {i + 1}/{n_trials} done")

    return np.concatenate(y_pred_all), np.concatenate(y_true_all)


# ─── Load stimulus / IDyOM data ───────────────────────────────────────────────

stim_mat = loadmat(constants.EEG_DIR / "dataStim.mat", struct_as_record=False, squeeze_me=True)
stim     = stim_mat["stim"]
stim_fs     = int(stim.fs)
stim_factor = stim_fs // SFREQ
stimFeature = stim.data[0, :]   # per-trial envelopes at stim_fs Hz

# Integer up/down factors for resample_poly when resampling stimulus to SFREQ.
# Computed once here so the same rational ratio is used for every trial —
# avoids per-trial floating-point drift that caused temporal misalignment.
_g        = gcd(stim_fs, SFREQ)
stim_up   = SFREQ   // _g
stim_down = stim_fs // _g
unique_song_ids = np.unique(stim.stimIdxs)

idyom_pitch_mat = loadmat(constants.PITCH_SURPRISAL_FILE, squeeze_me=True)
idyom_onset_mat = loadmat(constants.ONSET_SURPRISAL_FILE, squeeze_me=True)

pitch_surprisal_data, pitch_entropy_data = {}, {}
onset_surprisal_data, onset_entropy_data = {}, {}

for song_id in unique_song_ids:
    song_name = f"audio{song_id}"
    if song_name not in idyom_pitch_mat:
        raise KeyError(f"{song_name} not found in PITCH_SURPRISAL_FILE")
    if song_name not in idyom_onset_mat:
        raise KeyError(f"{song_name} not found in ONSET_SURPRISAL_FILE")

    raw_pitch = np.asarray(idyom_pitch_mat[song_name])
    raw_onset = np.asarray(idyom_onset_mat[song_name])
    pitch_surprisal_data[song_id] = np.clip(raw_pitch[0], 0, IC_CLIP)
    pitch_entropy_data[song_id]   = raw_pitch[1]
    onset_surprisal_data[song_id] = np.clip(raw_onset[0], 0, IC_CLIP)
    onset_entropy_data[song_id]   = raw_onset[1]

os.makedirs(constants.SAVE_DIR, exist_ok=True)

stim_pitch_surprisal_ndvars = {}
stim_pitch_entropy_ndvars   = {}
stim_onset_surprisal_ndvars = {}
stim_onset_entropy_ndvars   = {}


# ─── Main loop over subjects ──────────────────────────────────────────────────

for SUBJECT in constants.SUBJECTS:

    eeg_data = eeg_func.load_subject_raw_eeg(
        constants.EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)

    # ── Per-trial preprocessing: LPF → resample_poly → HPF → remove padding ──
    # This replaces the old pipeline which:
    #   (a) resampled to 100 Hz inside load_subject_raw_eeg  (1st resample)
    #   (b) concatenated trials and filtered the whole signal  (filter bleed)
    #   (c) resampled again to 64 Hz via resample_poly         (2nd resample)
    #
    # The new pipeline processes every trial independently, matching the MATLAB
    # CNSP cellfun-per-trial approach in CNSP2025_EEGpreprocessing.m.
    # Padding is removed last (after filtering), as in MATLAB cndDownsample.
    preprocessed_trials = eeg_func.preprocess_eeg_trials(
        eeg_data,
        target_fs=SFREQ,
        lpf_hz=constants.HIGH_FREQUENCY,
        hpf_hz=constants.LOW_FREQUENCY,
        debug=DEBUG
    )
    eeg_trial_lengths = [t.shape[0] for t in preprocessed_trials]

    raw    = eeg_func.create_mne_raw_from_preprocessed(
        preprocessed_trials, SFREQ, eeg_data['chanlocs'])
    events = eeg_func.create_eelbrain_events(raw)

    # ── Stimulus resampling: align to the shorter of stim / EEG ─────────────
    # Old approach: sp_resample(env, n_target) — DFT-based (no anti-aliasing),
    # and the name sp_resample was never imported → NameError at runtime.
    # New approach: resample_poly with consistent integer up/down factors, then
    # trim both to min(stim_len, eeg_len) matching MATLAB's approach.
    envelopes = []
    for i in range(len(events['event'])):
        env_raw = np.asarray(stimFeature[i], dtype=np.float64)
        n_eeg   = eeg_trial_lengths[i]

        # Resample stimulus with resample_poly (consistent with EEG downsampling)
        env_resampled = sp_resample_poly(env_raw, stim_up, stim_down)

        # Replicate MATLAB's min(envLen, eegLen) alignment from
        # CNSP2025_forwardTRF_example1.m — trim to the shorter of stim and EEG.
        # A ~64-sample (1 s) mismatch is normal for this dataset: the EEG
        # recording overruns the audio slightly.  MATLAB silently trims it.
        # Only warn for mismatches > 4 s, which would indicate a real error.
        n_min = min(len(env_resampled), n_eeg)
        diff  = len(env_resampled) - n_eeg
        if abs(diff) > 4 * SFREQ:
            import warnings
            warnings.warn(
                f"Trial {i}: unusually large stim/EEG mismatch "
                f"(stim={len(env_resampled)}, EEG={n_eeg}, diff={diff} smp, "
                f"{abs(diff)/SFREQ:.2f} s). Check padding removal and resampling."
            )
        if DEBUG and diff != 0:
            print(f"  [align] Trial {i}: stim={len(env_resampled)}, EEG={n_eeg}, "
                  f"n_min={n_min} (diff={diff} smp, {diff*1000/SFREQ:.1f} ms)")
        env_resampled = env_resampled[:n_min]

        if DEBUG:
            # Cross-correlation peak lag between envelope and first EEG channel
            eeg_ch0 = preprocessed_trials[i][:n_min, 0]
            sig_a   = (env_resampled - env_resampled.mean()) / (env_resampled.std() + 1e-12)
            sig_b   = (eeg_ch0       - eeg_ch0.mean())       / (eeg_ch0.std()       + 1e-12)
            xcorr   = np.correlate(sig_b, sig_a, mode='full')
            lag_smp = int(np.argmax(xcorr)) - (n_min - 1)
            lag_ms  = lag_smp * 1000.0 / SFREQ
            ok      = -200 <= lag_ms <= 600
            print(f"  [xcorr] Trial {i}: peak lag = {lag_smp} smp "
                  f"({lag_ms:.1f} ms)  {'[OK]' if ok else '[WARNING: implausible]'}")

        time_axis = eelbrain.UTS(0, 1 / SFREQ, n_min)
        envelopes.append(eelbrain.NDVar(env_resampled, (time_axis,)))

    events['envelope'] = envelopes
    events['onsets']   = [env.diff('time').clip(0) for env in envelopes]
    events['duration'] = eelbrain.Var([env.time.tstop for env in envelopes])
    events['eeg']      = eelbrain.load.mne.variable_length_epochs(
        events, 0, tstop='duration', decim=1, adjacency='auto')

    for i, stimulus_id in enumerate(events['event']):
        song_id = int(stimulus_id % 10) or 10
        if song_id in stim_pitch_surprisal_ndvars:
            continue
        midi_path = constants.MIDI_DIR / f"audio{song_id}.mid"
        time      = events['envelope'][i].time
        n_times   = time.nsamples

        stim_pitch_surprisal_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, pitch_surprisal_data[song_id], SFREQ, n_times),
            dims=(time,), name="pitch_surprisal")
        stim_pitch_entropy_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, pitch_entropy_data[song_id], SFREQ, n_times),
            dims=(time,), name="pitch_entropy")
        stim_onset_surprisal_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, onset_surprisal_data[song_id], SFREQ, n_times),
            dims=(time,), name="onset_surprisal")
        stim_onset_entropy_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(midi_path, onset_entropy_data[song_id], SFREQ, n_times),
            dims=(time,), name="onset_entropy")

    events['pitch_surprisal'] = [stim_pitch_surprisal_ndvars[int(sid % 10) or 10] for sid in events['event']]
    events['pitch_entropy']   = [stim_pitch_entropy_ndvars[int(sid % 10) or 10]   for sid in events['event']]
    events['onset_surprisal'] = [stim_onset_surprisal_ndvars[int(sid % 10) or 10] for sid in events['event']]
    events['onset_entropy']   = [stim_onset_entropy_ndvars[int(sid % 10) or 10]   for sid in events['event']]

    # Convert every trial to plain numpy arrays, checking EEG/stimulus alignment
    trials = []
    for i in range(len(events['event'])):
        eeg_arr  = events['eeg'][i].get_data(('sensor', 'time')).T   # (n_times, n_ch)
        stim_arr = {
            'envelope':        events['envelope'][i].x,
            'onsets':          events['onsets'][i].x,
            'pitch_surprisal': events['pitch_surprisal'][i].x,
            'pitch_entropy':   events['pitch_entropy'][i].x,
            'onset_surprisal': events['onset_surprisal'][i].x,
            'onset_entropy':   events['onset_entropy'][i].x,
        }
        eeg_arr, stim_arr = align_trial(eeg_arr, stim_arr, trial_idx=i)
        trials.append({'eeg': eeg_arr, **stim_arr})

    sensor_dim = events['eeg'][0].sensor

    for condition, feature_keys in [
        ('acoustic',               ['envelope', 'onsets']),
        ('acoustic_and_surprisal', ['envelope', 'onsets', 'pitch_surprisal',
                                    'pitch_entropy', 'onset_surprisal', 'onset_entropy']),
    ]:
        # ── Build per-trial NDVars for boosting ──────────────────────────────
        # EEG: (sensor, time) NDVar; stimuli: one (time,) NDVar per feature.
        # z-score here (scale_data=False in boosting) to match the ridge pipeline.
        y_all        = []
        x_all_trials = []

        for t in trials:
            n_t       = t['eeg'].shape[0]
            time_axis = eelbrain.UTS(0, 1 / SFREQ, n_t)

            y_all.append(eelbrain.NDVar(
                zscore(t['eeg']).T,
                dims=(sensor_dim, time_axis), name='eeg'
            ))
            x_all_trials.append([
                eelbrain.NDVar(zscore(t[k]), dims=(time_axis,), name=k)
                for k in feature_keys
            ])

        # ── Step 1: LOOCV for prediction accuracy ────────────────────────────
        # Same leave-one-trial-out structure as loocv_ridge: fit on n-1 trials,
        # predict on held-out, concatenate to compute r.
        y_pred, y_true = loocv_boosting(y_all, x_all_trials, TMIN, TMAX)

        n_ch   = y_true.shape[1]
        r_vals = np.array([pearsonr(y_true[:, c], y_pred[:, c])[0]
                           for c in range(n_ch)])

        # ── Step 2: fit on all data to get TRF kernels ───────────────────────
        y_full = eelbrain.concatenate(y_all)
        x_full = [
            eelbrain.concatenate([x_all_trials[j][k] for j in range(len(trials))])
            for k in range(len(feature_keys))
        ]
        full_result = eelbrain.boosting(y_full, x_full, TMIN, TMAX,
                                        scale_data=False, test=0)

        if condition == 'acoustic':
            x_label = "['envelope', 'onsets']"
            suffix  = 'acoustic_data'
        else:
            x_label = "['envelope', 'onsets', 'pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy']"
            suffix  = 'acoustic_and_surprisal_data'

        save_dict = {
            'r': eelbrain.NDVar(r_vals, dims=(sensor_dim,), name='r'),
            'h': full_result.h,   # TRF kernels from full-data fit
        }
        eelbrain.save.pickle(
            save_dict,
            constants.SAVE_DIR / f'{SUBJECT}_{x_label}_boosting_{suffix}.pkl')

        print(f"  {SUBJECT} | {condition}: mean r = {r_vals.mean():.4f}")
