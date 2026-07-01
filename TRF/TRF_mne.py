"""
TRF_mne.py — MNE ReceptiveField TRF pipeline (standalone)
===========================================================

This script implements the MNE ReceptiveField TRF pipeline ONLY.
All preprocessing, alignment, resampling, feature generation, and trial
construction are identical to TRF_ridge_3.py.

WHY THIS SCRIPT EXISTS
----------------------
The original TRF_ridge_3.py selected best_alpha using the explicit Toeplitz
lag matrix (Phi_all / select_alpha_loocv) and then reused that same alpha
inside MNE ReceptiveField.  This is statistically inconsistent:

  - MNE builds its own internal lag matrix differently from the explicit
    Toeplitz construction.
  - MNE may scale or center features differently during the fit.
  - Therefore an alpha tuned on one implementation's design matrix is not
    directly comparable to an alpha tuned on the other's.

The fix: each implementation now selects its own alpha using ONLY its own
lag-matrix construction.  This produces a fair, reproducible comparison.

This script: alpha is selected via select_alpha_loocv_mne(), which performs
leave-one-trial-out CV entirely inside MNE ReceptiveField.

Counterpart: TRF_sklearn.py selects alpha using the explicit Toeplitz matrices.
"""

import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func

from types import SimpleNamespace
import eelbrain
import numpy as np
import os
from mne.decoding import ReceptiveField
from sklearn.linear_model import Ridge
from scipy.io import loadmat
import mne
from math import gcd
from scipy.signal import resample_poly as sp_resample_poly
from scipy.stats import pearsonr


# ─── TRF config ───────────────────────────────────────────────────────────────
TMIN = -0.1   # seconds
TMAX = 0.600
SFREQ = 64    # Hz after resampling

# Alpha candidates; one is selected per condition via trial-based LOOCV.
# Here alpha selection uses MNE ReceptiveField exclusively — values are NOT
# transferable to/from the explicit Toeplitz implementation in TRF_sklearn.py.
RIDGE_ALPHAS = np.logspace(1, 7, 25)

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


def select_alpha_loocv_mne(X_all, Y_all, feature_keys, alphas):
    """
    Pick regularisation strength for MNE ReceptiveField via trial-based LOOCV.

    For each alpha candidate, leaves out one trial at a time, trains
    ReceptiveField (with Ridge(alpha, fit_intercept=False)) on the remaining
    trials, predicts the held-out trial, and computes mean Pearson r across
    EEG channels.  Selects the alpha with the highest mean held-out r.

    IMPORTANT: this function uses MNE's internal lag-matrix construction
    exclusively.  Alpha values selected here are NOT directly comparable to
    those selected by select_alpha_loocv() in TRF_sklearn.py, because the two
    implementations construct lag matrices differently.
    """
    best_alpha, best_r = alphas[0], -np.inf
    n_trials = len(X_all)

    for alpha in alphas:
        r_folds = []
        for i in range(n_trials):
            X_train = np.concatenate([X_all[j] for j in range(n_trials) if j != i])
            Y_train = np.concatenate([Y_all[j] for j in range(n_trials) if j != i])

            # Pass Ridge explicitly so alpha lives in the Ridge regularisation
            # space, not MNE's internal re-parameterisation.
            rf = ReceptiveField(
                TMIN, TMAX, SFREQ,
                feature_names=feature_keys,
                estimator=Ridge(alpha=alpha, fit_intercept=False),
            )
            rf.fit(X_train, Y_train)
            Y_pred_i = rf.predict(X_all[i])
            Y_true_i = Y_all[i]

            r_fold = np.mean([
                pearsonr(Y_true_i[:, ch], Y_pred_i[:, ch])[0]
                for ch in range(Y_true_i.shape[1])
            ])
            r_folds.append(r_fold)

        avg_r = np.mean(r_folds)
        if avg_r > best_r:
            best_r     = avg_r
            best_alpha = alpha

    return best_alpha, best_r


def loocv_mne(X_all, Y_all, feature_keys, alpha):
    """
    Leave-one-trial-out CV using MNE ReceptiveField (Ridge, alpha fixed).

    X_all: list of (n_times, n_features) raw feature arrays (no lag expansion).
    MNE builds its own Toeplitz lag matrix internally.

    Uses Ridge(alpha=alpha, fit_intercept=False) — alpha has been selected by
    select_alpha_loocv_mne(), which used MNE's own lag-matrix construction.
    """
    Y_pred_all = []
    Y_true_all = []

    for i in range(len(X_all)):
        X_train = np.concatenate([X_all[j] for j in range(len(X_all)) if j != i])
        Y_train = np.concatenate([Y_all[j] for j in range(len(Y_all)) if j != i])

        rf = ReceptiveField(
            TMIN, TMAX, SFREQ,
            feature_names=feature_keys,
            estimator=Ridge(alpha=alpha, fit_intercept=False),
        )
        rf.fit(X_train, Y_train)
        Y_pred_all.append(rf.predict(X_all[i]))
        Y_true_all.append(Y_all[i])

    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all)


def make_trf_result(Y_true, Y_pred, sensor_dim, coefs=None):
    """
    SimpleNamespace that mimics the eelbrain BoostingResult interface.
    r is computed on all concatenated held-out samples (same as boosting test=True).
    """
    n_ch   = Y_true.shape[1]
    r_vals = np.array([pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(n_ch)])
    result = SimpleNamespace(r=eelbrain.NDVar(r_vals, dims=(sensor_dim,), name='r'))
    if coefs is not None:
        result.coefs = coefs
    return result


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
        # Raw feature arrays (no lag expansion) — MNE ReceptiveField builds its
        # own Toeplitz lag matrix internally.
        X_all = [np.column_stack([zscore(t[k]) for k in feature_keys]) for t in trials]
        Y_all = [zscore(t['eeg']) for t in trials]

        # ── Alpha selection: MNE LOOCV only ──────────────────────────────────
        # Alpha is selected using MNE ReceptiveField's internal lag matrix.
        # This is independent of the explicit Toeplitz implementation in
        # TRF_sklearn.py.  The two alpha values will generally differ and
        # SHOULD differ — they are optimal for different design matrices.
        best_alpha, best_r = select_alpha_loocv_mne(
            X_all, Y_all, feature_keys, RIDGE_ALPHAS
        )
        print(f"  {SUBJECT} | {condition} [MNE]: "
              f"selected alpha = {best_alpha:.2e}  (mean CV r = {best_r:.4f})")

        # ── MNE ReceptiveField LOOCV (alpha selected above) ──────────────────
        mne_Y_pred, mne_Y_true = loocv_mne(X_all, Y_all, feature_keys, best_alpha)

        if condition == 'acoustic':
            x_label = "['envelope', 'onsets']"
            suffix  = 'acoustic_data'
        else:
            x_label = "['envelope', 'onsets', 'pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy']"
            suffix  = 'acoustic_and_surprisal_data'

        trf_mne = make_trf_result(mne_Y_true, mne_Y_pred, sensor_dim)

        eelbrain.save.pickle(
            {'trf_cv': trf_mne},
            constants.SAVE_DIR / f'{SUBJECT}_{x_label}_mne_ridge_{suffix}.pkl')

        print(f"  {SUBJECT} | {condition}:")
        print(f"    MNE Ridge:  mean r = {trf_mne.r.mean():.4f}")
