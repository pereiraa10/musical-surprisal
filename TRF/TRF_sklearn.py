"""
TRF_sklearn.py — Explicit Toeplitz ridge TRF pipeline (standalone)
===================================================================

This script implements the custom explicit Toeplitz ridge TRF pipeline ONLY.
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

This script: alpha is selected via select_alpha_loocv(), which performs
leave-one-trial-out CV on the explicit Toeplitz design matrices (Phi_all).

Counterpart: TRF_mne.py selects alpha using MNE ReceptiveField internally.
"""

import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func

from types import SimpleNamespace
import eelbrain
import numpy as np
import os
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
# Here alpha selection uses the explicit Toeplitz design matrix exclusively —
# values are NOT transferable to/from the MNE implementation in TRF_mne.py.
RIDGE_ALPHAS = np.logspace(1, 7, 25)

IC_CLIP = 15.0

# Set True to print per-trial lengths, alignment diffs, and xcorr lag diagnostics
DEBUG = True


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_lag_matrix(x, tmin, tmax, sfreq):
    """
    Vectorized Toeplitz lag matrix using stride tricks — no Python loop over lags.

    For tmin=-0.05, tmax=0.55, sfreq=100 this returns a (n_times, 61) array.
    Derivation: X[t, j] = x_padded[t + lag_max - j], so each row is a reversed
    sliding window of x_padded, giving us all lags [lag_min … lag_max] at once.
    """
    n_lags = int(round((tmax - tmin) * sfreq)) + 1
    lag_min = int(round(tmin * sfreq))   # e.g. -5  (pre-stimulus)
    lag_max = lag_min + n_lags - 1       # e.g.  55 (post-stimulus)
    n = len(x)

    # Pad: lag_max zeros before (handles positive lags), |lag_min| zeros after
    x_pad = np.concatenate([np.zeros(lag_max), x, np.zeros(max(0, -lag_min))])
    wins  = np.lib.stride_tricks.sliding_window_view(x_pad, n_lags)
    # Reverse column order so column 0 → lag_min, column n_lags-1 → lag_max
    return np.ascontiguousarray(wins[:n, ::-1])


def build_design_matrix(features, tmin, tmax, sfreq):
    """Concatenate lag matrices for all features. features: dict {name: 1-D array}"""
    return np.hstack([build_lag_matrix(v, tmin, tmax, sfreq) for v in features.values()])


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


def select_alpha_loocv(Phi_all, Y_all, alphas):
    """
    Pick regularisation strength via trial-based LOOCV on the explicit Toeplitz
    design matrices.

    For each alpha candidate, leaves out one trial at a time using the same
    XTX rank-1 update trick as loocv_ridge (O(T_i·p²) per fold instead of
    restacking n-1 trials).  Selects the alpha with the highest mean held-out
    Pearson r (averaged across channels and folds), matching the MATLAB criterion.

    IMPORTANT: alpha is tuned on the explicit Toeplitz Phi matrices and is NOT
    directly comparable to an alpha selected by MNE ReceptiveField — the two
    implementations construct lag matrices differently.
    """
    Phi_full = np.concatenate(Phi_all)
    Y_full   = np.concatenate(Y_all)
    p        = Phi_full.shape[1]
    XTX      = Phi_full.T @ Phi_full
    XTY      = Phi_full.T @ Y_full

    best_alpha, best_r = alphas[0], -np.inf
    for alpha in alphas:
        alpha_I  = alpha * np.eye(p)
        r_folds  = []
        for Phi_i, Y_i in zip(Phi_all, Y_all):
            XTX_tr = XTX - Phi_i.T @ Phi_i
            XTY_tr = XTY - Phi_i.T @ Y_i
            W      = np.linalg.solve(XTX_tr + alpha_I, XTY_tr)
            r_fold = np.mean([
                pearsonr(Y_i[:, ch], (Phi_i @ W)[:, ch])[0]
                for ch in range(Y_i.shape[1])
            ])
            r_folds.append(r_fold)
        avg_r = np.mean(r_folds)
        if avg_r > best_r:
            best_r     = avg_r
            best_alpha = alpha

    return best_alpha, best_r


def loocv_ridge(Phi_all, Y_all, alpha):
    """
    Leave-one-trial-out CV with a fixed alpha using XTX rank-1 updates.

    Key idea: precompute XTX and XTY for the full dataset once (O(T·p²)),
    then subtract each held-out trial's contribution (O(T_i·p²)) rather than
    re-stacking n-1 trials every fold.  The solve is O(p³) per fold.

    Returns (Y_pred_concatenated, Y_true_concatenated, coefs_per_fold).
    """
    Phi_full = np.concatenate(Phi_all)               # (T_total, p)
    Y_full   = np.concatenate(Y_all)                 # (T_total, n_channels)

    p       = Phi_full.shape[1]
    XTX     = Phi_full.T @ Phi_full                  # computed once
    XTY     = Phi_full.T @ Y_full
    alpha_I = alpha * np.eye(p)

    Y_pred  = np.zeros_like(Y_full)
    coefs   = []
    offset  = 0

    for Phi_i, Y_i in zip(Phi_all, Y_all):
        n_i = len(Phi_i)

        # Rank-k update: O(n_i · p²) — much cheaper than restacking (n-1) trials
        XTX_train = XTX - Phi_i.T @ Phi_i
        XTY_train = XTY - Phi_i.T @ Y_i

        W = np.linalg.solve(XTX_train + alpha_I, XTY_train)   # O(p³)
        Y_pred[offset : offset + n_i] = Phi_i @ W
        coefs.append(W.T)                                       # (n_channels, p)
        offset += n_i

    return Y_pred, Y_full, coefs


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
        # ── Step 1: build ALL design matrices once, before any fold loop ──────
        # Previously this was repeated 90× (10 folds × 9 training trials each).
        Phi_all = [
            build_design_matrix({k: zscore(t[k]) for k in feature_keys}, TMIN, TMAX, SFREQ)
            for t in trials
        ]
        Y_all = [zscore(t['eeg']) for t in trials]

        # ── Step 2: select alpha via trial-based LOOCV on Toeplitz matrices ──
        # Replicates MATLAB's LOOCV-over-trials alpha selection.
        # Uses XTX rank-1 updates so each fold is O(T_i·p²) not O(T·p²).
        # Alpha is tuned on the explicit Toeplitz Phi matrices and is NOT
        # comparable to an alpha selected by MNE ReceptiveField.
        best_alpha, best_r = select_alpha_loocv(Phi_all, Y_all, RIDGE_ALPHAS)
        print(f"  {SUBJECT} | {condition} [sklearn]: "
              f"selected alpha = {best_alpha:.2e}  (mean CV r = {best_r:.4f})")

        # ── Step 3: sklearn Ridge LOOCV with XTX rank-update ─────────────────
        # Subtracts each held-out trial from precomputed XTX/XTY (O(T_i·p²))
        # instead of restacking n-1 trials per fold.
        ridge_Y_pred, ridge_Y_true, coefs = loocv_ridge(Phi_all, Y_all, best_alpha)

        if condition == 'acoustic':
            x_label = "['envelope', 'onsets']"
            suffix  = 'acoustic_data'
        else:
            x_label = "['envelope', 'onsets', 'pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy']"
            suffix  = 'acoustic_and_surprisal_data'

        trf_ridge = make_trf_result(ridge_Y_true, ridge_Y_pred, sensor_dim, coefs=coefs)

        eelbrain.save.pickle(
            {'trf_cv': trf_ridge},
            constants.SAVE_DIR / f'{SUBJECT}_{x_label}_sklearn_ridge_{suffix}.pkl')

        print(f"  {SUBJECT} | {condition}:")
        print(f"    sklearn Ridge: mean r = {trf_ridge.r.mean():.4f}")
