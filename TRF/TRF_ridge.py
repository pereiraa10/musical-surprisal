import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func

from types import SimpleNamespace
import eelbrain
import numpy as np
import os
from mne.decoding import ReceptiveField
from scipy.io import loadmat
from scipy.stats import pearsonr


# ─── TRF config ───────────────────────────────────────────────────────────────
TMIN = -0.05   # seconds — same window as boosting baseline
TMAX = 0.550
SFREQ = 100    # Hz after resampling

# Alpha candidates; one is selected per condition via GCV on the full data
RIDGE_ALPHAS = np.logspace(1, 7, 25)

IC_CLIP = 15.0


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


def select_alpha_gcv(XTX, XTY, Phi, Y, alphas):
    """
    Pick regularisation strength via GCV on the full concatenated data.

    Uses the eigendecomposition of X^T X (computed once, O(p³)) so each alpha
    candidate costs only O(p) — no per-alpha matrix solve needed.
    GCV formula: RSS(α) / (T · (1 − p_eff(α)/T)²)
    where RSS and p_eff are expressed purely in terms of eigenvalues.
    """
    T = Phi.shape[0]
    eigvals, eigvecs = np.linalg.eigh(XTX)           # O(p³), symmetric → fast
    eigvals = np.maximum(eigvals, 0)                  # numerical safety

    # Project X^T Y onto the eigenbasis; shape (p, n_channels)
    UtXTY = eigvecs.T @ XTY

    # ||UtXTY[j,:]||² — needed for RSS without forming U explicitly
    UtXTY_sq = (UtXTY ** 2).sum(axis=1)              # (p,)
    Y_sq = float((Y ** 2).sum())

    best_alpha, best_gcv = alphas[0], np.inf
    for alpha in alphas:
        d     = eigvals / (eigvals + alpha)           # smoothing per component
        p_eff = d.sum()
        # ||Y − H_α Y||² = ||Y||² − 2·(d·UtXTY²) + (d²·UtXTY²)
        rss   = Y_sq - 2.0*(d * UtXTY_sq).sum() + (d**2 * UtXTY_sq).sum()
        gcv   = rss / (T * (1.0 - p_eff / T) ** 2)
        if gcv < best_gcv:
            best_gcv  = gcv
            best_alpha = alpha

    return best_alpha


def loocv_mne(X_all, Y_all, feature_keys, alpha):
    """
    Leave-one-trial-out CV using MNE ReceptiveField (Ridge, alpha fixed).

    X_all: list of (n_times, n_features) raw feature arrays (no lag expansion).
    MNE builds its own Toeplitz lag matrix internally — kept separate so we can
    verify that our explicit Toeplitz (loocv_ridge) and MNE's agree.
    """
    Y_pred_all = []
    Y_true_all = []

    for i in range(len(X_all)):
        X_train = np.concatenate([X_all[j] for j in range(len(X_all)) if j != i])
        Y_train = np.concatenate([Y_all[j] for j in range(len(Y_all)) if j != i])

        rf = ReceptiveField(TMIN, TMAX, SFREQ, feature_names=feature_keys,
                            estimator=alpha)
        rf.fit(X_train, Y_train)
        Y_pred_all.append(rf.predict(X_all[i]))
        Y_true_all.append(Y_all[i])

    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all)


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
    raw = eeg_func.create_mne_raw_from_loaded(eeg_data)
    raw.filter(constants.LOW_FREQUENCY, constants.HIGH_FREQUENCY, n_jobs=1)
    events = eeg_func.create_eelbrain_events(raw)

    envelopes = []
    for stimulus_id in events['event']:
        song_id = int(stimulus_id % 10) or 10
        wav = eelbrain.load.wav(constants.WAV_DIR / f'{song_id}.wav')
        envelope = eelbrain.resample(wav.envelope(), SFREQ)
        envelopes.append(envelope)

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

    # Convert every trial to plain numpy arrays
    trials = []
    for i in range(len(events['event'])):
        eeg_ndvar = events['eeg'][i]
        trials.append({
            'eeg':             eeg_ndvar.get_data(('sensor', 'time')).T,
            'envelope':        events['envelope'][i].x,
            'onsets':          events['onsets'][i].x,
            'pitch_surprisal': events['pitch_surprisal'][i].x,
            'pitch_entropy':   events['pitch_entropy'][i].x,
            'onset_surprisal': events['onset_surprisal'][i].x,
            'onset_entropy':   events['onset_entropy'][i].x,
        })

    sensor_dim = events['eeg'][0].sensor

    for condition, feature_keys in [
        ('acoustic',               ['envelope', 'onsets']),
        ('acoustic_and_surprisal', ['envelope', 'onsets', 'pitch_surprisal',
                                    'pitch_entropy', 'onset_surprisal', 'onset_entropy']),
    ]:
        # ── Step 1: build ALL design matrices once, before any fold loop ──────
        # Previously this was repeated 90× (10 folds × 9 training trials each).
        Phi_all = [
            build_design_matrix({k: t[k] for k in feature_keys}, TMIN, TMAX, SFREQ)
            for t in trials
        ]
        # Raw feature arrays (no lag expansion) — used by MNE ReceptiveField
        X_all = [np.column_stack([t[k] for k in feature_keys]) for t in trials]
        Y_all = [t['eeg'] for t in trials]

        Phi_full = np.concatenate(Phi_all)
        Y_full   = np.concatenate(Y_all)

        # ── Step 2: select alpha once via GCV on the full dataset ─────────────
        # Both methods share this alpha so the comparison is fair.
        # GCV uses eigh(X^T X) once (O(p³)); each of 25 candidates then costs O(p).
        XTX = Phi_full.T @ Phi_full
        XTY = Phi_full.T @ Y_full
        best_alpha = select_alpha_gcv(XTX, XTY, Phi_full, Y_full, RIDGE_ALPHAS)
        print(f"  {SUBJECT} | {condition}: selected alpha = {best_alpha:.2e}")

        # ── Step 3a: MNE ReceptiveField LOOCV (Ridge, MNE builds lag matrix) ──
        mne_Y_pred, mne_Y_true = loocv_mne(X_all, Y_all, feature_keys, best_alpha)

        # ── Step 3b: sklearn Ridge LOOCV with XTX rank-update ─────────────────
        # Subtracts each held-out trial from precomputed XTX/XTY (O(T_i·p²))
        # instead of restacking n-1 trials per fold.
        ridge_Y_pred, ridge_Y_true, coefs = loocv_ridge(Phi_all, Y_all, best_alpha)

        if condition == 'acoustic':
            x_label = "['envelope', 'onsets']"
            suffix  = 'acoustic_data'
        else:
            x_label = "['envelope', 'onsets', 'pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy']"
            suffix  = 'acoustic_and_surprisal_data'

        trf_mne   = make_trf_result(mne_Y_true,   mne_Y_pred,   sensor_dim)
        trf_ridge = make_trf_result(ridge_Y_true, ridge_Y_pred, sensor_dim, coefs=coefs)

        eelbrain.save.pickle(
            {'trf_cv': trf_mne},
            constants.SAVE_DIR / f'{SUBJECT}_{x_label}_mne_ridge_{suffix}.pkl')

        eelbrain.save.pickle(
            {'trf_cv': trf_ridge},
            constants.SAVE_DIR / f'{SUBJECT}_{x_label}_sklearn_ridge_{suffix}.pkl')

        print(f"  {SUBJECT} | {condition}:")
        print(f"    MNE Ridge:    mean r = {trf_mne.r.mean():.4f}")
        print(f"    sklearn Ridge: mean r = {trf_ridge.r.mean():.4f}")
