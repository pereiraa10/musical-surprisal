"""
diagnostic_d2_shuffle.py  —  Diagnostic D2: shuffle / null test on real data
────────────────────────────────────────────────────────────────────────────────
Addresses hypothesis H3 (data leak) from TRF_conv_DIAGNOSTICS.md by breaking
the stimulus–EEG correspondence for the HELD-OUT trial and checking whether
r collapses to ~0.  If r stays high after shuffling the held-out pairing,
there is a leak (temporal autocorrelation artifact, inner-val split exposure,
or something else).  If r collapses, the high real-data r is genuine signal.

Also prints the LOOCV-selected ridge alpha (H2 evidence: a very large alpha
means ridge is over-regularised relative to conv's weight_decay=1e-3).

Protocol:
  1. Load Sub2 with the exact same preprocessing as TRF_conv_1.py /
     TRF_ridge_3.py: eeg_functions.preprocess_eeg_trials → LOOCV.
  2. Run the ridge pipeline → print selected alpha.
  3. Run the linear conv LOOCV → r_conv_real.
  4. Run the linear conv LOOCV with circularly-shifted held-out EEG
     (shift = T//2 so stimulus and EEG are maximally misaligned) → r_conv_shuffled.
  5. Run the linear conv LOOCV with cross-trial pairing shuffled on the
     held-out trial (each held-out EEG is paired with a DIFFERENT trial's
     stimulus, preserving distributions but destroying stimulus–response
     correspondence) → r_conv_xshuffle.

Interpretation guide (printed at end):
  r_conv_shuffled ≈ 0   → no circular-autocorrelation artifact.
  r_conv_xshuffle ≈ 0   → no stimulus-distribution-level leak.
  Either >> 0            → examine D3 (regularisation) anyway; genuine signal
                           may still exist on top of the inflation.

Run from musical-surprisal/TRF/:
    python diagnostic_d2_shuffle.py
Requires:  numpy, scipy, torch, mne, eelbrain, pretty_midi + EEG .mat files.
"""

import os
import sys
import warnings
from math import gcd

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly as sp_resample_poly
from scipy.stats import pearsonr

import torch
import torch.nn as nn

import eelbrain
import constants as constants
import eeg_functions as eeg_func
import midi_func as midi_func


# ── TRF constants (must match TRF_conv_1.py and TRF_ridge_3.py) ──────────────
SFREQ    = 64
TMIN     = -0.1
TMAX     = 0.600
IC_CLIP  = 15.0
N_LAGS   = int(round((TMAX - TMIN) * SFREQ)) + 1   # 46
LAG_MIN  = int(round(TMIN * SFREQ))                 # -6
LAG_MAX  = LAG_MIN + N_LAGS - 1                     # 39

RIDGE_ALPHAS        = np.logspace(1, 7, 25)
WEIGHT_DECAY        = 1e-3
EPOCHS              = 50
LR                  = 1e-3
EARLY_STOP_PATIENCE = 25

SUBJECT = 'Sub2'   # change to test a different subject


# ── Helpers (verbatim from TRF_ridge_3.py) ───────────────────────────────────
def build_lag_matrix(x, tmin, tmax, sfreq):
    n_lags  = int(round((tmax - tmin) * sfreq)) + 1
    lag_min = int(round(tmin * sfreq))
    lag_max = lag_min + n_lags - 1
    n       = len(x)
    x_pad   = np.concatenate([np.zeros(lag_max), x, np.zeros(max(0, -lag_min))])
    wins    = np.lib.stride_tricks.sliding_window_view(x_pad, n_lags)
    return np.ascontiguousarray(wins[:n, ::-1])


def build_design_matrix(features_dict, tmin, tmax, sfreq):
    return np.hstack([build_lag_matrix(v, tmin, tmax, sfreq)
                      for v in features_dict.values()])


def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)


def select_alpha_loocv(Phi_all, Y_all, alphas):
    Phi_full = np.concatenate(Phi_all)
    Y_full   = np.concatenate(Y_all)
    p        = Phi_full.shape[1]
    XTX      = Phi_full.T @ Phi_full
    XTY      = Phi_full.T @ Y_full
    best_alpha, best_r = alphas[0], -np.inf
    for alpha in alphas:
        alpha_I = alpha * np.eye(p)
        r_folds = []
        for Phi_i, Y_i in zip(Phi_all, Y_all):
            XTX_tr = XTX - Phi_i.T @ Phi_i
            XTY_tr = XTY - Phi_i.T @ Y_i
            W      = np.linalg.solve(XTX_tr + alpha_I, XTY_tr)
            r_fold = np.mean([pearsonr(Y_i[:, c], (Phi_i @ W)[:, c])[0]
                              for c in range(Y_i.shape[1])])
            r_folds.append(r_fold)
        avg_r = np.mean(r_folds)
        if avg_r > best_r:
            best_r, best_alpha = avg_r, alpha
    return best_alpha


def loocv_ridge(Phi_all, Y_all, alpha):
    Phi_full = np.concatenate(Phi_all)
    Y_full   = np.concatenate(Y_all)
    p        = Phi_full.shape[1]
    XTX      = Phi_full.T @ Phi_full
    XTY      = Phi_full.T @ Y_full
    alpha_I  = alpha * np.eye(p)
    Y_pred   = np.zeros_like(Y_full)
    offset   = 0
    for Phi_i, Y_i in zip(Phi_all, Y_all):
        n_i  = len(Phi_i)
        W    = np.linalg.solve(XTX - Phi_i.T @ Phi_i + alpha_I,
                               XTY - Phi_i.T @ Y_i)
        Y_pred[offset: offset + n_i] = Phi_i @ W
        offset += n_i
    return Y_pred, Y_full


def mean_r(Y_true, Y_pred):
    n = Y_true.shape[1]
    return np.mean([pearsonr(Y_true[:, c], Y_pred[:, c])[0] for c in range(n)])


# ── Conv (verbatim from TRF_conv_1.py) ───────────────────────────────────────
class CausalPad(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left, self.right = left, right

    def forward(self, x):
        return nn.functional.pad(x, (self.left, self.right))


def make_linear_conv(n_features, n_channels):
    return nn.Sequential(
        CausalPad(LAG_MAX, max(0, -LAG_MIN)),
        nn.Conv1d(n_features, n_channels, kernel_size=N_LAGS, bias=True),
    )


def _to_tensor(arr_2d):
    return torch.from_numpy(
        np.ascontiguousarray(arr_2d.T[None].astype(np.float32)))


def train_conv(X_tr, Y_tr, n_features, n_channels):
    model   = make_linear_conv(n_features, n_channels)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    inner_val_idx = len(X_tr) - 1
    tr_idx        = list(range(inner_val_idx))
    Xv = _to_tensor(X_tr[inner_val_idx])
    Yv = _to_tensor(Y_tr[inner_val_idx])
    best_val, best_state, since = np.inf, None, 0
    for _ in range(EPOCHS):
        model.train()
        for i in tr_idx:
            opt.zero_grad()
            nn.functional.mse_loss(model(_to_tensor(X_tr[i])),
                                   _to_tensor(Y_tr[i])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xv), Yv).item()
        if v < best_val - 1e-6:
            best_val, best_state, since = v, \
                {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= EARLY_STOP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def loocv_conv_normal(X_all, Y_all):
    """Standard LOOCV — identical to TRF_conv_1.loocv_conv."""
    n_features = X_all[0].shape[1]
    n_channels = Y_all[0].shape[1]
    Y_pred_all, Y_true_all = [], []
    for i in range(len(X_all)):
        X_tr = [X_all[j] for j in range(len(X_all)) if j != i]
        Y_tr = [Y_all[j] for j in range(len(Y_all)) if j != i]
        model = train_conv(X_tr, Y_tr, n_features, n_channels)
        model.eval()
        with torch.no_grad():
            pred = model(_to_tensor(X_all[i])).numpy()[0].T
        Y_pred_all.append(pred)
        Y_true_all.append(Y_all[i])
    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all)


def loocv_conv_circular_shift(X_all, Y_all):
    """
    D2a: Circularly shift the held-out EEG by T//2 before evaluating r.
    Training is IDENTICAL to normal; only the r computation uses shifted EEG.
    This tests whether the model's r is driven by temporal autocorrelation
    rather than genuine stimulus-EEG correlation.
    """
    n_features = X_all[0].shape[1]
    n_channels = Y_all[0].shape[1]
    Y_pred_all, Y_true_all = [], []
    for i in range(len(X_all)):
        X_tr = [X_all[j] for j in range(len(X_all)) if j != i]
        Y_tr = [Y_all[j] for j in range(len(Y_all)) if j != i]
        model = train_conv(X_tr, Y_tr, n_features, n_channels)
        model.eval()
        with torch.no_grad():
            pred = model(_to_tensor(X_all[i])).numpy()[0].T
        T_i = Y_all[i].shape[0]
        Y_shifted = np.roll(Y_all[i], T_i // 2, axis=0)  # large circular shift
        Y_pred_all.append(pred)
        Y_true_all.append(Y_shifted)
    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all)


def loocv_conv_xshuffle(X_all, Y_all, rng):
    """
    D2b: Cross-trial pairing shuffle — pair held-out stimulus from fold i
    with EEG from a DIFFERENT fold j (j ≠ i, drawn without replacement).
    Training is normal; only the held-out evaluation pair changes.
    This breaks stimulus-response correspondence entirely while preserving
    signal distributions, so any residual r indicates distribution-level leak.
    """
    n_features = X_all[0].shape[1]
    n_channels = Y_all[0].shape[1]
    n_trials   = len(X_all)
    # Build a permutation that pairs each test fold with a DIFFERENT trial's EEG
    perm = rng.permutation(n_trials)
    for k in range(n_trials):
        if perm[k] == k:   # avoid identical pairing
            swap = (k + 1) % n_trials
            perm[k], perm[swap] = perm[swap], perm[k]

    Y_pred_all, Y_true_all = [], []
    for i in range(n_trials):
        X_tr = [X_all[j] for j in range(n_trials) if j != i]
        Y_tr = [Y_all[j] for j in range(n_trials) if j != i]
        model = train_conv(X_tr, Y_tr, n_features, n_channels)
        model.eval()
        # Predict from trial i's stimulus (as trained), evaluate against
        # a DIFFERENT trial's EEG to break stimulus-response pairing.
        with torch.no_grad():
            pred = model(_to_tensor(X_all[i])).numpy()[0].T
        # Trim to the shorter of pred and the paired EEG trial
        j   = perm[i]
        n   = min(pred.shape[0], Y_all[j].shape[0])
        Y_pred_all.append(pred[:n])
        Y_true_all.append(Y_all[j][:n])
    return np.concatenate(Y_pred_all), np.concatenate(Y_true_all)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading — identical to TRF_ridge_3.py and TRF_conv_1.py
# ═════════════════════════════════════════════════════════════════════════════

def load_subject_trials(subject):
    stim_mat    = loadmat(constants.EEG_DIR / "dataStim.mat",
                          struct_as_record=False, squeeze_me=True)
    stim        = stim_mat["stim"]
    stim_fs     = int(stim.fs)
    stimFeature = stim.data[0, :]

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
        raw_pitch = np.asarray(idyom_pitch_mat[song_name])
        raw_onset = np.asarray(idyom_onset_mat[song_name])
        pitch_surprisal_data[song_id] = np.clip(raw_pitch[0], 0, IC_CLIP)
        pitch_entropy_data[song_id]   = raw_pitch[1]
        onset_surprisal_data[song_id] = np.clip(raw_onset[0], 0, IC_CLIP)
        onset_entropy_data[song_id]   = raw_onset[1]

    eeg_data = eeg_func.load_subject_raw_eeg(
        constants.EEG_DIR / f'data{subject}.mat', subject)
    preprocessed_trials = eeg_func.preprocess_eeg_trials(
        eeg_data, target_fs=SFREQ,
        lpf_hz=constants.HIGH_FREQUENCY,
        hpf_hz=constants.LOW_FREQUENCY,
        debug=False)
    eeg_trial_lengths = [t.shape[0] for t in preprocessed_trials]

    raw    = eeg_func.create_mne_raw_from_preprocessed(
        preprocessed_trials, SFREQ, eeg_data['chanlocs'])
    events = eeg_func.create_eelbrain_events(raw)

    envelopes = []
    for i in range(len(events['event'])):
        env_raw       = np.asarray(stimFeature[i], dtype=np.float64)
        n_eeg         = eeg_trial_lengths[i]
        env_resampled = sp_resample_poly(env_raw, stim_up, stim_down)
        n_min         = min(len(env_resampled), n_eeg)
        diff          = len(env_resampled) - n_eeg
        if abs(diff) > 4 * SFREQ:
            warnings.warn(f"Trial {i}: large stim/EEG mismatch (diff={diff})")
        env_resampled = env_resampled[:n_min]
        time_axis     = eelbrain.UTS(0, 1 / SFREQ, n_min)
        envelopes.append(eelbrain.NDVar(env_resampled, (time_axis,)))

    events['envelope'] = envelopes
    events['onsets']   = [env.diff('time').clip(0) for env in envelopes]
    events['duration'] = eelbrain.Var([env.time.tstop for env in envelopes])
    events['eeg']      = eelbrain.load.mne.variable_length_epochs(
        events, 0, tstop='duration', decim=1, adjacency='auto')

    sp_ndvars = {}
    se_ndvars = {}
    op_ndvars = {}
    oe_ndvars = {}
    for i, stimulus_id in enumerate(events['event']):
        song_id = int(stimulus_id % 10) or 10
        if song_id in sp_ndvars:
            continue
        midi_path = constants.MIDI_DIR / f"audio{song_id}.mid"
        time      = events['envelope'][i].time
        n_times   = time.nsamples
        sp_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(
                midi_path, pitch_surprisal_data[song_id], SFREQ, n_times),
            dims=(time,))
        se_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(
                midi_path, pitch_entropy_data[song_id], SFREQ, n_times),
            dims=(time,))
        op_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(
                midi_path, onset_surprisal_data[song_id], SFREQ, n_times),
            dims=(time,))
        oe_ndvars[song_id] = eelbrain.NDVar(
            midi_func.make_surprisal_timeseries(
                midi_path, onset_entropy_data[song_id], SFREQ, n_times),
            dims=(time,))

    events['pitch_surprisal'] = [sp_ndvars[int(s % 10) or 10] for s in events['event']]
    events['pitch_entropy']   = [se_ndvars[int(s % 10) or 10] for s in events['event']]
    events['onset_surprisal'] = [op_ndvars[int(s % 10) or 10] for s in events['event']]
    events['onset_entropy']   = [oe_ndvars[int(s % 10) or 10] for s in events['event']]

    trials = []
    for i in range(len(events['event'])):
        eeg_arr = events['eeg'][i].get_data(('sensor', 'time')).T
        n = min(eeg_arr.shape[0], events['envelope'][i].x.shape[0])
        trials.append({
            'eeg':             eeg_arr[:n],
            'envelope':        events['envelope'][i].x[:n],
            'onsets':          events['onsets'][i].x[:n],
            'pitch_surprisal': events['pitch_surprisal'][i].x[:n],
            'pitch_entropy':   events['pitch_entropy'][i].x[:n],
            'onset_surprisal': events['onset_surprisal'][i].x[:n],
            'onset_entropy':   events['onset_entropy'][i].x[:n],
        })

    return trials


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    print("=" * 70)
    print(f" Diagnostic D2 — shuffle / null test on {SUBJECT}")
    print("=" * 70)

    print(f"\n  Loading {SUBJECT} data...")
    trials = load_subject_trials(SUBJECT)
    print(f"  Loaded {len(trials)} trials.")
    for condition, feature_keys in [
        ('acoustic',               ['envelope', 'onsets']),
        ('acoustic_and_surprisal', ['envelope', 'onsets', 'pitch_surprisal',
                                    'pitch_entropy', 'onset_surprisal', 'onset_entropy']),
    ]:
        print()
        print(f"  ── Condition: {condition} ────────────────────────────────────")

        X_all = [np.column_stack([zscore(t[k]) for k in feature_keys]) for t in trials]
        Y_all = [zscore(t['eeg']) for t in trials]

        # ── Ridge: select alpha (H2 evidence) ────────────────────────────────
        Phi_all = [build_design_matrix({k: zscore(t[k]) for k in feature_keys},
                                        TMIN, TMAX, SFREQ)
                   for t in trials]
        best_alpha = select_alpha_loocv(Phi_all, Y_all, RIDGE_ALPHAS)
        Y_pred_ridge, Y_true_ridge = loocv_ridge(Phi_all, Y_all, best_alpha)
        r_ridge = mean_r(Y_true_ridge, Y_pred_ridge)
        print(f"    Ridge selected alpha: {best_alpha:.2e}  →  r = {r_ridge:.4f}")

        # ── Conv: normal LOOCV ────────────────────────────────────────────────
        print("    Conv (normal LOOCV)...")
        Y_pred_c, Y_true_c = loocv_conv_normal(X_all, Y_all)
        r_conv = mean_r(Y_true_c, Y_pred_c)
        print(f"    Conv (normal):                          r = {r_conv:.4f}  "
              f"(ratio vs ridge: {r_conv / (r_ridge + 1e-9):.2f}×)")

        # ── D2a: circular-shift null ──────────────────────────────────────────
        print("    Conv (circular-shift null)...")
        Y_pred_s, Y_true_s = loocv_conv_circular_shift(X_all, Y_all)
        r_shift = mean_r(Y_true_s, Y_pred_s)
        print(f"    Conv (held-out EEG shifted T/2):        r = {r_shift:.4f}")

        # ── D2b: cross-trial pairing shuffle ─────────────────────────────────
        print("    Conv (cross-trial pairing shuffle)...")
        Y_pred_x, Y_true_x = loocv_conv_xshuffle(X_all, Y_all, rng)
        r_xshuffle = mean_r(Y_true_x, Y_pred_x)
        print(f"    Conv (different trial's EEG held-out):  r = {r_xshuffle:.4f}")

        # ── Interpretation ────────────────────────────────────────────────────
        thr = 0.01   # if shuffle r > this, it's suspicious
        print()
        print(f"    Summary — {condition}:")
        print(f"      Ridge r         = {r_ridge:.4f}  (alpha = {best_alpha:.2e})")
        print(f"      Conv normal r   = {r_conv:.4f}")
        print(f"      Conv shift null = {r_shift:.4f}  "
              f"({'[!!] SUSPICIOUS — autocorr artifact' if r_shift > thr else '[OK] collapsed as expected'})")
        print(f"      Conv xshuffle   = {r_xshuffle:.4f}  "
              f"({'[!!] SUSPICIOUS — distribution leak' if r_xshuffle > thr else '[OK] collapsed as expected'})")

        if best_alpha >= 1e5:
            print(f"\n    [H2] Ridge selected alpha={best_alpha:.0e} is LARGE.  "
                  "Likely over-regularised; try running with alpha fixed at 1e3–1e4.")
        else:
            print(f"\n    [H2] Ridge alpha={best_alpha:.0e} is moderate — "
                  "reg mismatch alone is unlikely to explain 3× gap.")

        if r_conv > r_ridge * 1.5 and r_shift < thr and r_xshuffle < thr:
            print("    [!] Large gap AND null r≈0 → H2 (reg mismatch) is the likely")
            print("        explanation.  Run D3-real: fix ridge alpha near 1e3-1e4")
            print("        and recompare, or reduce conv weight_decay to 0.1–1.0.")
        elif r_shift > thr or r_xshuffle > thr:
            print("    [!] Null r > 0 — investigate autocorrelation-driven inflation.")
            print("        Compare ridge null (shuffle ridge held-out EEG) for baseline.")


if __name__ == "__main__":
    main()
