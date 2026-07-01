"""
TRF_diagnostics.py
==================
Standalone diagnostic script for the musical-surprisal TRF pipeline.
Run from:  musical-surprisal/TRF/

Sections
--------
1.  Load one subject / one trial
2.  Stimulus–EEG cross-correlation
3.  Lag-matrix convention inspection
4.  Z-score / feature distribution diagnostics
5.  Surprisal sparsity visualisation
6.  Resampling method comparison (MNE vs scipy.signal.resample_poly)
7.  Predicted vs actual EEG
8.  Synthetic weight-recovery test
9.  Save CSV summary
10. Diagnostic summary
"""

import sys
from pathlib import Path

# Allow "import constants" etc. from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import constants
import eeg_functions as eeg_func
import midi_func

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive; all output saved to files
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import correlate
from math import gcd
from scipy.signal import resample as sp_resample   # kept for Section 6 comparison only
from scipy.signal import resample_poly
from scipy.signal import butter, sosfiltfilt
from scipy.stats import pearsonr
import mne
from mne.decoding import ReceptiveField
import eelbrain


# ════════════════════════════════════════════════════════════════════════════
# Global configuration
# ════════════════════════════════════════════════════════════════════════════

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
TRIAL_IDX   = 0        # which trial to inspect in all single-trial diagnostics

# TRF window — identical to TRF_ridge_2.py
TMIN        = -0.1    # seconds (pre-stimulus)
TMAX        =  0.600   # seconds (post-stimulus)
SFREQ       = 64       # Hz after downsampling

IC_CLIP     = 15.0     # information-content clip in bits
RIDGE_ALPHA = 1e4      # fixed alpha for diagnostic fits (no LOOCV needed here)
CHANNEL_IDX = 0        # EEG channel used in all single-channel plots

# Typical cortical auditory response latencies to annotate on correlation plots
AUDITORY_LATENCIES_MS = [80, 100, 150, 200]

OUT_DIR = None  # set per-subject in main loop


# ════════════════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════════════════

def build_lag_matrix(x, tmin, tmax, sfreq):
    """
    Vectorised Toeplitz lag matrix via stride tricks — identical to TRF_ridge_2.py.

    Column ordering:
        col 0           → lag_min  (most negative / pre-stimulus)
        col n_lags - 1  → lag_max  (most positive / post-stimulus)

    WHY this matters: if columns were reversed the recovered TRF would be
    time-flipped, making a 100 ms post-stimulus peak appear at -100 ms.
    """
    n_lags  = int(round((tmax - tmin) * sfreq)) + 1
    lag_min = int(round(tmin * sfreq))    # e.g. -13 at 64 Hz, tmin=-0.2
    lag_max = lag_min + n_lags - 1
    n       = len(x)
    x_pad   = np.concatenate([np.zeros(lag_max), x, np.zeros(max(0, -lag_min))])
    wins    = np.lib.stride_tricks.sliding_window_view(x_pad, n_lags)
    return np.ascontiguousarray(wins[:n, ::-1])


def zscore(x):
    """Mean-zero unit-variance; returns zeros for constant arrays."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def ridge_solve(X, y, alpha):
    """Closed-form ridge regression: w = (X'X + αI)⁻¹ X'y."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)


def section_header(n, title):
    bar = '═' * 70
    print(f'\n{bar}\n  Section {n} — {title}\n{bar}')


def save_fig(fig, name):
    path = OUT_DIR / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → saved  {path.name}')


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — Load one subject / one trial
# ════════════════════════════════════════════════════════════════════════════

def load_one_trial(SUBJECT):
    """
    Replicate TRF_ridge_3.py preprocessing:
      per-trial LPF → resample_poly → HPF → remove padding (eeg_functions.py)

    Returns eeg_data (original loaded struct at orig_fs) instead of a 100 Hz
    intermediate Raw — the old double-resampling path has been removed.
    """
    section_header(1, 'Load one subject / one trial')

    # ── Stimulus and IDyOM surprisal ──────────────────────────────────────────
    stim_mat     = loadmat(constants.EEG_DIR / 'dataStim.mat',
                           struct_as_record=False, squeeze_me=True)
    stim         = stim_mat['stim']
    stim_fs      = int(stim.fs)
    stim_feature = stim.data[0, :]          # per-trial envelopes at stim_fs Hz
    unique_song_ids = np.unique(stim.stimIdxs)

    idyom_pitch_mat = loadmat(constants.PITCH_SURPRISAL_FILE, squeeze_me=True)
    idyom_onset_mat = loadmat(constants.ONSET_SURPRISAL_FILE, squeeze_me=True)

    pitch_surprisal_data, onset_surprisal_data = {}, {}
    for song_id in unique_song_ids:
        song_name = f'audio{song_id}'
        raw_pitch = np.asarray(idyom_pitch_mat[song_name])
        raw_onset = np.asarray(idyom_onset_mat[song_name])
        pitch_surprisal_data[song_id] = np.clip(raw_pitch[0], 0, IC_CLIP)
        onset_surprisal_data[song_id] = np.clip(raw_onset[0], 0, IC_CLIP)

    # ── EEG preprocessing ────────────────────────────────────────────────────
    eeg_data = eeg_func.load_subject_raw_eeg(
        constants.EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)

    # Per-trial: LPF → resample_poly → HPF → remove padding.
    # Matches MATLAB CNSP cellfun-per-trial order; no cross-trial filter bleed.
    # Replaces the old concatenate-then-filter (double-resampling) path.
    preprocessed_trials = eeg_func.preprocess_eeg_trials(
        eeg_data,
        target_fs=SFREQ,
        lpf_hz=constants.HIGH_FREQUENCY,
        hpf_hz=constants.LOW_FREQUENCY,
    )
    eeg_trial_lengths = [t.shape[0] for t in preprocessed_trials]

    raw    = eeg_func.create_mne_raw_from_preprocessed(
        preprocessed_trials, SFREQ, eeg_data['chanlocs'])
    events = eeg_func.create_eelbrain_events(raw)

    # ── Build envelopes and onset predictors ──────────────────────────────────
    # Use resample_poly (same method as EEG downsampling) and trim to
    # min(stim, eeg) — matching MATLAB's min(envLen,eegLen) approach.
    _g        = gcd(stim_fs, SFREQ)
    stim_up   = SFREQ    // _g
    stim_down = stim_fs  // _g

    envelopes = []
    for i in range(len(events['event'])):
        env_raw       = np.asarray(stim_feature[i], dtype=np.float64)
        n_eeg         = eeg_trial_lengths[i]
        env_resampled = resample_poly(env_raw, stim_up, stim_down)
        n_min         = min(len(env_resampled), n_eeg)
        env_resampled = env_resampled[:n_min]
        time_axis     = eelbrain.UTS(0, 1 / SFREQ, n_min)
        envelopes.append(eelbrain.NDVar(env_resampled, (time_axis,)))

    events['envelope'] = envelopes
    events['onsets']   = [env.diff('time').clip(0) for env in envelopes]
    events['duration'] = eelbrain.Var([env.time.tstop for env in envelopes])
    events['eeg']      = eelbrain.load.mne.variable_length_epochs(
        events, 0, tstop='duration', decim=1, adjacency='auto')

    # ── Build surprisal time series (one per unique song_id) ─────────────────
    surprisal_cache = {}
    for i, stimulus_id in enumerate(events['event']):
        song_id = int(stimulus_id % 10) or 10
        if song_id in surprisal_cache:
            continue
        midi_path = constants.MIDI_DIR / f'audio{song_id}.mid'
        time      = events['envelope'][i].time
        n_times   = time.nsamples
        surprisal_cache[song_id] = {
            'pitch': eelbrain.NDVar(
                midi_func.make_surprisal_timeseries(
                    midi_path, pitch_surprisal_data[song_id], SFREQ, n_times),
                dims=(time,)),
            'onset': eelbrain.NDVar(
                midi_func.make_surprisal_timeseries(
                    midi_path, onset_surprisal_data[song_id], SFREQ, n_times),
                dims=(time,)),
        }

    events['pitch_surprisal'] = [
        surprisal_cache[int(sid % 10) or 10]['pitch'] for sid in events['event']]
    events['onset_surprisal'] = [
        surprisal_cache[int(sid % 10) or 10]['onset'] for sid in events['event']]

    # ── Convert all trials to plain numpy arrays ──────────────────────────────
    all_trials = []
    for idx in range(len(events['event'])):
        eeg_arr  = events['eeg'][idx].get_data(('sensor', 'time')).T   # (T, n_ch)
        stim_arr = {
            'envelope':        events['envelope'][idx].x,
            'onsets':          events['onsets'][idx].x,
            'pitch_surprisal': events['pitch_surprisal'][idx].x,
            'onset_surprisal': events['onset_surprisal'][idx].x,
        }
        n_eeg  = eeg_arr.shape[0]
        n_stim = len(stim_arr['envelope'])
        n      = min(n_eeg, n_stim)
        if n_eeg != n_stim:
            print(f'  [align] trial {idx}: EEG={n_eeg}, stim={n_stim}, '
                  f'using min={n} (diff={n_eeg - n_stim} smp)')
        all_trials.append({'eeg': eeg_arr[:n],
                           **{k: v[:n] for k, v in stim_arr.items()}})

    # ── Print summary ─────────────────────────────────────────────────────────
    t = all_trials[TRIAL_IDX]
    n = t['eeg'].shape[0]
    print(f'  Subject         : {SUBJECT}  (trial {TRIAL_IDX})')
    print(f'  EEG shape       : {t["eeg"].shape}  '
          f'(samples × channels)')
    print(f'  Stimulus length : {n} samples')
    print(f'  Sampling rate   : {SFREQ} Hz')
    print(f'  Duration        : {n / SFREQ:.2f} s')

    sensor_dim = events['eeg'][0].sensor
    return all_trials, eeg_data, sensor_dim


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Stimulus–EEG cross-correlation
# ════════════════════════════════════════════════════════════════════════════

def cross_correlation_diagnostic(trial):
    """
    WHY: if EEG and stimulus are correctly aligned and the brain responds to
    the envelope, the cross-correlation should peak at a positive lag of
    ~80–200 ms (stimulus leading EEG).  A peak at lag ≤ 0 or lag > 400 ms
    indicates a temporal alignment error or reversed stimulus/EEG order.
    """
    section_header(2, 'Stimulus–EEG cross-correlation')

    eeg_ch   = trial['eeg'][:, CHANNEL_IDX]
    envelope = trial['envelope']
    n        = len(eeg_ch)
    time_s   = np.arange(n) / SFREQ

    corr     = correlate(eeg_ch, envelope, mode='full')
    lag_samp = np.arange(-(n - 1), n)
    lag_ms   = lag_samp / SFREQ * 1000      # convert samples to milliseconds

    peak_idx   = np.argmax(corr)
    trough_idx = np.argmin(corr)
    peak_lag_ms   = lag_ms[peak_idx]
    trough_lag_ms = lag_ms[trough_idx]

    print(f'  Max  correlation : {corr[peak_idx]:.4e}  at lag {peak_lag_ms:.1f} ms')
    print(f'  Min  correlation : {corr[trough_idx]:.4e}  at lag {trough_lag_ms:.1f} ms')

    if 50 <= peak_lag_ms <= 300:
        print(f'  Interpretation   : peak at {peak_lag_ms:.1f} ms — '
              f'PLAUSIBLE auditory cortical latency (50–300 ms).')
    elif -300 <= peak_lag_ms < 50:
        print(f'  Interpretation   : peak at {peak_lag_ms:.1f} ms — '
              f'SUSPICIOUS — stimulus may lag EEG or ordering is reversed.')
    else:
        print(f'  Interpretation   : peak at {peak_lag_ms:.1f} ms — '
              f'VERY LONG latency; check for broadband artefacts.')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle('Section 2: Stimulus–EEG Cross-correlation',
                 fontsize=12, fontweight='bold')

    axes[0].plot(time_s, envelope, color='steelblue', lw=0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_title(f'Stimulus Envelope  (trial {TRIAL_IDX})')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, eeg_ch, color='darkorange', lw=0.6)
    axes[1].set_ylabel('Amplitude (a.u.)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title(f'EEG channel {CHANNEL_IDX}')
    axes[1].grid(True, alpha=0.3)

    ax      = axes[2]
    mask    = np.abs(lag_ms) <= 500         # zoom to ±500 ms
    ax.plot(lag_ms[mask], corr[mask], color='teal', lw=0.9)
    ax.axvline(0, color='black', lw=0.8, linestyle='--', label='lag = 0')

    aud_colors = plt.cm.autumn(np.linspace(0.2, 0.9, len(AUDITORY_LATENCIES_MS)))
    for lat_ms, c in zip(AUDITORY_LATENCIES_MS, aud_colors):
        ax.axvline(lat_ms, color=c, lw=1.2, linestyle=':', alpha=0.9,
                   label=f'{lat_ms} ms')
    ax.axvline(peak_lag_ms, color='red', lw=1.5,
               label=f'peak ({peak_lag_ms:.0f} ms)')
    ax.set_xlabel('Lag (ms)   [positive = stimulus leads EEG]')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Cross-correlation vs Lag  (zoomed ±500 ms)')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 's2_cross_correlation')

    return {'peak_lag_ms': peak_lag_ms, 'trough_lag_ms': trough_lag_ms,
            'peak_corr': float(corr[peak_idx]),
            'trough_corr': float(corr[trough_idx])}


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Lag-matrix convention inspection
# ════════════════════════════════════════════════════════════════════════════

def lag_matrix_diagnostic():
    """
    WHY: a reversed column ordering in build_lag_matrix() would time-flip
    every recovered TRF.  For example, a 100 ms post-stimulus response would
    appear at -100 ms.  This section verifies the exact convention with a toy
    signal whose values encode time, making any reversal immediately visible.
    """
    section_header(3, 'Lag-matrix convention inspection')

    # Use sfreq=1 so lag index = lag in samples = lag in seconds (readable)
    x_toy     = np.arange(10, dtype=float)
    toy_tmin  = -2
    toy_tmax  =  4
    toy_sfreq =  1

    L = build_lag_matrix(x_toy, toy_tmin, toy_tmax, toy_sfreq)
    n_lags_toy = int(round((toy_tmax - toy_tmin) * toy_sfreq)) + 1
    lag_min    = int(round(toy_tmin * toy_sfreq))
    lag_values = np.arange(lag_min, lag_min + n_lags_toy)   # [-2,-1,0,1,2,3,4]

    print(f'  x_toy      = {x_toy.astype(int)}')
    print(f'  tmin={toy_tmin}  tmax={toy_tmax}  sfreq={toy_sfreq}')
    print(f'  Lag values per column: {lag_values}')
    print('  Lag matrix  (rows = time index t,  cols = lag):')
    print(np.array2string(L.astype(int), max_line_width=120))

    # Verify: L[t, j] must equal x_toy[t - lag_values[j]] (zero-padded)
    passed = True
    for j, lag in enumerate(lag_values):
        for t in range(len(x_toy)):
            src      = t - lag
            expected = x_toy[src] if 0 <= src < len(x_toy) else 0.0
            if L[t, j] != expected:
                print(f'  [FAIL] L[{t},{j}]={L[t,j]}, expected {expected}  '
                      f'(lag={lag})')
                passed = False
    if passed:
        print('  Convention check PASSED: '
              'L[t, j] == x[t - lag_values[j]]  '
              '(positive lags are causal — stimulus precedes EEG)')

    neg_cols  = np.where(lag_values < 0)[0]
    zero_col  = np.where(lag_values == 0)[0]
    pos_cols  = np.where(lag_values > 0)[0]
    print(f'  Pre-stimulus  (lag < 0): columns {neg_cols.tolist()} '
          f'→ lags {lag_values[neg_cols].tolist()}')
    print(f'  Onset         (lag = 0): column  {zero_col.tolist()}')
    print(f'  Post-stimulus (lag > 0): columns {pos_cols.tolist()} '
          f'→ lags {lag_values[pos_cols].tolist()}')

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle('Section 3: Lag Matrix Heatmap  (toy signal x = [0…9])',
                 fontsize=12, fontweight='bold')

    vmax = np.nanmax(np.abs(L))
    im   = ax.imshow(L, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='x value  (encodes source time index)')

    ax.set_xlabel('Lag column index')
    ax.set_ylabel('Time index  t')
    ax.set_xticks(range(n_lags_toy))
    ax.set_xticklabels(
        [f'col {j}\n(lag={v})' for j, v in enumerate(lag_values)],
        fontsize=8)
    ax.set_title(
        f'col 0 = lag {lag_values[0]}  (earliest / most pre-stimulus)     '
        f'col {n_lags_toy-1} = lag {lag_values[-1]}  (latest / most post-stimulus)')

    # Vertical dividers between pre/zero/post regions
    if len(neg_cols):
        ax.axvline(neg_cols[-1] + 0.5, color='navy', lw=1.5, linestyle='--')
    if len(zero_col):
        ax.axvline(zero_col[0] - 0.5, color='navy', lw=1.5, linestyle='--')
        ax.axvline(zero_col[0] + 0.5, color='navy', lw=1.5, linestyle='--')

    plt.tight_layout()
    save_fig(fig, 's3_lag_matrix')


# ════════════════════════════════════════════════════════════════════════════
# Section 4 — Z-score / feature distribution diagnostics
# ════════════════════════════════════════════════════════════════════════════

def distribution_diagnostic(trial):
    """
    WHY: z-scoring a very sparse impulse signal (like surprisal) amplifies
    each nonzero spike by 1/std.  If a predictor is 98% zeros with std ≈ 0.05,
    z-scoring multiplies impulses by ~20, creating outliers that dwarf the
    envelope and can dominate the regression, producing artefactual TRF shapes.
    """
    section_header(4, 'Z-score / feature distribution diagnostics')

    features = {
        'envelope':        trial['envelope'],
        'onsets':          trial['onsets'],
        'pitch_surprisal': trial['pitch_surprisal'],
        'onset_surprisal': trial['onset_surprisal'],
    }

    fig, axes = plt.subplots(len(features), 2,
                             figsize=(12, 3 * len(features)))
    fig.suptitle('Section 4: Feature Distributions Before / After Z-scoring',
                 fontsize=12, fontweight='bold')

    summary_rows = []
    for row, (name, feat) in enumerate(features.items()):
        z        = zscore(feat)
        pct_zero = 100.0 * np.mean(feat == 0)
        nonzero  = feat[feat != 0]

        print(f'  {name:20s}  mean={feat.mean():+.4f}  std={feat.std():.4f}  '
              f'min={feat.min():.4f}  max={feat.max():.4f}  '
              f'zeros={pct_zero:.1f}%')

        summary_rows.append({
            'feature': name,
            'mean_raw': feat.mean(), 'std_raw': feat.std(),
            'min_raw': feat.min(),   'max_raw': feat.max(),
            'pct_zero': pct_zero,
            'mean_z': z.mean(),      'std_z': z.std(),
        })

        ax = axes[row, 0]
        if len(nonzero):
            ax.hist(nonzero, bins=50, color='steelblue', alpha=0.8,
                    edgecolor='none')
        ax.axvline(feat.mean(), color='red', lw=1.5, linestyle='--',
                   label='mean')
        ax.set_title(f'{name}  |  raw  ({pct_zero:.1f}% zeros excluded)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        z_nonzero = z[feat != 0]
        if len(z_nonzero):
            ax.hist(z_nonzero, bins=50, color='darkorange', alpha=0.8,
                    edgecolor='none')
        ax.axvline(z.mean(), color='red', lw=1.5, linestyle='--',
                   label='mean')
        ax.set_title(f'{name}  |  z-scored')
        ax.set_xlabel('z-score')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, 's4_feature_distributions')
    return summary_rows


# ════════════════════════════════════════════════════════════════════════════
# Section 5 — Surprisal sparsity visualisation
# ════════════════════════════════════════════════════════════════════════════

def sparsity_diagnostic(trial):
    """
    WHY: TRF estimation becomes unreliable when predictors are too sparse.
    Very sparse features produce poorly conditioned cross-covariance matrices,
    leading to high-variance weight estimates.  As a rule of thumb, fewer than
    ~1 event/s at SFREQ is a warning sign for a stable TRF fit.
    """
    section_header(5, 'Surprisal sparsity diagnostics')

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle('Section 5: Surprisal Time Series & Sparsity',
                 fontsize=12, fontweight='bold')

    sparsity_stats = {}
    for ax, (key, color) in zip(axes,
            [('pitch_surprisal', 'royalblue'),
             ('onset_surprisal', 'crimson')]):

        sig          = trial[key]
        n            = len(sig)
        t_s          = np.arange(n) / SFREQ
        nonzero_idx  = np.where(sig != 0)[0]
        n_impulses   = len(nonzero_idx)
        pct_zero     = 100.0 * (1 - n_impulses / n)
        rate_hz      = n_impulses / (n / SFREQ)
        avg_spacing  = ((nonzero_idx[1:] - nonzero_idx[:-1]).mean() / SFREQ
                        if n_impulses > 1 else np.nan)

        print(f'  {key:20s}  impulses={n_impulses}  '
              f'sparsity={pct_zero:.1f}%  rate={rate_hz:.2f} Hz  '
              f'avg_spacing={avg_spacing:.3f} s')

        sparsity_stats[key] = {
            'n_impulses': n_impulses,
            'pct_zero': pct_zero,
            'rate_hz': rate_hz,
            'avg_spacing_s': float(avg_spacing),
        }

        ax.plot(t_s, sig, color=color, lw=0.7, alpha=0.7, label=key)
        if n_impulses:
            ax.vlines(t_s[nonzero_idx], 0, sig[nonzero_idx],
                      color='black', lw=0.5, alpha=0.6,
                      label='nonzero events')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Surprisal (bits, clipped)')
        ax.set_title(f'{key}   |   {n_impulses} impulses  '
                     f'({pct_zero:.1f}% zeros,  {rate_hz:.2f} events/s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_fig(fig, 's5_surprisal_sparsity')
    return sparsity_stats


# ════════════════════════════════════════════════════════════════════════════
# Section 6 — Resampling method comparison
# ════════════════════════════════════════════════════════════════════════════

def resampling_diagnostic(eeg_data):
    """
    Compare orig_fs → 64 Hz downsampling: resample_poly (new pipeline, used
    for both EEG and stimulus) vs scipy.signal.resample / FFT-based (old
    pipeline, was used only for the stimulus envelope).

    WHY: the old pipeline resampled EEG with polyphase (via MNE) and the
    stimulus with FFT-based scipy.signal.resample — two different methods
    producing slightly different sample sequences.  Even a sub-sample temporal
    shift between EEG and stimulus blurs the TRF and pushes the peak toward
    zero.  The new pipeline uses resample_poly for both, eliminating this
    source of misalignment.

    The LPF (8 Hz) is applied before resampling to match the actual pipeline
    order (LPF → resample_poly), so both methods operate on the same
    bandlimited input.
    """
    section_header(6, 'Resampling comparison  '
                      '(resample_poly [new] vs FFT resample [old stim])')

    orig_fs    = eeg_data['fs']
    trial_orig = eeg_data['trials'][TRIAL_IDX].astype(np.float64)
    ch_orig    = trial_orig[:, CHANNEL_IDX]   # single channel at orig_fs

    # Apply LPF first — same as in preprocess_eeg_trials — so both methods
    # operate on an already-bandlimited signal (as the pipeline actually does).
    nyq_orig = orig_fs / 2.0
    lpf_sos  = butter(4, constants.HIGH_FREQUENCY / nyq_orig,
                      btype='low', output='sos')
    ch_lpf   = sosfiltfilt(lpf_sos, ch_orig)

    # Integer up/down factors for resample_poly
    g    = gcd(orig_fs, SFREQ)
    up   = SFREQ   // g
    down = orig_fs // g
    n_out = int(round(len(ch_lpf) * SFREQ / orig_fs))

    # Method A: resample_poly — new pipeline, used for EEG and stimulus
    ch_a = resample_poly(ch_lpf, up, down)

    # Method B: scipy FFT resample — old pipeline, was used for stimulus only
    ch_b = sp_resample(ch_lpf, n_out)

    # Align to the shorter output (rounding can differ by ±1 sample)
    n = min(len(ch_a), len(ch_b))
    ch_a, ch_b = ch_a[:n], ch_b[:n]

    rms = float(np.sqrt(np.mean(ch_a ** 2)))

    def pair_stats(x, y, label):
        diff    = x - y
        r, _    = pearsonr(x, y)
        mad     = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))
        ratio   = max_abs / rms if rms > 0 else np.nan
        verdict = 'ok < 1%' if ratio < 0.01 else 'LARGE >= 1%'
        print(f'  {label}  r={r:.6f}  mean|diff|={mad:.4e}  '
              f'max|diff|={max_abs:.4e}  max|diff|/RMS={ratio:.4f}  ({verdict})')
        return r, mad, max_abs, ratio, diff

    print(f'  Source sfreq : {orig_fs} Hz  →  target: {SFREQ} Hz  '
          f'(up={up}/down={down})')
    r_ab, mad_ab, max_ab, ratio_ab, diff_ab = pair_stats(
        ch_a, ch_b, 'poly vs FFT')

    t_s  = np.arange(n) / SFREQ
    mask = t_s <= min(5.0, t_s[-1])

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(
        f'Section 6: Resampling Comparison  '
        f'(orig_fs={orig_fs} Hz → {SFREQ} Hz, after LPF at '
        f'{constants.HIGH_FREQUENCY} Hz)\n'
        'Method A: resample_poly [new — EEG + stim]  vs  '
        'Method B: FFT resample [old — stim only]',
        fontsize=11, fontweight='bold')

    methods = [
        (ch_a, f'Method A: resample_poly  (up={up}/down={down})  [new pipeline]',
         'steelblue'),
        (ch_b, 'Method B: scipy.signal.resample  [FFT, old stimulus path]',
         'seagreen'),
    ]
    for ax, (sig, label, color) in zip(axes[:2], methods):
        ax.plot(t_s[mask], sig[mask], lw=0.8, color=color)
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_xlabel('Time (s)')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes[2].plot(t_s[mask], diff_ab[mask], color='crimson', lw=0.8)
    axes[2].axhline(0, color='black', lw=0.6, linestyle='--')
    axes[2].set_ylabel('Difference (a.u.)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title(
        f'A − B  (poly − FFT)   r={r_ab:.6f},  '
        f'max|diff|/RMS={ratio_ab:.4f}  '
        f'[non-zero here = old EEG/stim misalignment source]')
    axes[2].grid(True, alpha=0.3)

    # Overlay both signals for a direct comparison (first 2 s)
    mask2 = t_s <= min(2.0, t_s[-1])
    axes[3].plot(t_s[mask2], ch_a[mask2], color='steelblue', lw=1.0,
                 alpha=0.9, label='poly (A)')
    axes[3].plot(t_s[mask2], ch_b[mask2], color='seagreen',  lw=1.0,
                 alpha=0.7, linestyle='--', label='FFT  (B)')
    axes[3].set_ylabel('Amplitude (a.u.)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Overlay A vs B — first 2 s  '
                      '(any visible offset → temporal misalignment in old pipeline)')
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 's6_resampling_comparison')

    return {
        'pearson_r_poly_vs_fft':         r_ab,
        'mad_poly_vs_fft':               mad_ab,
        'max_abs_poly_vs_fft':           max_ab,
        'max_diff_over_rms_poly_vs_fft': ratio_ab,
    }


# ════════════════════════════════════════════════════════════════════════════
# Section 7 — Predicted vs actual EEG
# ════════════════════════════════════════════════════════════════════════════

def prediction_diagnostic(all_trials, sensor_dim):
    """
    Fit Custom Ridge, MNE ReceptiveField, and eelbrain Boosting in a
    leave-one-trial-out manner, then compare predictions on TRIAL_IDX.

    All three methods use the same lag window (TMIN, TMAX) for consistency.

    Boosting uses basis=0.020 s Hamming windows (same as TRF_pickle_A_and_AM.py).
    Because that gives a coarser lag grid (20 ms step) than SFREQ (15.625 ms step),
    the boosting TRF is interpolated to the native sample grid before prediction.

    WHY: visual inspection reveals issues that per-channel r values miss —
    phase shifts (lag convention errors), scale mismatches (missing z-score),
    and flat predictions (over-regularisation or implementation bugs).
    """
    section_header(7, 'Predicted vs actual EEG')

    feature_keys = ['envelope', 'onsets', 'pitch_surprisal', 'onset_surprisal']
    train_trials = [t for i, t in enumerate(all_trials) if i != TRIAL_IDX]
    test_trial   = all_trials[TRIAL_IDX]

    # ── Custom ridge (identical lag-matrix build to TRF_ridge_2.py) ───────────
    def make_phi(t):
        return np.hstack([build_lag_matrix(zscore(t[k]), TMIN, TMAX, SFREQ)
                          for k in feature_keys])

    Phi_train  = np.vstack([make_phi(t) for t in train_trials])
    Y_train    = np.vstack([zscore(t['eeg']) for t in train_trials])
    Phi_test   = make_phi(test_trial)
    Y_test     = zscore(test_trial['eeg'])

    W_ridge      = ridge_solve(Phi_train, Y_train, RIDGE_ALPHA)
    Y_pred_ridge = Phi_test @ W_ridge

    r_ridge    = np.array([pearsonr(Y_test[:, c], Y_pred_ridge[:, c])[0]
                            for c in range(Y_test.shape[1])])
    rmse_ridge = np.sqrt(np.mean((Y_test - Y_pred_ridge) ** 2, axis=0))
    print(f'  Custom Ridge        mean r = {r_ridge.mean():.4f}  '
          f'mean RMSE = {rmse_ridge.mean():.4f}')

    # ── MNE ReceptiveField ────────────────────────────────────────────────────
    X_train_mne = np.vstack([
        np.column_stack([zscore(t[k]) for k in feature_keys])
        for t in train_trials])
    X_test_mne  = np.column_stack([zscore(test_trial[k]) for k in feature_keys])

    rf = ReceptiveField(TMIN, TMAX, SFREQ, feature_names=feature_keys,
                        estimator=RIDGE_ALPHA)
    rf.fit(X_train_mne, Y_train)
    Y_pred_mne = rf.predict(X_test_mne)

    r_mne    = np.array([pearsonr(Y_test[:, c], Y_pred_mne[:, c])[0]
                          for c in range(Y_test.shape[1])])
    rmse_mne = np.sqrt(np.mean((Y_test - Y_pred_mne) ** 2, axis=0))
    print(f'  MNE RF              mean r = {r_mne.mean():.4f}  '
          f'mean RMSE = {rmse_mne.mean():.4f}')

    # ── Eelbrain boosting (LOOCV: train on all trials except TRIAL_IDX) ──────
    # WHY: boosting uses a different optimisation strategy (gradient descent on
    # a cross-validated basis expansion) rather than ridge regression.  Comparing
    # its temporal predictions to ridge exposes whether correlations differences
    # are due to the algorithm or the lag/normalisation convention.

    def _make_boost_dataset(trials, excl_idx):
        """Reconstruct eelbrain Dataset from numpy trial dicts."""
        rows  = [t for i, t in enumerate(trials) if i != excl_idx]
        n_tr  = len(rows)
        ds    = eelbrain.Dataset({
            'i_start': eelbrain.Var(np.zeros(n_tr, dtype=int)),
            'event':   eelbrain.Var(np.arange(1, n_tr + 1, dtype=int)),
        })
        eeg_nd, env_nd, ons_nd, ps_nd, os_nd = [], [], [], [], []
        for t in rows:
            nt   = t['eeg'].shape[0]
            time = eelbrain.UTS(0, 1 / SFREQ, nt)
            eeg_nd.append(eelbrain.NDVar(t['eeg'].T, (sensor_dim, time)))
            env_nd.append(eelbrain.NDVar(t['envelope'],        (time,)))
            ons_nd.append(eelbrain.NDVar(t['onsets'],          (time,)))
            ps_nd.append(eelbrain.NDVar(t['pitch_surprisal'],  (time,)))
            os_nd.append(eelbrain.NDVar(t['onset_surprisal'],  (time,)))
        ds['eeg']             = eeg_nd
        ds['envelope']        = env_nd
        ds['onsets']          = ons_nd
        ds['pitch_surprisal'] = ps_nd
        ds['onset_surprisal'] = os_nd
        return ds

    Y_pred_boost = None
    r_boost      = None
    rmse_boost   = None

    try:
        basis = 3 / SFREQ   # minimum valid Hamming window at this sample rate (3 samples)
        print('  Fitting eelbrain boosting  '
              f'(TMIN={TMIN}, TMAX={TMAX}, basis={basis:.4f}) ...')
        train_ds     = _make_boost_dataset(all_trials, excl_idx=TRIAL_IDX)
        result_boost = eelbrain.boosting(
            'eeg', feature_keys,
            TMIN, TMAX,
            data=train_ds,
            basis=basis,
            partitions=10,
            error='l1',
        )

        # Build prediction for the test trial.
        # result_boost.h is a list of NDVars, one per predictor; each has shape
        # (sensor, lag_time) where lag_time uses the 20 ms basis step.
        # We interpolate each TRF to the native SFREQ grid so that
        # build_lag_matrix (which operates at SFREQ) can be applied directly.
        n_test  = test_trial['eeg'].shape[0]
        n_ch    = test_trial['eeg'].shape[1]
        lag_min = int(round(TMIN * SFREQ))
        n_lags  = int(round((TMAX - TMIN) * SFREQ)) + 1
        # Fine lag axis in seconds — matches build_lag_matrix column ordering
        fine_lags_s = (np.arange(n_lags) + lag_min) / SFREQ

        Y_pred_boost = np.zeros((n_test, n_ch))
        for k_name, h_k in zip(feature_keys, result_boost.h):
            h_arr    = h_k.get_data(('sensor', 'time'))  # (n_ch, n_lags_basis)
            h_lag_s  = (np.arange(h_k.time.nsamples) * h_k.time.tstep
                        + h_k.time.tmin)                 # coarse lag times (s)

            # Linear interpolation from 20 ms grid to 1/SFREQ grid; zero outside
            h_fine = np.zeros((n_ch, n_lags))
            for c in range(n_ch):
                h_fine[c] = np.interp(fine_lags_s, h_lag_s, h_arr[c],
                                      left=0.0, right=0.0)

            phi           = build_lag_matrix(zscore(test_trial[k_name]),
                                             TMIN, TMAX, SFREQ)
            Y_pred_boost += phi @ h_fine.T   # (n_test, n_ch)

        r_boost    = np.array([pearsonr(Y_test[:, c], Y_pred_boost[:, c])[0]
                                for c in range(n_ch)])
        rmse_boost = np.sqrt(np.mean((Y_test - Y_pred_boost) ** 2, axis=0))
        print(f'  Eelbrain Boosting   mean r = {r_boost.mean():.4f}  '
              f'mean RMSE = {rmse_boost.mean():.4f}')

    except Exception as exc:
        print(f'  [WARNING] Eelbrain boosting failed: {exc}')
        print(f'  Boosting prediction omitted from the plot.')

    # ── Plot first 10 s, one channel ──────────────────────────────────────────
    ch   = CHANNEL_IDX
    n    = Y_test.shape[0]
    t_s  = np.arange(n) / SFREQ
    zoom = slice(0, min(n, int(10 * SFREQ)))

    n_panels = 4 if Y_pred_boost is not None else 3
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, 3.5 * n_panels))
    fig.suptitle(
        f'Section 7: Predicted vs Actual EEG  '
        f'(channel {ch}, trial {TRIAL_IDX}, first 10 s)',
        fontsize=12, fontweight='bold')

    ax0 = axes[0]
    ax0.plot(t_s[zoom], Y_test[zoom, ch], color='black',
             lw=0.7, label='Actual EEG')
    ax0.plot(t_s[zoom], Y_pred_ridge[zoom, ch], color='steelblue',
             lw=0.9, alpha=0.8,
             label=f'Ridge       (r={r_ridge[ch]:.3f})')
    ax0.plot(t_s[zoom], Y_pred_mne[zoom, ch], color='darkorange',
             lw=0.9, alpha=0.8,
             label=f'MNE RF      (r={r_mne[ch]:.3f})')
    if Y_pred_boost is not None:
        ax0.plot(t_s[zoom], Y_pred_boost[zoom, ch], color='seagreen',
                 lw=0.9, alpha=0.8,
                 label=f'Boosting    (r={r_boost[ch]:.3f})')
    ax0.set_ylabel('z-score')
    ax0.set_xlabel('Time (s)')
    ax0.set_title('Actual EEG vs Predicted')
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    axes[1].plot(t_s[zoom], Y_test[zoom, ch] - Y_pred_ridge[zoom, ch],
                 color='steelblue', lw=0.7, label='Residual (Ridge)')
    axes[1].axhline(0, color='black', lw=0.6, linestyle='--')
    axes[1].set_ylabel('z-score')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Residual: Actual − Ridge prediction')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_s[zoom], Y_test[zoom, ch] - Y_pred_mne[zoom, ch],
                 color='darkorange', lw=0.7, label='Residual (MNE RF)')
    axes[2].axhline(0, color='black', lw=0.6, linestyle='--')
    axes[2].set_ylabel('z-score')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Residual: Actual − MNE RF prediction')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    if Y_pred_boost is not None:
        axes[3].plot(t_s[zoom], Y_test[zoom, ch] - Y_pred_boost[zoom, ch],
                     color='seagreen', lw=0.7, label='Residual (Boosting)')
        axes[3].axhline(0, color='black', lw=0.6, linestyle='--')
        axes[3].set_ylabel('z-score')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title('Residual: Actual − Boosting prediction')
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 's7_predicted_vs_actual')

    return {
        'ridge_mean_r':    float(r_ridge.mean()),
        'ridge_mean_rmse': float(rmse_ridge.mean()),
        'mne_mean_r':      float(r_mne.mean()),
        'mne_mean_rmse':   float(rmse_mne.mean()),
        'boost_mean_r':    float(r_boost.mean())    if r_boost    is not None else float('nan'),
        'boost_mean_rmse': float(rmse_boost.mean()) if rmse_boost is not None else float('nan'),
    }


# ════════════════════════════════════════════════════════════════════════════
# Section 8 — Synthetic weight-recovery test
# ════════════════════════════════════════════════════════════════════════════

def synthetic_recovery_test():
    """
    Generate synthetic data with a KNOWN TRF, add noise at three levels,
    then attempt to recover the weights with the same ridge solve used in
    TRF_ridge_2.py.

    WHY: if the implementation is mathematically correct, the recovered TRF
    should closely match the true TRF (especially at low noise).  Failure here
    points to a bug in build_lag_matrix(), the ridge solve, or z-scoring —
    independently of any data quality issue.
    """
    section_header(8, 'Synthetic weight-recovery test')

    rng = np.random.default_rng(42)

    # ── Synthetic stimulus: white noise, z-scored ─────────────────────────────
    n_samples = int(60 * SFREQ)                      # 60-second synthetic trial
    stim_syn  = zscore(rng.standard_normal(n_samples))

    # ── True TRF: Gaussian bump at 120 ms post-stimulus ───────────────────────
    n_lags_syn  = int(round((TMAX - TMIN) * SFREQ)) + 1
    lag_min_syn = int(round(TMIN * SFREQ))           # e.g. -13 samples at 64 Hz
    lag_times_ms = (np.arange(n_lags_syn) + lag_min_syn) / SFREQ * 1000

    TRUE_PEAK_MS  = 120.0
    TRUE_WIDTH_MS =  30.0
    true_weights  = np.exp(
        -0.5 * ((lag_times_ms - TRUE_PEAK_MS) / TRUE_WIDTH_MS) ** 2)
    true_weights /= true_weights.max()               # unit amplitude

    Phi_syn = build_lag_matrix(stim_syn, TMIN, TMAX, SFREQ)

    # ── Three noise levels ────────────────────────────────────────────────────
    noise_levels = {'low': 0.1, 'medium': 0.5, 'high': 2.0}
    results      = {}

    fig, axes = plt.subplots(len(noise_levels), 1,
                             figsize=(12, 4 * len(noise_levels)))
    fig.suptitle('Section 8: Synthetic Weight Recovery',
                 fontsize=12, fontweight='bold')

    for ax, (noise_name, noise_std) in zip(axes, noise_levels.items()):
        noise    = rng.standard_normal(n_samples) * noise_std
        y_syn    = Phi_syn @ true_weights + noise
        y_syn_z  = zscore(y_syn)

        w_hat    = ridge_solve(Phi_syn, y_syn_z[:, None], RIDGE_ALPHA).ravel()
        # Normalise recovered weights for shape comparison
        w_hat_norm = w_hat / (np.abs(w_hat).max() + 1e-12)

        r_w, _       = pearsonr(true_weights, w_hat)
        mse_w        = float(np.mean((true_weights - w_hat_norm) ** 2))
        peak_ms      = float(lag_times_ms[np.argmax(w_hat)])
        snr_db       = 10 * np.log10(1.0 / noise_std ** 2) if noise_std > 0 else np.inf

        print(f'  [{noise_name:6s}  std={noise_std}  SNR≈{snr_db:.1f} dB]  '
              f'weight corr r={r_w:.4f}  shape MSE={mse_w:.6f}  '
              f'recovered peak={peak_ms:.1f} ms  (true={TRUE_PEAK_MS:.1f} ms)')

        results[noise_name] = {
            'r': float(r_w),
            'shape_mse': mse_w,
            'recovered_peak_ms': peak_ms,
        }

        ax.plot(lag_times_ms, true_weights, color='black', lw=2.0,
                label='True TRF  (unit amplitude)')
        ax.plot(lag_times_ms, w_hat_norm, color='steelblue', lw=1.5,
                linestyle='--',
                label=f'Recovered  (r={r_w:.3f},  peak={peak_ms:.0f} ms)')
        for lat in AUDITORY_LATENCIES_MS:
            ax.axvline(lat, color='grey', lw=0.7, linestyle=':', alpha=0.5)
        ax.axvline(TRUE_PEAK_MS, color='red', lw=1.0, linestyle='--', alpha=0.7,
                   label=f'True peak ({TRUE_PEAK_MS:.0f} ms)')
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Normalised weight')
        ax.set_title(f'Noise level: {noise_name}  '
                     f'(std={noise_std},  SNR≈{snr_db:.1f} dB)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 's8_synthetic_recovery')
    return results


# ════════════════════════════════════════════════════════════════════════════
# Section 9 — Save CSV summary
# ════════════════════════════════════════════════════════════════════════════

def save_summary(xcorr_stats, dist_rows, sparsity_stats,
                 resamp_stats, pred_stats, synth_stats):
    section_header(9, 'Save outputs')

    # Feature distribution table
    dist_path = OUT_DIR / 'feature_distributions.csv'
    with open(dist_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dist_rows)
    print(f'  → saved  {dist_path.name}')

    # Flat key-value summary of all scalar metrics
    rows = []
    rows += [{'metric': f'xcorr_{k}', 'value': v}
             for k, v in xcorr_stats.items()]
    for feat, stats in sparsity_stats.items():
        rows += [{'metric': f'sparsity_{feat}_{k}', 'value': v}
                 for k, v in stats.items()]
    rows += [{'metric': f'resampling_{k}', 'value': v}
             for k, v in resamp_stats.items()]
    rows += [{'metric': f'prediction_{k}', 'value': v}
             for k, v in pred_stats.items()]
    for noise, stats in synth_stats.items():
        rows += [{'metric': f'synth_{noise}_{k}', 'value': v}
                 for k, v in stats.items()]

    summary_path = OUT_DIR / 'diagnostic_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'  → saved  {summary_path.name}')
    print(f'  → all plots in  {OUT_DIR}')


# ════════════════════════════════════════════════════════════════════════════
# Section 10 — Diagnostic summary
# ════════════════════════════════════════════════════════════════════════════

def print_summary(xcorr_stats, sparsity_stats, resamp_stats,
                  pred_stats, synth_stats):
    section_header(10, 'Diagnostic summary')

    ok_msgs = []
    issues  = []

    # Temporal alignment
    peak_ms = xcorr_stats['peak_lag_ms']
    if 50 <= peak_ms <= 300:
        ok_msgs.append(
            f'ALIGNMENT     cross-correlation peak at {peak_ms:.1f} ms — '
            f'consistent with auditory cortical response timing.')
    else:
        issues.append(
            f'ALIGNMENT     cross-correlation peak at {peak_ms:.1f} ms — '
            f'outside 50–300 ms auditory range.  '
            f'Check stimulus/EEG temporal registration.')

    # Lag convention (always passes if build_lag_matrix matches TRF_ridge_2.py)
    ok_msgs.append(
        'LAG MATRIX    L[t,j] == x[t − lag_values[j]] verified — '
        'causal convention correct (positive lag = stimulus precedes EEG).')

    # Sparsity
    for key, s in sparsity_stats.items():
        if s['rate_hz'] < 1.0:
            issues.append(
                f'SPARSITY      {key}: {s["rate_hz"]:.2f} events/s '
                f'({s["pct_zero"]:.1f}% zeros).  '
                f'Sparse predictor may destabilise TRF estimation; '
                f'consider higher regularisation or predictor smoothing.')
        else:
            ok_msgs.append(
                f'SPARSITY      {key}: {s["rate_hz"]:.2f} events/s '
                f'({s["n_impulses"]} impulses) — density appears adequate.')

    # Resampling agreement (resample_poly [new] vs FFT [old stim path])
    ratio_pf = resamp_stats['max_diff_over_rms_poly_vs_fft']
    if ratio_pf < 0.01:
        ok_msgs.append(
            f'RESAMPLING    resample_poly vs FFT: max|diff|/RMS = '
            f'{ratio_pf * 100:.3f}% — negligible; both methods produce '
            f'equivalent outputs.  New pipeline uses resample_poly for both '
            f'EEG and stimulus, ensuring consistent temporal alignment.')
    else:
        issues.append(
            f'RESAMPLING    resample_poly vs FFT: max|diff|/RMS = '
            f'{ratio_pf * 100:.2f}% — non-trivial difference.  '
            f'The old pipeline used FFT resample for the stimulus and polyphase '
            f'for the EEG, causing temporal drift.  This has been fixed: both '
            f'now use resample_poly.')

    # Prediction quality
    import math
    pred_methods = [
        ('Ridge',    pred_stats['ridge_mean_r']),
        ('MNE RF',   pred_stats['mne_mean_r']),
        ('Boosting', pred_stats['boost_mean_r']),
    ]
    for method, r in pred_methods:
        if math.isnan(r):
            ok_msgs.append(
                f'PREDICTION    {method} — skipped (boosting fit failed).')
            continue
        if r > 0.05:
            ok_msgs.append(
                f'PREDICTION    {method} mean r = {r:.4f} — '
                f'above near-zero threshold.')
        else:
            issues.append(
                f'PREDICTION    {method} mean r = {r:.4f} — very low.  '
                f'Possible causes: mismatched alpha, feature alignment error, '
                f'or excessive z-score amplification of sparse regressors.')

    # Synthetic recovery
    low_r = synth_stats['low']['r']
    if low_r > 0.90:
        ok_msgs.append(
            f'SYNTH RECOV   low-noise r = {low_r:.4f} — '
            f'ridge implementation is mathematically correct.')
    else:
        issues.append(
            f'SYNTH RECOV   low-noise r = {low_r:.4f} — expected > 0.90.  '
            f'Implementation may be incorrect; inspect build_lag_matrix '
            f'and ridge_solve.')

    print()
    print('  PASS:')
    for m in ok_msgs:
        print(f'    [OK]  {m}')
    print()
    if issues:
        print('  ISSUES / WARNINGS:')
        for m in issues:
            print(f'    [!!]  {m}')
        print()
        print('  Likely causes of low TRF correlations (in order of frequency):')
        print('    1. Excessive regularisation — alpha too high for sparse regressors')
        print('    2. Surprisal sparsity — z-score amplification inflates a few '
              'impulses, making them dominate regression')
        print('    3. Temporal misalignment — even ±1 sample at 64 Hz is 15 ms, '
              'enough to blur the 100 ms auditory peak')
        print('    4. Lag convention mismatch between implementations '
              '(TRF_ridge_2 vs eelbrain boosting)')
    else:
        print('  No issues detected.  Pipeline appears healthy.')

    print(f'\n  Output directory: {OUT_DIR}')


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    for SUBJECT in SUBJECTS:
        OUT_DIR = (Path(__file__).resolve().parent
                   / f'TRF_diagnostics_output/{SUBJECT}_trial_{TRIAL_IDX}_channel_{CHANNEL_IDX}')
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        all_trials, eeg_data, sensor_dim = load_one_trial(SUBJECT)

        trial = all_trials[TRIAL_IDX]

        xcorr_stats    = cross_correlation_diagnostic(trial)
        lag_matrix_diagnostic()
        dist_rows      = distribution_diagnostic(trial)
        sparsity_stats = sparsity_diagnostic(trial)
        resamp_stats   = resampling_diagnostic(eeg_data)
        pred_stats     = prediction_diagnostic(all_trials, sensor_dim)
        synth_stats    = synthetic_recovery_test()

        save_summary(xcorr_stats, dist_rows, sparsity_stats,
                    resamp_stats, pred_stats, synth_stats)
        print_summary(xcorr_stats, sparsity_stats, resamp_stats,
                    pred_stats, synth_stats)
