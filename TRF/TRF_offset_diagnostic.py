"""
TRF_offset_diagnostic.py
────────────────────────────────────────────────────────────────────────────────
Systematic stimulus-offset sweep to identify the optimal temporal delay between
the audio envelope and EEG that maximises Pearson correlation.

For each combination of subject × channel × offset value the envelope is
shifted forward in time by `offset_ms` samples (simulating a brain response
at that latency) and Pearson r is computed on the full concatenated session.
Results are aggregated across subjects (per channel), channels (per subject),
and globally for a bar chart.

Run from musical-surprisal/TRF/:
    python TRF_offset_diagnostic.py
"""

import sys
from pathlib import Path
from math import gcd
import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import resample_poly, correlate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants
import eeg_functions as eeg_func


# ════════════════════════════════════════════════════════════════════════════════
# Parameters — edit these to select a subset of subjects / channels
# ════════════════════════════════════════════════════════════════════════════════

SUBJECTS         = constants.SUBJECTS   # e.g. ['Sub2', 'Sub3'] for a quick test
CHANNELS         = list(range(64))      # e.g. [0, 1, 2] for a quick test
OFFSET_VALUES_MS = [0, 50, 100, 150, 200, 250, 300, 350]
SFREQ            = 64                   # Hz — must match preprocessing

# Lag range (ms) for the continuous cross-correlation peak search
XCORR_SEARCH_MS = (0, 500)

OUT_DIR = Path(__file__).resolve().parent / 'TRF_offset_diagnostic_output'


# ════════════════════════════════════════════════════════════════════════════════
# Load stimulus envelopes — done once, shared across all subjects
# ════════════════════════════════════════════════════════════════════════════════

stim_mat    = loadmat(constants.EEG_DIR / 'dataStim.mat',
                      struct_as_record=False, squeeze_me=True)
stim        = stim_mat['stim']
stim_fs     = int(stim.fs)
stim_trials = stim.data[0, :]   # per-trial envelopes at stim_fs Hz

_g        = gcd(stim_fs, SFREQ)
stim_up   = SFREQ   // _g
stim_down = stim_fs // _g


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def load_subject_concat(subject):
    """
    Preprocess and concatenate all trials for one subject.

    Pipeline (per trial, matching TRF_ridge_3.py):
        LPF at orig_fs  →  resample_poly to SFREQ  →  HPF at SFREQ  →  remove padding

    Returns
    -------
    eeg_concat : ndarray (n_total, n_channels)
    env_concat : ndarray (n_total,)
    """
    eeg_data = eeg_func.load_subject_raw_eeg(
        constants.EEG_DIR / f'data{subject}.mat', subject)

    preprocessed = eeg_func.preprocess_eeg_trials(
        eeg_data,
        target_fs=SFREQ,
        lpf_hz=constants.HIGH_FREQUENCY,
        hpf_hz=constants.LOW_FREQUENCY,
    )

    eeg_segs, env_segs = [], []
    for i, trial_eeg in enumerate(preprocessed):
        # trial_eeg: (n_time, n_channels)
        env_raw = np.asarray(stim_trials[i], dtype=np.float64)
        env_res = resample_poly(env_raw, stim_up, stim_down)

        n_min = min(trial_eeg.shape[0], len(env_res))
        diff  = abs(trial_eeg.shape[0] - len(env_res))
        if diff > 4 * SFREQ:
            warnings.warn(
                f'{subject} trial {i}: large stim/EEG mismatch '
                f'({diff} samples = {diff / SFREQ:.2f} s). '
                'Check padding removal and resampling.')

        eeg_segs.append(trial_eeg[:n_min, :])
        env_segs.append(env_res[:n_min])

    return np.vstack(eeg_segs), np.concatenate(env_segs)


def pearson_r_vectorized(env, eeg):
    """
    Pearson r between env (n,) and every column of eeg (n, n_ch).

    Returns
    -------
    ndarray (n_ch,)  — NaN for zero-variance channels
    """
    env_c   = env - env.mean()
    env_std = env.std()
    if env_std < 1e-12:
        return np.zeros(eeg.shape[1])
    env_z = env_c / env_std

    eeg_c   = eeg - eeg.mean(axis=0)
    eeg_std = eeg.std(axis=0)
    safe    = eeg_std > 1e-12
    eeg_z   = np.where(safe, eeg_c / np.where(safe, eeg_std, 1.0), 0.0)

    return (env_z @ eeg_z) / len(env)


def compute_xcorr_peak_ms(env, eeg_ch):
    """
    Normalised cross-correlation peak lag (ms) restricted to XCORR_SEARCH_MS.
    Positive lag means the stimulus leads the EEG (causal auditory response).
    Returns NaN if either signal has zero variance.
    """
    n     = len(env)
    s_env = env.std()
    s_eeg = eeg_ch.std()
    if s_env < 1e-12 or s_eeg < 1e-12:
        return np.nan

    corr    = correlate(eeg_ch - eeg_ch.mean(), env - env.mean(), mode='full')
    corr   /= s_eeg * s_env * n

    lag_ms  = (np.arange(-(n - 1), n)) / SFREQ * 1000.0
    mask    = (lag_ms >= XCORR_SEARCH_MS[0]) & (lag_ms <= XCORR_SEARCH_MS[1])
    if not np.any(mask):
        return np.nan

    return float(lag_ms[mask][np.argmax(corr[mask])])


# ════════════════════════════════════════════════════════════════════════════════
# Main computation
# ════════════════════════════════════════════════════════════════════════════════

n_subj  = len(SUBJECTS)
n_ch    = len(CHANNELS)
n_off   = len(OFFSET_VALUES_MS)

corr_matrix = np.full((n_subj, n_ch, n_off), np.nan)
xcorr_peaks = np.full((n_subj, n_ch),        np.nan)

print(f'Offset sweep: {n_subj} subjects × {n_ch} channels × {n_off} offsets')
print(f'Offsets (ms): {OFFSET_VALUES_MS}\n')

for s_idx, subject in enumerate(SUBJECTS):
    print(f'[{s_idx + 1}/{n_subj}]  {subject}')
    eeg_concat, env_concat = load_subject_concat(subject)

    # Per-offset Pearson r (all channels at once)
    for off_idx, offset_ms in enumerate(OFFSET_VALUES_MS):
        k = int(round(offset_ms * SFREQ / 1000))
        if k == 0:
            env_trim = env_concat
            eeg_trim = eeg_concat[:, CHANNELS]
        else:
            env_trim = env_concat[:-k]
            eeg_trim = eeg_concat[k:, :][:, CHANNELS]

        r_all = pearson_r_vectorized(env_trim, eeg_trim)
        corr_matrix[s_idx, :, off_idx] = r_all
        print(f'  offset={offset_ms:>4} ms  mean_r={np.nanmean(r_all):.4f}')

    # Continuous xcorr peak per channel
    for c_idx, ch in enumerate(CHANNELS):
        xcorr_peaks[s_idx, c_idx] = compute_xcorr_peak_ms(
            env_concat, eeg_concat[:, ch])

    best_off_idx = int(np.nanargmax(np.nanmean(corr_matrix[s_idx], axis=0)))
    print(f'  → best offset for {subject} (avg over channels): '
          f'{OFFSET_VALUES_MS[best_off_idx]} ms\n')


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation
# ════════════════════════════════════════════════════════════════════════════════

corr_avg_subj = np.nanmean(corr_matrix, axis=0)     # (n_ch, n_off)
corr_avg_ch   = np.nanmean(corr_matrix, axis=1)     # (n_subj, n_off)
corr_grand    = np.nanmean(corr_matrix, axis=(0, 1))  # (n_off,)
corr_grand_sd = np.nanstd(
    corr_matrix.reshape(n_subj * n_ch, n_off), axis=0)  # (n_off,) SD over subj × ch

xcorr_avg_subj = np.nanmean(xcorr_peaks, axis=0)   # avg over subjects, per channel
xcorr_avg_ch   = np.nanmean(xcorr_peaks, axis=1)   # avg over channels, per subject

off_col_names = [f'r_{o}ms' for o in OFFSET_VALUES_MS]

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# CSV 1 — per subject × channel
# ════════════════════════════════════════════════════════════════════════════════

path_sc = OUT_DIR / 'offset_correlations_per_subject_channel.csv'
with open(path_sc, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['subject', 'channel'] + off_col_names + ['best_offset_ms', 'xcorr_peak_ms'])
    for s_idx, subject in enumerate(SUBJECTS):
        for c_idx, ch in enumerate(CHANNELS):
            r_row    = corr_matrix[s_idx, c_idx, :].tolist()
            best_off = OFFSET_VALUES_MS[int(np.nanargmax(corr_matrix[s_idx, c_idx, :]))]
            xpeak    = float(xcorr_peaks[s_idx, c_idx])
            writer.writerow([subject, ch] + r_row + [best_off, xpeak])

print(f'Saved: {path_sc.name}')


# ════════════════════════════════════════════════════════════════════════════════
# CSV 2 — averaged over subjects (one row per channel)
# ════════════════════════════════════════════════════════════════════════════════

path_avg_s = OUT_DIR / 'offset_correlations_avg_over_subjects.csv'
with open(path_avg_s, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['channel'] + off_col_names + ['best_offset_ms', 'xcorr_peak_ms_avg'])
    for c_idx, ch in enumerate(CHANNELS):
        r_row    = corr_avg_subj[c_idx, :].tolist()
        best_off = OFFSET_VALUES_MS[int(np.nanargmax(corr_avg_subj[c_idx, :]))]
        xpeak    = float(xcorr_avg_subj[c_idx])
        writer.writerow([ch] + r_row + [best_off, xpeak])

print(f'Saved: {path_avg_s.name}')


# ════════════════════════════════════════════════════════════════════════════════
# CSV 3 — averaged over channels (one row per subject)
# ════════════════════════════════════════════════════════════════════════════════

path_avg_c = OUT_DIR / 'offset_correlations_avg_over_channels.csv'
with open(path_avg_c, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['subject'] + off_col_names + ['best_offset_ms', 'xcorr_peak_ms_avg'])
    for s_idx, subject in enumerate(SUBJECTS):
        r_row    = corr_avg_ch[s_idx, :].tolist()
        best_off = OFFSET_VALUES_MS[int(np.nanargmax(corr_avg_ch[s_idx, :]))]
        xpeak    = float(xcorr_avg_ch[s_idx])
        writer.writerow([subject] + r_row + [best_off, xpeak])

print(f'Saved: {path_avg_c.name}')


# ════════════════════════════════════════════════════════════════════════════════
# Bar chart — grand-average correlation per offset
# ════════════════════════════════════════════════════════════════════════════════

best_idx  = int(np.nanargmax(corr_grand))
best_off  = OFFSET_VALUES_MS[best_idx]
x_pos     = np.arange(n_off)
bar_colors = ['tomato' if i == best_idx else 'steelblue' for i in range(n_off)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x_pos, corr_grand, yerr=corr_grand_sd, capsize=4,
       color=bar_colors, width=0.6,
       error_kw=dict(elinewidth=1.2, ecolor='#444444'))
ax.axvline(best_idx, color='darkred', lw=1.5, linestyle='--',
           label=f'best offset = {best_off} ms  (r = {corr_grand[best_idx]:.4f})')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{o} ms' for o in OFFSET_VALUES_MS])
ax.set_xlabel('Stimulus offset (ms)', fontsize=12)
ax.set_ylabel('Mean Pearson r  (envelope ↔ EEG)', fontsize=12)
ax.set_title(
    f'Offset sweep  |  {n_subj} subjects × {n_ch} channels\n'
    'Error bars = ±1 SD across subjects × channels',
    fontsize=11)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

chart_path = OUT_DIR / 'offset_sweep_bar_chart.png'
fig.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {chart_path.name}')


# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════

print(f'\n── Results ─────────────────────────────────────────────────────')
print(f'  Grand-average correlation per offset:')
for off, r, sd in zip(OFFSET_VALUES_MS, corr_grand, corr_grand_sd):
    marker = ' ← best' if off == best_off else ''
    print(f'    {off:>4} ms   r = {r:.4f}  (±{sd:.4f}){marker}')
print(f'\n  Outputs in: {OUT_DIR}')
