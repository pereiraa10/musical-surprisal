"""viz_smoke_test.py — methods-figure generator for dataset.py's smoke test.

Turns the PreparedSubject/TRFDataset objects the smoke test already builds
into a set of methodology-ready PNGs (raw stimulus -> envelope -> onsets ->
EEG preprocessing -> alignment/z-scoring -> windowing/tensor shapes), for
direct use in a paper's methods section.

Read-only with respect to the pipeline: every array plotted here either comes
straight out of utils.py/dataset.py's normal return values, or (for a few
intermediate stages nothing currently returns, e.g. "EEG after LPF but before
downsampling") via the opt-in `capture=` callbacks added to
utils.preprocess_eeg_trials / utils.align_stimulus_and_idyom /
utils.compute_envelope_from_audio. None of that changes what the pipeline
computes -- see those functions' docstrings.

Entry point: run(). Called from dataset.py's __main__ when --visualize is
active (the default; pass --no-visualize to skip). Builds its own
PreparedSubject instances (with capture hooks wired in) independently of
whatever dataset.py's own smoke-test assertions built, so this module can
never affect those assertions' results.

See the USER-EDITABLE CONFIG block just below for the subject/trial/channel
and clip-window constants every figure shares.
"""

from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe on headless machines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

import mne

import utils
from config import load_config
from dataset import PreparedSubject

FIGURES_ROOT = Path(__file__).resolve().parent / 'figures' / 'smoke_test'
SAVE_KW = dict(dpi=150, bbox_inches='tight')

# ═══════════════════════════════════════════════════════════════════════════
# USER-EDITABLE CONFIG — change these to look at a different subject / trial /
# channel or a different representative clip, without touching any plotting
# code below.
# ═══════════════════════════════════════════════════════════════════════════

# Which subject / trial / channel every figure is drawn from. Deliberately
# NOT (0, 0, 0): always showing "the first of everything" is an easy way to
# hide bugs that only show up past the first trial or first channel. Each is
# clamped (with a printed warning) if it's out of range for a given config,
# e.g. config.yaml today only has one subject uncommented.
SUBJECT_INDEX = 0
TRIAL_INDEX = 5
CHANNEL_INDEX = 10

# Shared "representative clip" used by every figure that zooms into a short
# stretch of a longer signal (envelope / onsets / resampling / surprisal
# figures). CLIP_START_SEC is deliberately not 0 and not the trial's end: a
# stimulus with a non-musical lead-in before the real content starts (e.g.
# OpenMIIR's cued melodies open with a few seconds of metronome clicks, not
# the tune) would otherwise show that lead-in instead of the stimulus content
# these figures are meant to illustrate. If a trial is too short for this
# window (OpenMIIR's full stimuli are only ~9-16s), _resolve_clip() falls
# back to a centered clip automatically and prints a note.
CLIP_START_SEC = 5.0
CLIP_DURATION_SEC = 2.0

# Windowed-mode figures (16-19): match TRF_conv.py's actual model
# hyperparameters (see TRF_conv.py's WINDOW_SEC / HOP_SEC / BATCH_SIZE)
# instead of an arbitrary toy window size, so these figures show what the
# model actually trains on.
WINDOW_SEC = 7.0
HOP_SEC = 6.0
BATCH_SIZE = 64
N_EXAMPLE_WINDOWS = 4   # how many consecutive windows figure 17 draws

# Figure 6's cross-correlation peak-search/plot window, in ms (EEG relative to
# envelope). Mirrors utils._XCORR_ERP_LAG_MIN_MS/MAX_MS (that's the value
# actually used by the capture hook) -- kept here too so both live in one
# editable place.
XCORR_LAG_MIN_MS = utils._XCORR_ERP_LAG_MIN_MS
XCORR_LAG_MAX_MS = utils._XCORR_ERP_LAG_MAX_MS


def _save(fig, out_dir, filename):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def _try(ctx, name, fn):
    try:
        fn(ctx)
        ctx.produced.append(name)
        print(f"  [viz] {ctx.config_name}/{ctx.subject}: wrote {name}")
    except Exception as e:
        ctx.errors[name] = f"{type(e).__name__}: {e}"
        print(f"  [viz] {ctx.config_name}/{ctx.subject}: FAILED {name} ({type(e).__name__}: {e})")


# ═══════════════════════════════════════════════════════════════════════════
# Context construction
# ═══════════════════════════════════════════════════════════════════════════

class _Capture:
    """Collects utils.py capture-hook payloads for one PreparedSubject build.

    Keeps full per-stage arrays only for `target_trial` (the figures' chosen
    representative trial), but keeps lightweight per-trial scalars
    (length-mismatch diagnostics) for every trial, since the cross-trial
    summary figures need those for the whole subject.
    """

    def __init__(self, target_trial=0):
        self.target_trial = target_trial
        self.meta = {}
        self.stages = {}          # stage -> array, target_trial only
        self.alignment_all = []   # [(trial_idx, payload), ...] every trial
        self.xcorr = None         # payload dict, target_trial only

    def preprocess(self, trial_idx, stage, payload):
        if trial_idx is None:
            self.meta = payload
        elif trial_idx == self.target_trial:
            self.stages[stage] = payload

    def align(self, trial_idx, stage, payload):
        if stage == 'alignment':
            self.alignment_all.append((trial_idx, payload))
        elif stage == 'xcorr' and trial_idx == self.target_trial:
            self.xcorr = payload


def _resolve_clip(duration_s, clip_start_s=CLIP_START_SEC, clip_dur_s=CLIP_DURATION_SEC):
    """Pick the (start_s, dur_s) clip every zoomed-in figure shares, for a
    representative trial of length `duration_s` seconds. Falls back to a
    centered clip (shortened if necessary) when the trial is too short for
    the requested CLIP_START_SEC/CLIP_DURATION_SEC."""
    if clip_start_s + clip_dur_s > duration_s:
        eff_dur = min(clip_dur_s, duration_s)
        start = max(0.0, duration_s / 2 - eff_dur / 2)
        print(f"  [viz] requested clip {clip_start_s:.1f}-{clip_start_s + clip_dur_s:.1f}s "
              f"doesn't fit a {duration_s:.1f}s trial; using {start:.1f}-{start + eff_dur:.1f}s instead")
        return start, eff_dur
    return clip_start_s, clip_dur_s


def _build_context(config_name, expected_source_type):
    """Load `config_name`, build a PreparedSubject (with capture hooks) for
    the configured representative subject, and bundle everything the figure
    functions need. Returns None (with a printed reason) if this config's
    data isn't available in this environment -- callers must handle that
    gracefully."""
    config_path = Path(__file__).resolve().parent / config_name
    if not config_path.exists():
        print(f"  [viz] skip {config_name}: file not found")
        return None

    config = load_config(path=config_path)
    if config.stimulus_source_type != expected_source_type:
        print(f"  [viz] skip {config_name}: stimulus_source_type="
              f"{config.stimulus_source_type!r}, expected {expected_source_type!r}")
        return None

    subj_idx = SUBJECT_INDEX
    if subj_idx >= len(config.subjects):
        print(f"  [viz] {config_name}: SUBJECT_INDEX={SUBJECT_INDEX} out of range "
              f"({len(config.subjects)} configured) — using {len(config.subjects) - 1}")
        subj_idx = len(config.subjects) - 1
    subject = config.subjects[subj_idx]

    eeg_path = config.paths.eeg_dir / config.eeg_filename_pattern.format(subject=subject)
    if not eeg_path.exists():
        print(f"  [viz] skip {config_name} ({subject}): EEG file not found ({eeg_path})")
        return None

    stim_paths_for_subject = None
    if config.stimulus_source_type == 'audio_files':
        stim_paths_for_subject = config.trial_to_stimulus.get(subject)
        first_ok = (stim_paths_for_subject and stim_paths_for_subject[0] is not None
                    and Path(stim_paths_for_subject[0]).exists())
        if not first_ok:
            print(f"  [viz] skip {config_name} ({subject}): trial_to_stimulus has no "
                  "usable path for the first trial (placeholder/unfilled mapping)")
            return None

    try:
        eeg_data = utils.load_subject_raw_eeg(eeg_path, subject, trial_to_stimulus=stim_paths_for_subject)

        n_trials_avail = len(eeg_data['trials'])
        trial_idx = TRIAL_INDEX
        if trial_idx >= n_trials_avail:
            print(f"  [viz] {config_name} ({subject}): TRIAL_INDEX={TRIAL_INDEX} out of range "
                  f"({n_trials_avail} trials) — using {n_trials_avail - 1}")
            trial_idx = n_trials_avail - 1

        n_channels_avail = eeg_data['trials'][0].shape[1]
        channel_idx = CHANNEL_INDEX
        if channel_idx >= n_channels_avail:
            print(f"  [viz] {config_name} ({subject}): CHANNEL_INDEX={CHANNEL_INDEX} out of range "
                  f"({n_channels_avail} channels) — using {n_channels_avail - 1}")
            channel_idx = n_channels_avail - 1

        capture = _Capture(target_trial=trial_idx)
        prepared = PreparedSubject(
            subject, eeg_data, config, debug=True,
            preprocess_capture=capture.preprocess, align_capture=capture.align)

        ds_acoustic = prepared.to_dataset('acoustic', window_samples=None)
        ds_full = prepared.to_dataset('acoustic_and_surprisal', window_samples=None)

        ws = int(round(WINDOW_SEC * config.sfreq))
        hs = int(round(HOP_SEC * config.sfreq))
        ds_win = prepared.to_dataset('acoustic_and_surprisal', window_samples=ws, hop_samples=hs)
    except Exception as e:
        print(f"  [viz] skip {config_name} ({subject}): failed to build pipeline "
              f"({type(e).__name__}: {e})")
        return None

    trial_duration_s = ds_acoustic.trial_lengths[trial_idx] / config.sfreq
    clip_start_s, clip_dur_s = _resolve_clip(trial_duration_s)

    out_dir = FIGURES_ROOT / Path(config_name).stem / subject
    return SimpleNamespace(
        config_name=config_name, config=config, subject=subject, trial_idx=trial_idx,
        channel_idx=channel_idx, clip_start_s=clip_start_s, clip_dur_s=clip_dur_s,
        eeg_data=eeg_data, capture=capture, prepared=prepared,
        ds_acoustic=ds_acoustic, ds_full=ds_full, ds_win=ds_win,
        out_dir=out_dir, produced=[], errors={},
    )


def _align_payload(ctx, trial_idx=None):
    ti = ctx.trial_idx if trial_idx is None else trial_idx
    for idx, payload in ctx.capture.alignment_all:
        if idx == ti:
            return payload
    raise KeyError(f"no alignment capture for trial {ti}")


def _channel_name(ctx, ch=None):
    ch = ctx.channel_idx if ch is None else ch
    names = ctx.prepared.channel_names
    return names[ch] if ch < len(names) else str(ch)


def _load_native_stimulus(ctx):
    """Raw-rate waveform + envelope for ctx's representative trial, sourced
    the way that's appropriate for this config's stimulus_source_type (see
    module docstring's mat-vs-audio_files fork)."""
    config = ctx.config
    ti = ctx.trial_idx
    align_payload = _align_payload(ctx, ti)

    if config.stimulus_source_type == 'mat':
        marker = int(ctx.prepared.events['event'][ti])
        song_id = utils.song_id_for_marker(marker, config.trial_to_song_id)
        wav_path = config.paths.wav_dir / f"{song_id}.wav"
        audio, audio_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        env_native = align_payload['env_raw']
        env_native_fs = align_payload['stim_fs']
    else:
        stim_path = config.trial_to_stimulus[ctx.subject][ti]
        stash = {}

        def _cb(audio_, sr_, env_native_):
            stash['audio'], stash['sr'], stash['env_native'] = audio_, sr_, env_native_

        utils.compute_envelope_from_audio(stim_path, config.sfreq, capture=_cb)
        audio, audio_sr = stash['audio'], stash['sr']
        env_native = stash['env_native']
        env_native_fs = stash['sr']
        wav_path = stim_path

    return dict(
        audio=audio, audio_sr=audio_sr, wav_path=wav_path,
        env_native=env_native, env_native_fs=env_native_fs,
        env_resampled=align_payload['env_resampled_full'], env_resampled_fs=config.sfreq,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STIMULUS / ENVELOPE / ONSETS (figures 2-7; figure 1 removed -- redundant
# with figure 2, which shows the same waveform with the envelope overlaid)
# ═══════════════════════════════════════════════════════════════════════════

def _fig02_envelope_over_waveform(ctx):
    stim = _load_native_stimulus(ctx)
    start, dur = ctx.clip_start_s, ctx.clip_dur_s

    a0, a1 = int(start * stim['audio_sr']), int((start + dur) * stim['audio_sr'])
    e0, e1 = int(start * stim['env_native_fs']), int((start + dur) * stim['env_native_fs'])
    audio_clip = stim['audio'][a0:a1]
    env_clip = stim['env_native'][e0:e1]

    audio_n = audio_clip / (np.max(np.abs(audio_clip)) + 1e-12)
    env_n = env_clip / (np.max(env_clip) + 1e-12)
    t_audio = np.arange(len(audio_n)) / stim['audio_sr'] + start
    t_env = np.arange(len(env_n)) / stim['env_native_fs'] + start

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_audio, audio_n, lw=0.5, color='0.7', label='raw waveform (normalized)')
    ax.plot(t_env, env_n, lw=1.6, color='C1',
            label=f"envelope @ {stim['env_native_fs']} Hz (normalized)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized amplitude')
    source = ('Hilbert envelope' if ctx.config.stimulus_source_type == 'audio_files'
              else 'precomputed dataStim.mat envelope')
    ax.set_title(f"Envelope over raw waveform ({source}), {dur:.0f}s clip — "
                 f"{ctx.subject}, trial {ctx.trial_idx}")
    ax.legend(loc='upper right', fontsize=8)
    _save(fig, ctx.out_dir, '02_envelope_over_waveform.png')


def _has_mat_provided_envelope(ctx):
    return ctx.config.stimulus_source_type == 'mat'


def _fig02b_envelope_provided_vs_computed(ctx):
    """Sanity-check liberi_dataset's precomputed dataStim.mat envelope
    against what utils.compute_envelope_from_audio (the Hilbert-based method
    used for audio_files-mode datasets) computes from the same raw stimulus
    audio. These come from two different pipelines -- dataStim.mat's envelope
    was computed upstream by the dataset's original authors, not by this
    codebase -- so exact numerical agreement isn't expected, but the two
    should track the same dynamics; that agreement is what this figure
    checks."""
    stim = _load_native_stimulus(ctx)
    computed = utils.compute_envelope_from_audio(stim['wav_path'], ctx.config.sfreq)

    start, dur = ctx.clip_start_s, ctx.clip_dur_s
    sfreq = ctx.config.sfreq
    n0 = int(start * sfreq)
    n1 = min(int((start + dur) * sfreq), len(stim['env_resampled']), len(computed))

    provided = stim['env_resampled'][n0:n1]
    computed_clip = computed[n0:n1]
    provided_n = provided / (np.max(np.abs(provided)) + 1e-12)
    computed_n = computed_clip / (np.max(np.abs(computed_clip)) + 1e-12)
    t = np.arange(len(provided_n)) / sfreq + start

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, provided_n, color='C0', lw=1.3, label='provided (dataStim.mat)')
    ax.plot(t, computed_n, color='C1', lw=1.1, ls='--',
            label='computed (Hilbert, utils.compute_envelope_from_audio)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized envelope')
    ax.set_title(f"Provided vs. computed envelope, same stimulus, {dur:.0f}s clip — "
                 f"{ctx.subject}, trial {ctx.trial_idx}\n"
                 "(different pipelines — not expected to match exactly, just to track "
                 "the same dynamics)")
    ax.legend(loc='best', fontsize=8)
    _save(fig, ctx.out_dir, '02b_envelope_provided_vs_computed.png')


def _fig03_envelope_resample_zoom(ctx):
    stim = _load_native_stimulus(ctx)
    start, dur = ctx.clip_start_s, ctx.clip_dur_s
    n0_native, n1_native = int(start * stim['env_native_fs']), int((start + dur) * stim['env_native_fs'])
    n0_resamp, n1_resamp = int(start * stim['env_resampled_fs']), int((start + dur) * stim['env_resampled_fs'])
    env_native = stim['env_native'][n0_native:n1_native]
    env_resamp = stim['env_resampled'][n0_resamp:n1_resamp]
    t_native = np.arange(len(env_native)) / stim['env_native_fs'] + start
    t_resamp = np.arange(len(env_resamp)) / stim['env_resampled_fs'] + start

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_native, env_native, 'o-', ms=3, lw=0.8, color='0.5',
            label=f"before resample_poly ({stim['env_native_fs']} Hz)")
    ax.plot(t_resamp, env_resamp, 'x-', ms=4, lw=1.2, color='C0',
            label=f"after resample_poly ({stim['env_resampled_fs']} Hz)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Envelope amplitude')
    ax.set_title(f"Envelope resampling to target sfreq, {dur:.0f}s clip — "
                 f"{ctx.subject}, trial {ctx.trial_idx}")
    ax.legend(loc='upper right', fontsize=8)
    _save(fig, ctx.out_dir, '03_envelope_resample_zoom.png')


def _fig04_envelope_eeg_length_diff(ctx):
    trials = sorted(ctx.capture.alignment_all, key=lambda p: p[0])
    diffs = np.array([p['diff'] for _, p in trials])
    sfreq = ctx.config.sfreq
    threshold = 4 * sfreq
    max_abs_ms = np.max(np.abs(diffs)) * 1000.0 / sfreq if len(diffs) else 0.0

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(np.arange(len(diffs)), diffs, color='C0')
    ax.axhline(threshold, color='r', ls='--', lw=1)
    ax.axhline(-threshold, color='r', ls='--', lw=1)
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_xlabel('Trial index')
    ax.set_ylabel(f'len(resampled stimulus) − len(EEG trial)  (samples @ {sfreq} Hz)')
    ax_ms = ax.secondary_yaxis(
        'right', functions=(lambda s: s * 1000.0 / sfreq, lambda ms: ms * sfreq / 1000.0))
    ax_ms.set_ylabel('ms')
    ax.set_title(
        "How much the resampled stimulus envelope over/under-runs its matched EEG "
        f"trial, before align_stimulus_and_idyom trims to the shorter of the two — "
        f"{ctx.subject} ({ctx.config_name}, {len(diffs)} trials)\n"
        f"largest mismatch: {max_abs_ms:.0f} ms  |  dashed line = the function's own "
        f"±4×sfreq warning threshold ({threshold} smp = {threshold * 1000 / sfreq:.0f} ms)",
        fontsize=10)
    fig.text(
        0.5, -0.06,
        "Positive = the stimulus ran longer than its matched EEG recording for that "
        "trial; negative = the reverse. Small values (a handful of ms, from resampling "
        "rounding) are normal and are silently trimmed away by align_stimulus_and_idyom. "
        "Only a bar approaching the dashed threshold would indicate a real alignment "
        "problem (wrong stimulus file, mis-segmented trial, etc.) worth investigating.",
        ha='center', fontsize=8, wrap=True)
    _save(fig, ctx.out_dir, '04_envelope_eeg_length_diff.png')


def _fig05_envelope_onsets(ctx):
    ti = ctx.trial_idx
    env = ctx.prepared.events['envelope'][ti].x
    onsets = ctx.prepared.events['onsets'][ti].x
    sfreq = ctx.config.sfreq
    start, dur = ctx.clip_start_s, ctx.clip_dur_s
    n0, n1 = int(start * sfreq), int((start + dur) * sfreq)
    env_clip = env[n0:n1]
    onsets_clip = onsets[n0:min(n1, len(onsets))]
    t = np.arange(len(env_clip)) / sfreq + start

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, env_clip, color='C0', lw=1.0)
    axes[0].set_ylabel('Envelope')
    axes[0].set_title(f"Envelope and derived onsets, {dur:.0f}s clip — {ctx.subject}, trial {ti}")
    axes[1].plot(t[:len(onsets_clip)], onsets_clip, color='C2', lw=1.0)
    axes[1].set_ylabel('Onsets\n(diff(env).clip(0))')
    axes[1].set_xlabel('Time (s)')
    _save(fig, ctx.out_dir, '05_envelope_onsets.png')


def _fig06_xcorr_lag(ctx):
    payload = ctx.capture.xcorr
    if payload is None:
        raise RuntimeError("no xcorr capture for the representative trial")
    xcorr_erp = payload['xcorr_erp']
    zero_lag_idx = payload['zero_lag_idx']
    erp_lo = payload['erp_lo']
    lag_smp = payload['lag_smp']
    sfreq = ctx.config.sfreq
    lag_axis_ms = (np.arange(len(xcorr_erp)) + erp_lo - zero_lag_idx) * 1000.0 / sfreq

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(lag_axis_ms, xcorr_erp, color='0.4', lw=1.0)
    ax.axvline(lag_smp * 1000.0 / sfreq, color='r', ls='--',
               label=f"peak lag = {lag_smp} smp ({payload['lag_ms']:.1f} ms)")
    ax.axvline(0, color='0.6', lw=0.8, ls=':')
    ax.set_xlabel('Lag (ms)  [EEG relative to envelope]')
    ax.set_ylabel('Cross-correlation')
    ax.set_title(
        f"EEG/envelope cross-correlation, restricted to a plausible auditory-ERP "
        f"window ([{XCORR_LAG_MIN_MS}, {XCORR_LAG_MAX_MS}] ms) — {ctx.subject}, "
        f"trial {ctx.trial_idx}\n"
        "(searching the full ±trial-duration lag range instead lets a long, "
        "rhythmically periodic musical trial's own beat-rate autocorrelation "
        "dominate the argmax — see utils.align_stimulus_and_idyom)", fontsize=10)
    ax.legend(loc='best', fontsize=8)
    _save(fig, ctx.out_dir, '06_xcorr_lag.png')


_SURPRISAL_KEYS = ['pitch_surprisal', 'pitch_entropy', 'onset_surprisal', 'onset_entropy']
_CLIPPED_SURPRISAL_KEYS = {'pitch_surprisal', 'onset_surprisal'}  # see utils._StimulusLibrary


def _has_surprisal_features(ctx):
    feature_keys = ctx.config.feature_sets.get('acoustic_and_surprisal', [])
    return all(k in feature_keys for k in _SURPRISAL_KEYS)


def _fig07_idyom_surprisal_placement(ctx):
    surprisal_keys = _SURPRISAL_KEYS
    ti = ctx.trial_idx
    events = ctx.prepared.events
    env = events['envelope'][ti].x
    sfreq = ctx.config.sfreq
    start, dur = ctx.clip_start_s, ctx.clip_dur_s
    n0, n1 = int(start * sfreq), int((start + dur) * sfreq)
    env_clip = env[n0:n1]
    t = np.arange(len(env_clip)) / sfreq + start
    ic_clip = ctx.config.ic_clip

    fig, axes = plt.subplots(len(surprisal_keys) + 1, 1, figsize=(10, 9.5), sharex=True)
    axes[0].plot(t, env_clip, color='C0', lw=1.0)
    axes[0].set_ylabel('Envelope')
    axes[0].set_title(f"IDyOM surprisal/entropy placement onto the EEG time grid, "
                       f"{dur:.0f}s clip — {ctx.subject}, trial {ti}")
    for ax, key in zip(axes[1:], surprisal_keys):
        vals = events[key][ti].x[n0:n1]
        ax.stem(t[:len(vals)], vals, basefmt=' ', markerfmt='C1.', linefmt='C1-')
        label = key.replace('_', '\n')
        if key in _CLIPPED_SURPRISAL_KEYS:
            ax.axhline(ic_clip, color='r', ls='--', lw=0.8,
                       label=f'ic_clip ceiling ({ic_clip:g} bits)')
            ax.legend(loc='upper right', fontsize=6)
            label += f"\n(clipped\n0–{ic_clip:g})"
        else:
            label += "\n(NOT\nclipped)"
        ax.set_ylabel(label, fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.text(
        0.5, -0.02,
        "pitch_surprisal / onset_surprisal are clipped to [0, ic_clip] bits at the "
        "IDyOM source (utils._StimulusLibrary.__init__); pitch_entropy / onset_entropy "
        "are NOT clipped anywhere in this pipeline, so an occasional large raw value "
        "there reaches zscore_trials (and the model) unmodified.",
        ha='center', fontsize=8, wrap=True)
    _save(fig, ctx.out_dir, '07_idyom_surprisal_placement.png')


# ═══════════════════════════════════════════════════════════════════════════
# EEG PREPROCESSING (figures 8-12)
# ═══════════════════════════════════════════════════════════════════════════

def _fig08_eeg_raw_padding(ctx):
    meta = ctx.capture.meta
    ch = ctx.channel_idx
    raw_full = ctx.capture.stages['raw'][:, ch]
    orig_fs = meta['orig_fs']
    pad = meta['pad_start_orig']
    show_dur = 2.0
    n_show = int(show_dur * orig_fs)
    raw = raw_full[:n_show]
    t = np.arange(len(raw)) / orig_fs

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, raw, color='0.3', lw=0.7)
    ax.axvspan(0, min(pad, n_show) / orig_fs, color='r', alpha=0.15,
               label=f"padding ({pad} samples @ {orig_fs} Hz = {pad / orig_fs:.2f} s)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f"Raw EEG, channel {_channel_name(ctx)}, original fs, first "
                 f"{show_dur:.0f}s — {ctx.subject}, trial {ctx.trial_idx}")
    ax.legend(loc='best', fontsize=8)
    _save(fig, ctx.out_dir, '08_eeg_raw_padding.png')


def _fig09_eeg_preprocessing_stages(ctx):
    meta = ctx.capture.meta
    stages = ctx.capture.stages
    orig_fs, target_fs = meta['orig_fs'], meta['target_fs']
    pad_orig, pad_tgt = meta['pad_start_orig'], meta['pad_start_target']
    ch = ctx.channel_idx

    raw = stages['raw'][:, ch]
    lpf = stages['lpf'][:, ch]
    down = stages['downsampled'][:, ch]
    hpf = stages['hpf'][:, ch]
    final = stages['final'][:, ch]

    t_orig = np.arange(len(raw)) / orig_fs
    t_tgt = np.arange(len(down)) / target_fs
    t_final = np.arange(len(final)) / target_fs

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
    axes[0].plot(t_orig, raw, color='0.6', lw=0.5, label='raw')
    axes[0].axvspan(0, pad_orig / orig_fs, color='r', alpha=0.12)
    axes[0].set_title(f"1. Raw @ {orig_fs} Hz (padding shaded)")

    axes[1].plot(t_orig, lpf, color='C0', lw=0.6)
    axes[1].axvspan(0, pad_orig / orig_fs, color='r', alpha=0.12)
    axes[1].set_title(f"2. After LPF ({meta['lpf_hz']} Hz) @ {orig_fs} Hz")

    axes[2].plot(t_tgt, down, color='0.5', lw=0.6, alpha=0.6, label='downsampled')
    axes[2].plot(t_tgt, hpf, color='C1', lw=0.8, label=f"after HPF ({meta['hpf_hz']} Hz)")
    axes[2].axvspan(0, pad_tgt / target_fs, color='r', alpha=0.12)
    axes[2].set_title(f"3. After downsample -> HPF @ {target_fs} Hz (padding shaded)")
    axes[2].legend(loc='best', fontsize=7)

    axes[3].plot(t_final, final, color='C2', lw=0.6)
    axes[3].set_title(f"4. Final, after padding strip @ {target_fs} Hz")

    for ax in axes:
        ax.set_ylabel('µV')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f"EEG preprocessing stages, channel {_channel_name(ctx)} — "
                 f"{ctx.subject}, trial {ctx.trial_idx}", y=1.02)
    fig.tight_layout(h_pad=2.5)
    _save(fig, ctx.out_dir, '09_eeg_preprocessing_stages.png')


def _fig10_eeg_psd_filters(ctx):
    meta = ctx.capture.meta
    stages = ctx.capture.stages
    ch = ctx.channel_idx
    raw = stages['raw'][:, ch]
    hpf_final = stages['final'][:, ch]
    orig_fs, target_fs = meta['orig_fs'], meta['target_fs']

    f_raw, psd_raw = welch(raw, fs=orig_fs, nperseg=min(2048, len(raw)))
    f_lpf, psd_lpf = welch(stages['lpf'][:, ch], fs=orig_fs, nperseg=min(2048, len(stages['lpf'])))
    f_final, psd_final = welch(hpf_final, fs=target_fs, nperseg=min(2048, len(hpf_final)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f_raw, psd_raw, color='0.6', lw=0.8, label='raw')
    ax.semilogy(f_lpf, psd_lpf, color='C0', lw=0.8, label='after LPF')
    ax.semilogy(f_final, psd_final, color='C2', lw=1.0, label='final (after HPF)')
    ax.axvline(meta['hpf_hz'], color='r', ls='--', lw=1, label=f"low_frequency = {meta['hpf_hz']} Hz")
    ax.axvline(meta['lpf_hz'], color='m', ls='--', lw=1, label=f"high_frequency = {meta['lpf_hz']} Hz")
    ax.set_xlim(0, max(30, meta['lpf_hz'] * 2))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (µV²/Hz)')
    ax.set_title(f"Welch PSD across preprocessing stages, channel {_channel_name(ctx)} — "
                 f"{ctx.subject}, trial {ctx.trial_idx}")
    ax.legend(loc='best', fontsize=8)
    _save(fig, ctx.out_dir, '10_eeg_psd_filters.png')


def _fig11_raw_concat_markers(ctx):
    raw = ctx.prepared.raw
    sfreq = ctx.config.sfreq
    ch = ctx.channel_idx
    i_start = np.asarray(ctx.prepared.events['i_start'].x)
    n_show_trials = min(4, len(i_start))
    end_sample = int(i_start[n_show_trials]) if n_show_trials < len(i_start) else raw.n_times
    end_sample = min(end_sample, raw.n_times)

    trace = raw.get_data(picks=[ch])[0, :end_sample]
    t = np.arange(end_sample) / sfreq

    # Per-trial zero-phase filtering (LPF -> downsample -> HPF, each run
    # independently per trial) has no real data beyond a trial's own edges to
    # pad against. A slow drift right at a trial boundary can turn into a
    # filtfilt edge/boundary transient rather than a genuine neural event --
    # confirmed on this dataset: the raw, unfiltered channel is smooth going
    # into the last ~10 samples of several trials (std over the last 1s is an
    # order of magnitude below the trial's overall std), while the post-HPF
    # signal ramps to several times the trial's typical amplitude in exactly
    # those same last few samples. Liberi's trials have 1s of leading padding
    # for the filters to settle into at the START but nothing protecting the
    # END, which is exactly where this shows up. Use a robust percentile-
    # based ylim so those edge samples don't compress the rest of the trace,
    # and shade the last 0.5s of each shown trial rather than silently hiding it.
    lo, hi = np.percentile(trace, [0.5, 99.5])
    pad_y = 0.15 * (hi - lo)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, trace, color='0.3', lw=0.4)
    for i in range(n_show_trials):
        onset_s = i_start[i] / sfreq
        if onset_s <= t[-1]:
            ax.axvline(onset_s, color='r', ls='--', lw=1)
            ax.annotate(f"trial {i + 1} onset", (onset_s, hi),
                        fontsize=7, color='r', rotation=90, va='top')
        boundary_s = (i_start[i + 1] / sfreq) if i + 1 < len(i_start) else t[-1]
        edge_start = max(onset_s, boundary_s - 0.5)
        if edge_start < t[-1]:
            ax.axvspan(edge_start, min(boundary_s, t[-1]), color='orange', alpha=0.15)
    ax.set_ylim(lo - pad_y, hi + pad_y)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'µV (channel {_channel_name(ctx)})')
    ax.set_title(f"Concatenated MNE Raw with STI trial markers (first "
                 f"{n_show_trials} trials) — {ctx.subject}\n"
                 "(shaded = last 0.5s of each trial, where per-trial zero-phase "
                 "filtering is most likely to show an edge artifact)", fontsize=10)
    _save(fig, ctx.out_dir, '11_raw_concat_markers.png')


def _fig12_montage_sensors(ctx):
    fig = mne.viz.plot_sensors(ctx.prepared.raw.info, show_names=False, show=False)
    fig.suptitle(f"Sensor montage — {ctx.subject} ({ctx.config_name})")
    _save(fig, ctx.out_dir, '12_montage_sensors.png')


# ═══════════════════════════════════════════════════════════════════════════
# ALIGNMENT / Z-SCORING (figures 13-15)
# ═══════════════════════════════════════════════════════════════════════════

def _fig13_alignment_trim(ctx):
    ti = ctx.trial_idx
    ch = ctx.channel_idx
    trial = ctx.ds_acoustic.trials[ti]
    sfreq = ctx.config.sfreq
    t = np.arange(trial['eeg'].shape[0]) / sfreq

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, trial['eeg'][:, ch], color='C0', lw=0.7,
            label=f'EEG channel {_channel_name(ctx)} (z-scored)')
    ax.plot(t, trial['envelope'], color='C1', lw=0.7, alpha=0.8, label='Envelope (z-scored)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-score')
    ax.set_title(f"EEG vs envelope after align_trial trim — {ctx.subject}, trial {ti} "
                 f"(T={trial['eeg'].shape[0]} samples)")
    ax.legend(loc='best', fontsize=8)
    _save(fig, ctx.out_dir, '13_alignment_trim.png')


def _fig14_zscore_distributions(ctx):
    ti = ctx.trial_idx
    before = ctx.prepared._trials_raw[ti]
    after = ctx.ds_full.trials[ti]
    keys = ['eeg'] + list(ctx.ds_full.feature_keys)

    fig, axes = plt.subplots(len(keys), 2, figsize=(9, 2.4 * len(keys)))
    for row, k in enumerate(keys):
        b = before[k].ravel()
        a = after[k].ravel()
        axes[row, 0].hist(b, bins=40, color='0.5')
        axes[row, 0].set_yscale('log')
        axes[row, 0].set_ylabel(k, fontsize=9)
        axes[row, 1].hist(a, bins=40, color='C0')
        axes[row, 1].set_yscale('log')
        if row == 0:
            axes[row, 0].set_title('Before zscore_trials')
            axes[row, 1].set_title('After zscore_trials')
    fig.suptitle(f"Per-trial z-scoring, before vs after (log-scaled counts) — "
                 f"{ctx.subject}, trial {ti}", y=1.01)
    fig.text(
        0.5, -0.01,
        "Log-scaled counts so both the bulk of typical values and rare outliers stay "
        "visible in the same panel. envelope/onsets/surprisal are non-negative before "
        "z-scoring, so most of their z-scored mass sitting below 0 is expected: "
        "z-scoring centers on the trial mean, and a right-skewed, mostly-quiet signal "
        "has a mean above its typical value. A rare large raw outlier also inflates "
        "the trial's std, which compresses the rest of the distribution toward 0 and "
        "makes that outlier look even more extreme in z-score units — an expected "
        "property of z-scoring a heavy-tailed/impulse-like signal, not a bug.",
        ha='center', fontsize=8, wrap=True)
    fig.tight_layout()
    _save(fig, ctx.out_dir, '14_zscore_distributions.png')


def _fig15_feature_stack_trial(ctx):
    ti = ctx.trial_idx
    ch = ctx.channel_idx
    trial = ctx.ds_full.trials[ti]
    sfreq = ctx.config.sfreq
    T = trial['eeg'].shape[0]
    t = np.arange(T) / sfreq
    keys = ['eeg'] + list(ctx.ds_full.feature_keys)

    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 1.6 * len(keys)), sharex=True)
    for ax, k in zip(axes, keys):
        if k == 'eeg':
            ax.plot(t, trial['eeg'][:, ch], color='k', lw=0.7)
            ax.set_ylabel(f'EEG {_channel_name(ctx)}', fontsize=9)
        else:
            ax.plot(t, trial[k], lw=0.7)
            ax.set_ylabel(k.replace('_', '\n'), fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f"Stimulus representation, feature_set='acoustic_and_surprisal' "
                 f"(z-scored) — {ctx.subject}, trial {ti}", y=1.01)
    _save(fig, ctx.out_dir, '15_feature_stack_trial.png')


# ═══════════════════════════════════════════════════════════════════════════
# WINDOWING / TENSOR SHAPES (figures 16-19)
# ═══════════════════════════════════════════════════════════════════════════

def _fig16_window_timeline(ctx):
    ti = ctx.trial_idx
    ds = ctx.ds_win
    sfreq = ctx.config.sfreq
    T = ds.trial_lengths[ti]
    all_idxs = ds.windows_for_trial(ti)
    idxs = all_idxs[:BATCH_SIZE]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.add_patch(mpatches.Rectangle((0, 0), T / sfreq, 1, color='0.85'))
    for j, idx in enumerate(idxs):
        _, start, end = ds._index[idx]
        ax.add_patch(mpatches.Rectangle(
            (start / sfreq, 0.05 + 0.02 * (j % 3)), (end - start) / sfreq, 0.15,
            color=f"C{j % 10}", alpha=0.6))
    ax.set_xlim(0, T / sfreq)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    ax.set_title(
        f"Window extraction timeline (window={WINDOW_SEC:.0f}s/{ds.window_samples} smp, "
        f"hop={HOP_SEC:.0f}s/{ds.hop_samples} smp — matches TRF_conv.py; "
        f"{len(idxs)} of {len(all_idxs)} windows shown, capped at one batch = "
        f"{BATCH_SIZE}) — {ctx.subject}, trial {ti}", fontsize=10)
    _save(fig, ctx.out_dir, '16_window_timeline.png')


def _fig17_window_examples(ctx):
    ti = ctx.trial_idx
    ds = ctx.ds_win
    sfreq = ctx.config.sfreq
    idxs = ds.windows_for_trial(ti)[:N_EXAMPLE_WINDOWS]
    if not idxs:
        raise RuntimeError(
            f"trial {ti} produced no windows at window={ds.window_samples}/"
            f"hop={ds.hop_samples} smp — pick a longer TRIAL_INDEX or shorten "
            "WINDOW_SEC/HOP_SEC.")
    trial_env = ctx.ds_acoustic.trials[ti]['envelope']

    first_start = ds._index[idxs[0]][1]
    last_end = ds._index[idxs[-1]][2]
    t_full = np.arange(first_start, last_end) / sfreq

    fig, axes = plt.subplots(len(idxs) + 1, 1, figsize=(10, 1.4 * (len(idxs) + 1)), sharex=True)
    axes[0].plot(t_full, trial_env[first_start:last_end], color='0.4', lw=0.8)
    axes[0].set_ylabel(f'shown span\n({(last_end - first_start) / sfreq:.0f}s)', fontsize=8)
    for idx, ax in zip(idxs, axes[1:]):
        _, start, end = ds._index[idx]
        t_win = np.arange(start, end) / sfreq
        ax.plot(t_win, trial_env[start:end], color='C0', lw=1.0)
        for b in (start, end):
            axes[0].axvline(b / sfreq, color='r', lw=0.4, alpha=0.5)
        ax.set_ylabel(f"win {idx}", fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f"{len(idxs)} consecutive extracted windows (envelope), "
                 f"window={WINDOW_SEC:.0f}s hop={HOP_SEC:.0f}s (matches TRF_conv.py) — "
                 f"{ctx.subject}, trial {ti}", y=1.01)
    _save(fig, ctx.out_dir, '17_window_examples.png')


def _fig18_shape_flow(ctx):
    ti = ctx.trial_idx
    T = ctx.ds_full.trial_lengths[ti]
    n_ch = ctx.ds_full.n_channels
    n_feat = len(ctx.ds_full.feature_keys)
    ws = ctx.ds_win.window_samples
    batch = min(BATCH_SIZE, len(ctx.ds_win))

    boxes = [
        f"Raw EEG\n{T} samples × {n_ch} channels",
        f"Z-scored EEG\n{T} samples × {n_ch} channels",
        f"getitem()\ninput X: {n_feat} features × {ws} samples/window\n"
        f"target Y: {n_ch} channels × {ws} samples/window",
        f"DataLoader batch\n{batch} windows × ({n_feat} features or {n_ch} channels) "
        f"× {ws} samples",
    ]
    box_w, box_h = 0.21, 0.4
    half_w = box_w / 2
    fig, ax = plt.subplots(figsize=(14, 4.6))
    x_positions = np.linspace(half_w + 0.02, 1 - half_w - 0.02, len(boxes))
    for x, text in zip(x_positions, boxes):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - half_w, 0.55), box_w, box_h, boxstyle="round,pad=0.02",
            facecolor='#dbe9f6', edgecolor='0.3'))
        ax.text(x, 0.75, text, ha='center', va='center', fontsize=8)
    for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate('', xy=(x1 - half_w - 0.01, 0.75), xytext=(x0 + half_w + 0.01, 0.75),
                    arrowprops=dict(arrowstyle='->', color='0.3'))

    caption_text = (
        f"This trial: {T} samples total ({T / ctx.config.sfreq:.0f}s @ {ctx.config.sfreq} Hz). "
        f"One window: {WINDOW_SEC:.0f}s = {ws} samples.\n"
        f"Stimulus features in this feature_set ({n_feat} total): "
        f"{', '.join(ctx.ds_full.feature_keys)}.\n"
        "input X = the stimulus feature tensor for one window/batch (what the model reads) — "
        "\"X\"/\"Y\" here are the actual variable names returned by getitem().\n"
        "target Y = the EEG tensor for that same window/batch (what the model is trained to predict)"
    )
    ax.text(0.5, 0.18, caption_text, ha='center', va='center', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Tensor shape flow — {ctx.subject}, trial {ti} "
                 f"(live values from this smoke test's objects)")
    _save(fig, ctx.out_dir, '18_shape_flow.png')


def _fig19_window_count_sanity(ctx):
    ds = ctx.ds_win
    ws, hs = ds.window_samples, ds.hop_samples
    lengths = np.array(ds.trial_lengths)
    counts = np.array([len(ds.windows_for_trial(ti)) for ti in range(ds.n_trials)])
    closed_form = np.maximum(0, (lengths - ws) // hs + 1)
    all_match = np.array_equal(counts, closed_form)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.scatter(lengths, counts, color='C0', s=45, zorder=3,
               label='actual: len(windows_for_trial(ti))')
    order = np.argsort(lengths)
    ax.plot(lengths[order], closed_form[order], color='r', ls='--', zorder=2,
            label='predicted: (T − ws)//hs + 1')
    ax.set_xlabel('Trial length T (samples)')
    ax.set_ylabel('Number of windows extracted from that trial')
    ax.set_title(
        f"Window-count correctness check — {ctx.subject} (window={WINDOW_SEC:.0f}s/{ws} "
        f"smp, hop={HOP_SEC:.0f}s/{hs} smp, {ds.n_trials} trials)\n"
        f"every point sits exactly on the red line: {'YES' if all_match else 'NO — see errors'}",
        fontsize=10)
    ax.legend(loc='best', fontsize=8)
    fig.text(
        0.5, -0.07,
        "Each dot is one trial: x = how long that trial is, y = how many windows the "
        "code actually extracted from it. The red line is not a diagonal — it's the "
        "step-shaped formula (T−ws)//hs+1, since trials of very different lengths "
        "don't produce proportionally more whole windows once you floor-divide. What "
        "matters is that every blue dot lands exactly on the red line for its trial's "
        "length: that means the sliding-window code extracted precisely as many "
        "windows as the formula predicts, for every trial — not just on average.",
        ha='center', fontsize=8, wrap=True)
    _save(fig, ctx.out_dir, '19_window_count_sanity.png')


# ═══════════════════════════════════════════════════════════════════════════
# DATASET-LEVEL SUMMARY (figures 20-22)
# ═══════════════════════════════════════════════════════════════════════════

def _fig20_trial_duration_distribution(ctx):
    sfreq = ctx.config.sfreq
    durations = np.array(ctx.ds_full.trial_lengths) / sfreq
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(durations, bins=min(15, max(3, len(durations))), color='C0', edgecolor='white')
    ax.set_xlabel('Trial duration (s)')
    ax.set_ylabel('Count')
    ax.set_title(f"Trial-duration distribution — {ctx.subject} "
                 f"({ctx.config_name}, {len(durations)} trials)")
    _save(fig, ctx.out_dir, '20_trial_duration_distribution.png')


def _fig21_envelope_source_comparison(mat_ctx, audio_ctx):
    out_dir = FIGURES_ROOT / 'summary'
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    for ax, ctx, label in zip(
            axes, (mat_ctx, audio_ctx),
            (f"liberi (mat)\n{mat_ctx.subject}, trial {mat_ctx.trial_idx}",
             f"{audio_ctx.config_name} (audio_files)\n{audio_ctx.subject}, trial {audio_ctx.trial_idx}")):
        env = ctx.ds_acoustic.trials[ctx.trial_idx]['envelope']
        t = np.arange(len(env)) / ctx.config.sfreq
        ax.plot(t, env, color='C0', lw=0.7)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Time (s)')
    axes[0].set_ylabel('Envelope (z-scored)')
    fig.suptitle("Envelope sourcing comparison: precomputed (mat) vs Hilbert-from-audio "
                 "(audio_files), both z-scored", y=1.03)
    _save(fig, out_dir, '21_envelope_source_comparison.png')
    print("  [viz] summary: wrote 21_envelope_source_comparison.png")


def _fig22_shapes_table(contexts):
    out_dir = FIGURES_ROOT / 'summary'
    rows = []
    for ctx in contexts:
        ws = int(round(WINDOW_SEC * ctx.config.sfreq))
        hs = int(round(HOP_SEC * ctx.config.sfreq))
        for feature_set, keys in ctx.config.feature_sets.items():
            try:
                ds_win_fs = ctx.prepared.to_dataset(
                    feature_set, window_samples=ws, hop_samples=hs)
                len_windowed = len(ds_win_fs)
                window_shape = f"({len(keys)}, {ws})"
            except Exception as e:
                len_windowed = f"error: {e}"
                window_shape = "n/a"
            rows.append({
                'config': ctx.config_name,
                'subject': ctx.subject,
                'feature_set': feature_set,
                'n_trials': ctx.ds_full.n_trials,
                'n_channels': ctx.ds_full.n_channels,
                'n_features': len(keys),
                'len(dataset) windowed': len_windowed,
                'window shape (F/C, W)': window_shape,
            })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / '22_shapes_table.csv', index=False)

    fig, ax = plt.subplots(figsize=(18, 0.6 + 0.45 * len(df)))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    tbl.scale(1, 1.8)
    ax.set_title("Resolved tensor shapes per config/feature_set (generated from live objects; "
                 f"window/hop = TRF_conv.py's WINDOW_SEC={WINDOW_SEC:.0f}s/HOP_SEC={HOP_SEC:.0f}s)")
    _save(fig, out_dir, '22_shapes_table.png')
    print("  [viz] summary: wrote 22_shapes_table.png (+ .csv)")


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

_ALL_FIGURES = [
    ('02_envelope_over_waveform', _fig02_envelope_over_waveform, None),
    ('02b_envelope_provided_vs_computed', _fig02b_envelope_provided_vs_computed, _has_mat_provided_envelope),
    ('03_envelope_resample_zoom', _fig03_envelope_resample_zoom, None),
    ('04_envelope_eeg_length_diff', _fig04_envelope_eeg_length_diff, None),
    ('05_envelope_onsets', _fig05_envelope_onsets, None),
    ('06_xcorr_lag', _fig06_xcorr_lag, None),
    ('07_idyom_surprisal_placement', _fig07_idyom_surprisal_placement, _has_surprisal_features),
    ('08_eeg_raw_padding', _fig08_eeg_raw_padding, None),
    ('09_eeg_preprocessing_stages', _fig09_eeg_preprocessing_stages, None),
    ('10_eeg_psd_filters', _fig10_eeg_psd_filters, None),
    ('11_raw_concat_markers', _fig11_raw_concat_markers, None),
    ('12_montage_sensors', _fig12_montage_sensors, None),
    ('13_alignment_trim', _fig13_alignment_trim, None),
    ('14_zscore_distributions', _fig14_zscore_distributions, None),
    ('15_feature_stack_trial', _fig15_feature_stack_trial, None),
    ('16_window_timeline', _fig16_window_timeline, None),
    ('17_window_examples', _fig17_window_examples, None),
    ('18_shape_flow', _fig18_shape_flow, None),
    ('19_window_count_sanity', _fig19_window_count_sanity, None),
    ('20_trial_duration_distribution', _fig20_trial_duration_distribution, None),
]


def _generate_for_context(ctx):
    print(f"\n[viz] generating figures for {ctx.config_name} ({ctx.subject}, trial "
          f"{ctx.trial_idx}, channel {_channel_name(ctx)}, "
          f"stimulus_source_type={ctx.config.stimulus_source_type})")
    ctx.skipped = {}
    for name, fn, applicable in _ALL_FIGURES:
        if applicable is not None and not applicable(ctx):
            ctx.skipped[name] = 'not applicable to this config'
            print(f"  [viz] {ctx.config_name}/{ctx.subject}: skipping {name} (not applicable)")
            continue
        _try(ctx, name, fn)
    return ctx


def run():
    """Entry point called from dataset.py's __main__ under --visualize."""
    print("\n=== viz_smoke_test: generating methodology figures ===")
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    mat_ctx = _build_context('config.yaml', 'mat')
    if mat_ctx is not None:
        _generate_for_context(mat_ctx)

    audio_ctx = None
    for cfg_name in ('config_openmiir.yaml', 'config_daly.yaml'):
        audio_ctx = _build_context(cfg_name, 'audio_files')
        if audio_ctx is not None:
            break
    if audio_ctx is not None:
        _generate_for_context(audio_ctx)
    else:
        print("  [viz] no audio_files config has usable local data "
              "(checked config_openmiir.yaml, config_daly.yaml) — skipping that run")

    contexts = [c for c in (mat_ctx, audio_ctx) if c is not None]
    if mat_ctx is not None and audio_ctx is not None:
        try:
            _fig21_envelope_source_comparison(mat_ctx, audio_ctx)
        except Exception as e:
            print(f"  [viz] FAILED 21_envelope_source_comparison: {type(e).__name__}: {e}")
    else:
        print("  [viz] skip 21_envelope_source_comparison: need both a mat and an "
              "audio_files config's data locally")

    if contexts:
        try:
            _fig22_shapes_table(contexts)
        except Exception as e:
            print(f"  [viz] FAILED 22_shapes_table: {type(e).__name__}: {e}")

    print("\n=== viz_smoke_test: summary ===")
    for ctx in contexts:
        total = len(_ALL_FIGURES)
        msg = f"  {ctx.config_name} ({ctx.subject}): {len(ctx.produced)}/{total} figures written"
        if ctx.skipped:
            msg += f"; {len(ctx.skipped)} skipped: {list(ctx.skipped)}"
        if ctx.errors:
            msg += f"; {len(ctx.errors)} failed: {list(ctx.errors)}"
        print(msg)
    if mat_ctx is None:
        print("  config.yaml: SKIPPED (no local data)")
    if audio_ctx is None:
        print("  audio_files config: SKIPPED (no local data)")
    print(f"  figures written under: {FIGURES_ROOT}")
    return {'mat': mat_ctx, 'audio_files': audio_ctx}
