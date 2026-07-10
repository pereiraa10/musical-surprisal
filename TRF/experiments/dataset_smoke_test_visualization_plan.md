# Visual smoke test plan for `dataset.py`

Goal: turn the existing `dataset.py.__main__` smoke test into a methods-figure
generator — every assertion still runs and still fails loud, but each pipeline
stage also drops a PNG that can go straight into the methodology section.
Based on reading `dataset.py`, `utils.py`, and `config.py` as they exist now.

## Constraint that shapes everything else

None of the current pipeline functions (`preprocess_eeg_trials`,
`align_stimulus_and_idyom`, `zscore_trials`) return their intermediate stages —
only the final result. To plot "after LPF" vs "after downsample" vs "after
HPF" you need those intermediates, which means adding **opt-in** instrumentation
(an optional `capture=` callback or `return_stages=True` flag) that changes
nothing about the default return value or the numeric result. This has to be
non-invasive: the existing smoke-test assertions (shared-events aliasing,
z-score independence, window-count arithmetic) must keep passing byte-for-byte.

## Two runs, not one

The plan only works if the smoke test is run once against the default config
(`config.yaml`, liberi dataset, `stimulus_source_type: mat`, precomputed
envelope) and once against `config_daly.yaml` or `config_openmiir.yaml`
(`stimulus_source_type: audio_files`, envelope computed via
`compute_envelope_from_audio`'s Hilbert transform). The provided-vs-computed
envelope contrast is a real methodological fork in the code and is worth its
own comparison figure.

## Figure plan, in pipeline order

**1. Stimulus → envelope → onsets**
- Raw waveform of the stimulus audio at its native rate (audio_files case:
  read via `soundfile` same as `compute_envelope_from_audio` does; mat case:
  read the corresponding file in `wav_dir` if present, so both datasets get a
  waveform panel even though only one *computes* its envelope from it).
- Hilbert envelope overlaid on the raw waveform, before resampling — shows the
  `np.abs(hilbert(...))` step directly (audio_files only; mat case instead
  shows the precomputed `dataStim.mat` envelope at its native `stim_fs`, so
  the two panels are visually side-by-side comparable).
- Envelope before/after `resample_poly` to `target_fs`, zoomed to a ~2s window
  so the interpolation is actually visible (at full-trial zoom this is
  invisible).
- Envelope + EEG-length trim: a small bar chart of `len(env_resampled) - n_eeg`
  across all trials (the `align_stimulus_and_idyom` diff/warning threshold is
  `4 * sfreq`; plot that threshold as a reference line).
- Onsets derivation: envelope and `onsets = diff(envelope).clip(0)` stacked on
  shared time axis for one trial, so the reader can see the impulse-like onset
  signal come directly from the envelope's rising edges.
- Cross-correlation lag plot: `align_stimulus_and_idyom`'s debug branch
  already computes an EEG/envelope xcorr and peak lag but only prints it —
  plot the xcorr curve with the peak marked. This is a concrete alignment QA
  figure, not just decoration.
- (acoustic_and_surprisal only) IDyOM surprisal placement: MIDI note onsets as
  impulses (`make_surprisal_timeseries`) plotted under the envelope, for
  pitch_surprisal/pitch_entropy/onset_surprisal/onset_entropy — shows how a
  symbolic/note-level signal gets placed onto the continuous EEG time grid.

**2. EEG preprocessing (one subject, one trial, 1-3 channels)**
- Raw EEG at original `fs`, with the leading-padding region shaded
  (`pad_start_orig`) — establishes the "padding is still attached" starting
  state `preprocess_eeg_trials` describes.
- After LPF (`butter` order-4 lowpass at `high_frequency`) overlaid on raw.
- After downsample (`resample_poly` to `target_fs`) — a stem/marker plot
  showing the sample count actually drop, not just an overlaid line.
- After HPF (`butter` order-4 highpass at `low_frequency`) overlaid on the
  downsampled signal — should visibly remove slow drift.
- After padding strip — final trial with the removed region grayed out for
  reference.
- One PSD (Welch) panel with raw/LPF/HPF traces overlaid and the
  `low_frequency`/`high_frequency` cutoffs marked as vertical lines — this is
  the figure that actually justifies the filter choices in a methods section,
  the time-domain panels above are more "does this look sane" QA.
- Multi-trial concatenation check: the continuous `create_mne_raw_from_preprocessed`
  Raw with STI markers, showing 3-4 trial boundaries and their marker codes —
  confirms trial-onset placement.
- Optional: montage sanity check — `mne.viz.plot_sensors` from the built
  montage, confirming `chanlocs` X/Y/Z made it through correctly (cheap, and
  catches a whole class of silent bugs if a coordinate got swapped).

**3. Alignment & z-scoring**
- EEG channel 0 vs envelope, both z-scored for comparability, overlaid for one
  trial pre- and post- `align_trial` trimming (shows the "same length now"
  guarantee visually, not just via assertion).
- Distribution (histogram or KDE) of EEG channel 0 and each feature
  (envelope, onsets, ...) before vs after `zscore_trials`, one row per
  feature — demonstrates the standardization step and gives you a place to
  visually catch a dead-channel/zero-variance case before it throws.
- "Figure 1"-style stacked multi-panel: one trial's full feature set
  (envelope, onsets, pitch_surprisal, pitch_entropy, onset_surprisal,
  onset_entropy) as small multiples on a shared time axis, z-scored, EEG
  channel 0 on top — this is the standard stimulus-representation figure used
  in TRF papers (Crosse/Di Liberto-style) and probably the single most
  reusable figure for the paper.

**4. Windowing / tensor shapes**
- Timeline figure: one trial's full duration as a horizontal bar, with
  colored spans for each extracted window (`window_samples`/`hop_samples`),
  including overlap where hop < window — makes the sliding-window mechanics
  in `_build_window_index` legible at a glance.
- A handful of consecutive windows drawn as small stacked time-series panels
  under the full-trial trace, boundaries dropped down as vertical guides —
  ties the abstract "window i" back to an actual signal segment.
- Shape-flow diagram (schematic, not data-driven): boxes for
  `(T_trial, n_channels)` raw EEG → `(T_trial, n_channels)` z-scored →
  `(n_features, window)` X / `(n_channels, window)` Y per `__getitem__` →
  stacked batch `(batch, n_features/n_channels, window)` after `DataLoader`
  collation. Directly documents the tensor shapes named in `dataset.py`'s own
  docstring (`X0.shape == (len(ds_full.feature_keys), T0)` etc.).
- Window-count sanity scatter: per trial, `len(windows_for_trial(ti))` vs
  trial length, with the closed-form `(T - ws)//hs + 1` line overlaid —
  turns the existing `manual == len(ds_win)` assertion into a visual, and
  covers every trial instead of just the aggregate count.

**5. Dataset-level summary (nice-to-have, cheap once per-trial data exists)**
- Trial duration distribution across the whole subject/cohort.
- Provided-vs-computed envelope comparison: liberi (`mat`) trial envelope next
  to a daly/openmiir (`audio_files`) trial envelope, both z-scored, to make
  the "two sourcing paths, same downstream shape" point explicit.
- Total window count and per-subject/per-feature_set tensor shape table,
  auto-generated from the actual smoke-test objects rather than hand-typed
  into the paper (avoids the classic "code changed, methods section didn't"
  drift).

## Suggested output layout

```
experiments/figures/smoke_test/<config_name>/<subject>/
  01_stimulus_waveform_envelope.png
  02_envelope_resample_zoom.png
  03_envelope_eeg_length_diff.png
  04_envelope_onsets.png
  05_xcorr_lag.png
  06_idyom_surprisal_placement.png   # acoustic_and_surprisal only
  07_eeg_preprocessing_stages.png
  08_eeg_psd_filters.png
  09_raw_concat_markers.png
  10_montage_sensors.png
  11_alignment_trim.png
  12_zscore_distributions.png
  13_feature_stack_trial.png
  14_window_timeline.png
  15_window_examples.png
  16_shape_flow.png
  17_window_count_sanity.png
  summary/
    dataset_descriptives.png
    envelope_source_comparison.png
    shapes_table.png (or .csv, for a LaTeX table)
```

Matches the existing convention in `TRF_conv.py` (`Agg` backend, `dpi=150`,
`bbox_inches='tight'`).

---

## Prompt for Claude Code

```
In experiments/dataset.py, the `if __name__ == '__main__':` block is a smoke
test for PreparedSubject/TRFDataset. Extend it (don't replace the existing
assertions — every one of them must still pass unchanged) so that it also
produces a full set of methodology-ready figures showing how raw stimulus
audio and raw EEG become a TRFDataset's tensors. This is for a research
paper's methods section, so figures should be legible standalone (axis
labels, units, titles with subject/trial ids) — not just debug scribbles.

Constraints:
- Do not change the numeric behavior of any existing function in utils.py.
  Where you need an intermediate value that isn't currently returned (e.g.
  the EEG signal after LPF but before downsampling, inside
  preprocess_eeg_trials), add it as an OPT-IN capture: either a `capture=`
  callback argument or a `return_stages=False` flag that defaults to
  preserving today's return signature exactly. Every existing caller of these
  functions (TRF_conv.py, TRF_mne.py, TRF_sklearn.py, TRF_boosting.py,
  run_daly_dataset.py, etc.) must keep working with zero changes.
- Add a new `--visualize` / `--no-visualize` CLI-style flag (plain sys.argv
  check is fine, this file has no argparse today) so the figure generation
  can be skipped for a fast CI-style run of just the assertions.
- Put the new plotting code in a new module, experiments/viz_smoke_test.py,
  imported by dataset.py's __main__ block — don't bloat dataset.py itself.
  Use matplotlib with the 'Agg' backend (see TRF_conv.py's existing
  `matplotlib.use('Agg')` pattern) and save at dpi=150, bbox_inches='tight',
  matching TRF_conv.py's convention.
- Output figures to experiments/figures/smoke_test/<config_stem>/<subject>/,
  creating directories as needed. <config_stem> should distinguish the
  default config.yaml run from config_daly.yaml/config_openmiir.yaml runs.

Run the smoke test twice: once against the default config (config.yaml,
liberi_dataset, stimulus_source_type='mat', precomputed envelope) and once
against config_daly.yaml or config_openmiir.yaml (stimulus_source_type=
'audio_files', envelope computed from raw audio via
utils.compute_envelope_from_audio's Hilbert transform). Some figures apply to
both runs; the envelope-sourcing figures should make the mat-vs-audio_files
contrast explicit.

Generate this set of figures, one representative subject and trial per run
unless noted (pick the first configured subject and its first trial with
audio_files.  the 'events'-index that survives align_stimulus_and_idyom):

STIMULUS / ENVELOPE / ONSETS
1. Raw stimulus waveform (native sample rate) for the chosen trial.
2. Envelope overlaid on the raw waveform before resampling — Hilbert-derived
   for audio_files configs, the precomputed dataStim.mat envelope (at its
   native stim_fs) for mat configs.
3. Envelope before vs after resample_poly to target sfreq, zoomed to ~2
   seconds so the resampling effect is actually visible.
4. Bar chart across all of the subject's trials of
   (len(env_resampled) - n_eeg) from align_stimulus_and_idyom, with the
   function's own `4 * sfreq` warning threshold drawn as a reference line.
5. Envelope and its derived onsets (diff(envelope).clip(0)) stacked on a
   shared time axis, one trial.
6. The EEG/envelope cross-correlation curve that align_stimulus_and_idyom's
   debug branch already computes (currently print-only) with the peak lag
   marked — turn this into a saved figure instead of a console line.
7. (acoustic_and_surprisal feature_set only) MIDI note-onset impulses for
   pitch_surprisal/pitch_entropy/onset_surprisal/onset_entropy plotted under
   the envelope for one trial, showing how make_surprisal_timeseries places
   symbolic-score values onto the continuous time grid.

EEG PREPROCESSING
8. One channel's raw EEG at original fs with the leading-padding region
   shaded (pad_start at orig fs).
9. Same channel after LPF, after downsample, after HPF, and after padding
   strip — either four subplots or an overlaid progression — using a
   return_stages/capture hook into preprocess_eeg_trials as described above.
10. Welch PSD of raw vs LPF'd vs HPF'd (final) signal for that channel, with
    vertical lines at low_frequency and high_frequency marking the actual
    filter cutoffs used.
11. The concatenated MNE Raw (create_mne_raw_from_preprocessed) for a few
    trials, showing the STI marker channel and trial boundaries.
12. mne.viz.plot_sensors from the built montage, confirming chanlocs
    positions came through correctly.

ALIGNMENT / Z-SCORING
13. EEG channel 0 vs envelope (both z-scored), overlaid, for one trial after
    align_trial's length trim.
14. Before/after histograms of zscore_trials for the EEG and each feature key
    in that feature_set, one row per feature.
15. A "figure 1"-style stacked panel: EEG channel 0 plus every feature in
    the acoustic_and_surprisal feature_set (envelope, onsets,
    pitch_surprisal, pitch_entropy, onset_surprisal, onset_entropy), all
    z-scored, shared time axis, one trial — this should look like a
    publication figure, it's the main reusable one.

WINDOWING / TENSOR SHAPES
16. One trial's duration as a horizontal timeline with colored spans for
    each extracted window (from a TRFDataset built with window_samples/
    hop_samples set), showing overlap when hop < window.
17. 4-6 consecutive windows drawn as stacked mini time-series under the
    full-trial trace with window boundaries as vertical guides.
18. A schematic shape-flow diagram (text/boxes, doesn't need real axes):
    (T, n_channels) raw EEG -> (T, n_channels) z-scored -> (n_features,
    window) X / (n_channels, window) Y per __getitem__ -> (batch,
    n_features/n_channels, window) after DataLoader collation. Pull the
    actual numbers (T, n_channels, n_features, window, a chosen batch size)
    from the live smoke-test objects rather than hardcoding them.
19. Scatter of len(windows_for_trial(ti)) vs trial length across all trials,
    with the closed-form (T - ws)//hs + 1 line overlaid.

DATASET-LEVEL SUMMARY
20. Trial-duration distribution across the subject's trials.
21. Side-by-side envelope comparison: one liberi (mat) trial's envelope next
    to one daly/openmiir (audio_files) trial's envelope, both z-scored,
    making the two envelope-sourcing paths visually comparable. (Requires
    having run the smoke test against both configs; skip gracefully with a
    printed note if only one config's data is available in this environment.)
22. A small table (saved as PNG or CSV) of resolved tensor shapes per
    feature_set: n_trials, n_channels, per-feature_set n_features, and
    (if windowed) len(dataset) and window shape — generated from the actual
    objects, not hand-written, so it can't drift from the code.

After implementing, run the smoke test yourself (with --visualize) against
whichever config's data is actually present in this environment, confirm all
existing assertions still print "OK"/"PASSED" and the new figures are written
to disk without errors, and report which of the 22 figures were produced vs
skipped (e.g. audio_files-only or mat-only figures when only one config's
data exists locally). If eelbrain/mne/torch aren't installed in this
environment, say so rather than guessing at whether the code works.
```
