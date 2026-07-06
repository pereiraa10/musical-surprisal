"""
check_trial_song_repeats.py
────────────────────────────────────────────────────────────────────────────────
Standalone, fast, no-model diagnostic. Answers ONE question: does SUBJECT's
trial list replay the same song (song_id) more than once, and if so, does the
CURRENT trial-index-based holdout in TRF_conv_mini_windowtest_trialholdout.py
(last HELD_OUT_N_TRIALS trials, by index) accidentally hold out songs that are
ALSO present in the training trials?

WHY THIS MATTERS
────────────────────────────────────────────────────────────────────────────────
heldout_r values of 0.5-0.6 came back from TRF_conv_mini_windowtest_trialholdout.py
— an order of magnitude higher than anything else in this project (ridge ~0.02-
0.03; even the previously-flagged-as-suspicious conv numbers topped out ~0.08).
That is not a plausible "better windowing scheme" result; it is a strong signal
of a leak. The most likely candidate: song_id is computed everywhere in this
codebase as `int(stimulus_id % 10) or 10` — a strong hint that there are only
10 unique songs and that stimulus_id cycles through them across multiple blocks.
diagnostic_d2_shuffle.py's console output already showed "30 trials" for Sub2 —
if that's 10 songs x 3 repetitions, holding out the LAST 2 trials BY INDEX
almost certainly holds out repeats of songs the model already trained on under
a different trial index (a different presentation of the identical audio), which
would explain an artificially huge heldout_r without genuine generalization.

This script loads ONLY the EEG events (no windowing, no model, no torch) and
prints, for one subject, the song_id sequence across all trials, whether the
current last-N-by-index holdout would overlap with training songs, and how
many trials/songs you'd get with a corrected SONG-level holdout instead.

Run from musical-surprisal/TRF/:
    python check_trial_song_repeats.py
────────────────────────────────────────────────────────────────────────────────
"""

import constants as constants
import eeg_functions as eeg_func
import eelbrain

SUBJECT = 'Sub2'
HELD_OUT_N_TRIALS = 2   # matches the CURRENT (index-based) holdout being checked
HELD_OUT_N_SONGS  = 2   # matches the PROPOSED (song-based) holdout

print(f"Loading {SUBJECT} events (no windowing, no model)...")
eeg_data = eeg_func.load_subject_raw_eeg(
    constants.EEG_DIR / f'data{SUBJECT}.mat', SUBJECT)
preprocessed_trials = eeg_func.preprocess_eeg_trials(
    eeg_data, target_fs=64,
    lpf_hz=constants.HIGH_FREQUENCY, hpf_hz=constants.LOW_FREQUENCY, debug=False)
raw = eeg_func.create_mne_raw_from_preprocessed(
    preprocessed_trials, 64, eeg_data['chanlocs'])
events = eeg_func.create_eelbrain_events(raw)

stimulus_ids = list(events['event'])
song_ids = [int(sid % 10) or 10 for sid in stimulus_ids]
n_trials = len(stimulus_ids)

print(f"\n{SUBJECT}: {n_trials} trials, {len(set(song_ids))} unique songs\n")
print(f"{'trial_idx':>9}  {'stimulus_id':>11}  {'song_id':>7}")
for i, (sid, song) in enumerate(zip(stimulus_ids, song_ids)):
    print(f"{i:>9}  {sid:>11}  {song:>7}")

from collections import Counter
counts = Counter(song_ids)
print("\nRepetitions per song:")
for song, n in sorted(counts.items()):
    print(f"  song {song}: {n} trial(s) — indices "
          f"{[i for i, s in enumerate(song_ids) if s == song]}")

# ── Check 1: current index-based holdout (last N trials) ────────────────────────
train_idx_current = list(range(n_trials - HELD_OUT_N_TRIALS))
heldout_idx_current = list(range(n_trials - HELD_OUT_N_TRIALS, n_trials))
train_songs_current = set(song_ids[i] for i in train_idx_current)
heldout_songs_current = set(song_ids[i] for i in heldout_idx_current)
overlap_current = train_songs_current & heldout_songs_current

print(f"\n{'='*70}\nCHECK 1 — current index-based holdout "
      f"(last {HELD_OUT_N_TRIALS} trials by index)\n{'='*70}")
print(f"  Held-out trial indices: {heldout_idx_current} -> songs {heldout_songs_current}")
print(f"  Training songs: {sorted(train_songs_current)}")
if overlap_current:
    print(f"  [!!] LEAK CONFIRMED: song(s) {overlap_current} appear in BOTH "
          f"training and held-out trials.")
    print(f"       The model can see this exact audio during training (under a "
          f"different trial index) and then get evaluated on it again as if it "
          f"were unseen — this alone can explain an inflated heldout_r.")
else:
    print(f"  [OK] No song overlap between train and held-out trials under the "
          f"current index-based split.")

# ── Check 2: proposed song-based holdout ─────────────────────────────────────────
unique_songs_in_order = list(dict.fromkeys(song_ids))  # first-appearance order
heldout_songs_proposed = set(unique_songs_in_order[-HELD_OUT_N_SONGS:])
train_idx_song = [i for i, s in enumerate(song_ids) if s not in heldout_songs_proposed]
heldout_idx_song = [i for i, s in enumerate(song_ids) if s in heldout_songs_proposed]

print(f"\n{'='*70}\nCHECK 2 — proposed song-based holdout "
      f"(last {HELD_OUT_N_SONGS} unique songs)\n{'='*70}")
print(f"  Held-out songs: {sorted(heldout_songs_proposed)}")
print(f"  Held-out trial indices: {heldout_idx_song} ({len(heldout_idx_song)} trials)")
print(f"  Training trial indices: {train_idx_song} ({len(train_idx_song)} trials)")
assert set(song_ids[i] for i in train_idx_song).isdisjoint(heldout_songs_proposed)
print(f"  [OK by construction] No training trial shares a song with a held-out trial.")
