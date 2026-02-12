import numpy as np
import pretty_midi

def make_surprisal_timeseries(
    midi_path,
    surprisal_vec,
    sfreq,
    n_times
):
    """
    Create a time-varying surprisal predictor aligned to EEG time.

    Returns
    -------
    surprisal_ts : np.ndarray, shape (n_times,)
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Collect note onsets
    onsets = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            onsets.append(note.start)

    onsets = np.array(sorted(onsets))

    # Align lengths
    n = min(len(onsets), len(surprisal_vec))
    onsets = onsets[:n]
    surprisal_vec = surprisal_vec[:n]

    # Initialize continuous time series
    surprisal_ts = np.zeros(n_times)

    for t, s in zip(onsets, surprisal_vec):
        sample = int(round(t * sfreq))
        if 0 <= sample < n_times:
            surprisal_ts[sample] += s  # impulse

    return surprisal_ts
