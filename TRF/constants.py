from pathlib import Path 

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # TRF/

# Define the dataset root; replace it with the proper path if you downloaded the dataset in a different location
DATA_ROOT = BASE_DIR / 'liberi_dataset/doi_10_5061_dryad_g1jwstqmh__v20211008'

# Define paths that will be used throughout
WAV_DIR = DATA_ROOT / 'diliBach_wav_4dryad'
MIDI_DIR = DATA_ROOT / 'diliBach_midi_4dryad'
EEG_DIR = DATA_ROOT / 'diliBach_4dryad_CND'
SURPRISAL_FILE = BASE_DIR / '../codeForPaper-IDyOMpy-/benchmark_results/forBenchmark_IDyOMpy/eLife_trained_on_mixed2.mat'

SAVE_DIR = BASE_DIR / 'pickles'

# Load the raw EEG file for a single subject and a specified frequency band
# Adjust subject index and the low/high frequency values accordingly

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
LOW_FREQUENCY = 4
HIGH_FREQUENCY = 8