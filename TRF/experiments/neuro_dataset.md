## 1. DIRECTORY TREE & STRUCTURE
Root level:

```
/Users/arianapereira/Documents/Masters/Projects/SigMA/datasets/neuro_dataset/
├── README
├── dataset_description.json
├── participants.tsv
├── bidsignore
└── datalad/
    └── ds002725-1.0.0/
        ├── README
        ├── dataset_description.json
        ├── participants.tsv
        ├── code/
        │   └── load_EEG-fMRI.m
        ├── stimuli/
        │   ├── classical/          (7 MP3 files)
        │   ├── generated/          (216 WAV files)
        │   └── washout/            (10 MP3 files)
        ├── sub-01/ through sub-21/ (21 subjects total)
        │   ├── anat/              (T1w anatomical MRI)
        │   ├── eeg/               (EEG recordings + metadata)
        │   └── func/              (fMRI + events)
        └── .datalad/ (metadata)
```

## 2. EEG RECORDINGS
Format: EDF (European Data Format)
Sampling Rate: 1000 Hz
Channel Count: 30 EEG channels + 1 ECG channel = 31 total channels
Reference: FCz

EEG Channel Locations (10-20 system):
FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz, Oz, FC1, FC2, CP1, CP2, FC5, FC6, CP5, CP6, TP9, TP10, POz, + ECG

File Organization per subject:
Each subject (sub-01 through sub-21) has EEG data organized by task in /eeg/ folder:

- sub-XX_task-classicalMusic_eeg.edf
- sub-XX_task-genMusic01_eeg.edf
- sub-XX_task-genMusic02_eeg.edf
- sub-XX_task-genMusic03_eeg.edf
- sub-XX_task-washout_eeg.edf
Total EEG Files: 97 EDF files (21 subjects × 5 tasks, but some subjects may be missing data)

Metadata per EEG recording:

* _eeg.json - Technical parameters (sampling rate, channel count, reference, power line frequency)
* _channels.tsv - Channel descriptions (name, type, units, status)
* _events.tsv - Event timing (onset, duration, trial_type)
* _events.json - Event definitions

## 3. MUSICAL STIMULI
Total stimuli: 233 files

Classical Music (7 MP3 files):

- p1_chopin-n10-op12-bertoglio.mp3
- p2_rachmaninoff-n5-op32-prelude-in-g-major.mp3
- p3_chopin-n3-op10-bertoglio.mp3
- p4_chopin-n4-op28-Konczal.mp3
- p5_mendelssohn-variations-serieuses-op54-larrard.mp3
- p6_beethoven-n4-allegro-molte-e-con-brio-sinadinovic.mp3
- p7_beethoven-n3-adagio-con-expression-bertoli.mp3

Generated Music (216 WAV files):
- Named as: X-Y_Z.wav where:
    - X = 1-9 (9 categories, possibly arousal/valence levels)
    - Y = 1-9 (9 sub-categories or compositions)
    - Z = 1-3 (3 variations/repetitions)
    - Pattern examples: 1-2_1.wav, 2-8_2.wav, 9-7_3.wav, etc.

Washout/Baseline (10 MP3 files - animal sounds):

    cat.mp3, chicken.mp3, cow.mp3, crow.mp3, cuckoo.mp3, dog.mp3, duck.mp3, goat.mp3, owl.mp3, sheep.mp3

Stimulus Encoding in EEG data:

From the channels.tsv documentation: The "music" channel in the EEG records which piece was played. To convert: multiply value by 20, convert to string, construct filename from 3-element number (e.g., value 282 → "2-8_2.wav").

## 4. METADATA FILES
Root metadata:

README (full content):

Dataset: Joint EEG-fMRI recording during affective music listening.
This dataset was recorded from 21 healthy adult participants via a joint EEG-fMRI modality while they listened to a set of music stimuli chosen and generated to produce different affective (emotional) reponses. Participants self-reported their felt affective states as they listened to the music.

The full experiment description can be found in our paper (Daly et.al., 2019). 
Data recorded in 2016
Published in 2019

[1] Daly, I., Williams, D., Hwang, F., Kirke, A., Miranda, E. R., & Nasuto, S. J. (2019). Electroencephalography reflects the activity of sub-cortical brain regions during approach-withdrawal behaviour while listening to music. Scientific Reports, 9(1), 9415. https://doi.org/10.1038/s41598-019-45105-2
dataset_description.json (full content):

```
{
    "Name": "A dataset recording joint EEG-fMRI during affective music listening ",
    "BIDSVersion": "1.0.2",
    "Authors": [
        "Ian Daly",
        "Nicoletta Nicolaou",
        "Duncan Williams",
        "Faustina Hwang",
        "Alexis Kirke",
        "Eduardo Miranda",
        "Slawomir J. Nasuto"
    ],
    "License": "CC0",
    "DatasetDOI": "10.18112/openneuro.ds002725.v1.0.0"
}
```

participants.tsv (full content):


|participant_id	|age	|sex|
|--------------|-------|----|
|sub-01	|29|	F|
|sub-02	|21|	F|
|sub-03	|26|	M|
|sub-04	|25|	M|
|sub-05	|25|	M|
|sub-06	|26|	M|
|sub-07	|27|	F|
|sub-08	|26|	F|
|sub-09	|26|	F|
|sub-10	|27|	M|
|sub-11	|20|	F|
|sub-12	|23|	F|
|sub-13	|20|	M|
|sub-14	|22|	F|
|sub-15	|23|	M|
|sub-16	|23|	M|
|sub-17	|22|	F|
|sub-18	|21|	F|
|sub-19	|24|	M|
|sub-20	|27|	M|
|sub-21	|28|	M|


## 5. PRECOMPUTED AUDIO FEATURES
None present. The dataset contains only raw audio stimuli (MP3 and WAV files) and neural recordings. No envelope, onset detection, spectrograms, or other extracted features are present in the dataset.

## 6. PROCESSING SCRIPTS
One MATLAB script present:
``` 
 /datalad/ds002725-1.0.0/code/load_EEG-fMRI.m (798 lines)
```
What it does:

- Loads and processes joint EEG-fMRI data for all 21 subjects
- Extracts EEG data from GDF format and aligns with fMRI scans
- Processes trial onset times and creates design matrices
- Handles valence/arousal ratings from FEELTRACE (real-time emotion assessment device)
- Creates separate regressors for different trial types (music only, music + FEELTRACE, FEELTRACE only)
- Extracts head motion parameters as confounds
- Generates SPM batch job files for statistical analysis

Note: Contains hardcoded paths to local drives (E:, D:, C:) - would need updating to run


## 7. KEY EXPERIMENTAL PARADIGM DETAILS
Dataset Citation:

```
Daly et al., Scientific Reports (2019) - "Electroencephalography reflects the activity of sub-cortical brain regions during approach-withdrawal behaviour while listening to music."
DOI: 10.1038/s41598-019-45105-2
```

Key Recording Parameters:

- Subjects: 21 healthy adults (11F, 10M)
- Age range: 20-29 years (mean ~24.5)
- EEG Sampling Rate: 1000 Hz
- EEG Power line frequency: 50 Hz
- Modality: Joint EEG-fMRI (simultaneous recording)
- fMRI TR: 2 seconds
- Tasks (5 per subject):

    - classicalMusic: Listening to classical piano pieces (7 different compositions)
    - genMusic01, genMusic02, genMusic03: Three runs of generated music with different affective properties (216 pieces total across 9×9×3 naming structure)
    - washout: Baseline/control condition with animal sounds

Behavioral Measurements:

FEELTRACE: Real-time continuous valence/arousal ratings
SAMS: Self-Assessment Manikin for post-stimulus ratings
Behavioral responses recorded as additional channels in EEG
Event Markers in EEG:
Key trial-type codes in events files:

- 1: Experiment start
- 10: TTL pulse from fMRI scanner
- 47: TTL relay from experiment paradigm
- 265: Pulse artifact
- 768: Trial start (music playback onset)
- 1283: ECG Q-wave onset
- 1285: ECG S-wave onset
- 34053: Unused

## 8. FILE STATISTICS SUMMARY
|File Type	|Count	|Purpose|
|---|---|---|
|EDF (EEG)	97	Raw EEG recordings
|WAV	216	Generated music stimuli
|MP3	17	Classical music + washout sounds
|TSV	317	Events, channels, and behavioral data
|JSON	318	Metadata descriptors
|NII.GZ	122	fMRI bold images (compressed)
|MATLAB	1	Processing script
|TOTAL	1,104	
DIRECTORY PATH
Complete path to dataset: /Users/arianapereira/Documents/Masters/Projects/SigMA/datasets/neuro_dataset/datalad/ds002725-1.0.0/

This is a well-structured BIDS-compliant dataset suitable for building audio-to-EEG/fMRI pipelines for studying affective responses to music.