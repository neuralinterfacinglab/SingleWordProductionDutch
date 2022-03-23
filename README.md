# SingleWordProductionDutch

Scripts to work with the intracranial EEG data from LINK described in LINK

Packages necessary to run are:
numpy
scipy
scikit.learn
pandas 
and
pynwb

To recreate the experiments run

##extract_features.py
This script reads in the iBIDS dataset and extracts features which are then saved to './features'

##reconstruction_minimal.py
Reconstructs the spectrogram from the neural features in a 10-fold cross-validation and synthesizes the audio using the Method described by Griffin and Lim.

##viz-results.py
Can then be used to plot the results figure from the paper.

##reconstuctWave.py
Synthesizes an audio waveform using the method described by Griffin-Lim

##MelFilterBank.py
Applies mel filter banks to spectrograms.
