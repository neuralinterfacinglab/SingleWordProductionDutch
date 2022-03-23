# SingleWordProductionDutch

Scripts to work with the intracranial EEG data from LINK described in LINK

## Requirements
* Python > 3.6
* numpy
* scipy
* scikit.learn
* pandas 
* [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)

## Repository content
To recreate the experiments, run the following scripts.
* __extract_features.py__: Reads in the iBIDS dataset and extracts features which are then saved to './features'

* __reconstruction_minimal.py__: Reconstructs the spectrogram from the neural features in a 10-fold cross-validation and synthesizes the audio using the Method described by Griffin and Lim.

* __viz-results.py__: Can then be used to plot the results figure from the paper.

* __reconstuctWave.py__: Synthesizes an audio waveform using the method described by Griffin-Lim

* __MelFilterBank.py__: Applies mel filter banks to spectrograms.
