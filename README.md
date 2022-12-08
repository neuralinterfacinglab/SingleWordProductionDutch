# SingleWordProductionDutch

Scripts to work with the intracranial EEG data from [here](https://osf.io/nrgx6/) described in this [article](https://www.nature.com/articles/s41597-022-01542-9).

## Dependencies
The scripts require Python >= 3.6 and the following packages
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/scipylib/index.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/) 
* [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)

## Repository content
To recreate the experiments, run the following scripts.
* __extract_features.py__: Reads in the iBIDS dataset and extracts features which are then saved to './features'

* __reconstruction_minimal.py__: Reconstructs the spectrogram from the neural features in a 10-fold cross-validation and synthesizes the audio using the Method described by Griffin and Lim.

* __viz_results.py__: Can then be used to plot the results figure from the paper.

* __reconstuctWave.py__: Synthesizes an audio waveform using the method described by Griffin-Lim

* __MelFilterBank.py__: Applies mel filter banks to spectrograms.

* __load_data.mat__: Example of how to load data using Matlab instead of Python