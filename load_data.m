% Download MatNWB (https://neurodatawithoutborders.github.io/matnwb/)

nwbfile = nwbRead('...\SingleWordProductionDutch-iBIDS\sub-01\ieeg\sub-01_task-wordProduction_ieeg.nwb')
eeg = nwbfile.acquisition.get('iEEG').data.load;
audio = nwbfile.acquisition.get('Audio').data.load;
words = nwbfile.acquisition.get('Stimulus').data.load;
