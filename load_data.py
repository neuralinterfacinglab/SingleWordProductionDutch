from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd
import os

if __name__=="__main__":
    path_bids = r'./SingleWordProductionDutch-iBIDS'
    path_output = r'./'
    participants = pd.read_csv(f'{path_bids}/participants.tsv', delimiter='\t')
    for s_id, subject in enumerate(participants['participant_id']):
        
        #Load data
        io = NWBHDF5IO(f'{path_bids}/{subject}/ieeg/{subject}_task-wordProduction_ieeg.nwb', 'r')
        nwbfile = io.read()
        #sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        #audio
        audio = nwbfile.acquisition['Audio'].data[:]
        #words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()
        #channels
        channels = pd.read_csv(f'{path_bids}/{subject}/ieeg/{subject}_task-wordProduction_channels.tsv', delimiter='\t')
        channels = np.array(channels['name'])

        #Save as numpy array
        os.makedirs(f'{path_output}/raw', exist_ok=True)
        np.save(f'{path_output}/raw/{subject}_sEEG.npy', eeg)
        np.save(f'{path_output}/raw/{subject}_audio.npy',audio)
        np.save(f'{path_output}/raw/{subject}_words.npy', words)
        np.save(f'{path_output}/raw/{subject}_channels.npy', channels)

        print(f'Loaded data of {subject}')

    print('Loading data completed')
            