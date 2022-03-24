import os

import pandas as pd
import numpy as np 
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile

from pynwb import NWBHDF5IO
import MelFilterBank as mel

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr,windowLength=0.05,frameshift=0.01):
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    
    numWindows=int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    sos= scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise
    sos= scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise
    sos= scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)

    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() # Add 'F' if stacked the same as matlab
    return featStacked

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        newLabels[w]=scipy.stats.mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = scipy.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+windowLength*sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs,modelOrder=4):
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  # Add 'F' if stacked the same as matlab


if __name__=="__main__":
    winL = 0.05
    frameshift = 0.01
    modelOrder=4
    stepSize=5
    path_bids = r'./SingleWordProductionDutch-iBIDS'
    path_output = r'./features'
    participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')
    for p_id, participant in enumerate(participants['participant_id']):
        
        #Load data
        io = NWBHDF5IO(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        #sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr=1024
        #audio
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sampling_rate = 48000
        #words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()
        #channels
        channels = pd.read_csv(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_channels.tsv'), delimiter='\t')
        channels = np.array(channels['name'])


        # Extract HG features
        feat = extractHG(eeg,eeg_sr, windowLength=winL,frameshift=frameshift)

        # Stack features
        feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)
        
        #Process Audio
        target_SR = 16000
        audio = scipy.signal.decimate(audio,int(audio_sampling_rate / target_SR))
        audio_sampling_rate = target_SR
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
        scipy.io.wavfile.write(os.path.join(path_output,f'{participant}_orig_audio.wav'),audio_sampling_rate,scaled)   

        melSpec = extractMelSpecs(scaled,audio_sampling_rate,windowLength=winL,frameshift=frameshift)
        
        # Align to EEG features
        melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
        # Adjusting length (differences might occur due to rounding in the number of windows)
        if melSpec.shape[0]!=feat.shape[0]:
            tLen = np.min([melSpec.shape[0],feat.shape[0]])
            melSpec = melSpec[:tLen,:]
            feat = feat[:tLen,:]
        
        #Creating feature names by appending the temporal shift 
        feature_names = nameVector(channels[:,None], modelOrder=modelOrder)
        # Save everything
        np.save(os.path.join(path_output,f'{participant}_feat.npy'), feat)
        np.save(os.path.join(path_output,f'{participant}_procWords.npy'), words)
        np.save(os.path.join(path_output,f'{participant}_spec.npy'),melSpec)
        np.save(os.path.join(path_output,f'{participant}_feat_names.npy'), feature_names )