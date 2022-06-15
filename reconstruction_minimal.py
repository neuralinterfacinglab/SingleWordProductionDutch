import os

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import reconstructWave as rW
import MelFilterBank as mel


def createAudio(spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01):
    """
    Create a reconstructed audio wavefrom
    
    Parameters
    ----------
    spectrogram: array
        Spectrogram of the audio
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram was calculated
    frameshift: float
        Shift (in seconds) after which next window was extracted
    Returns
    ----------
    scaled: array
        Scaled audio waveform
    """
    mfb = mel.MelFilterBank(int((audiosr*winLength)/2+1), spectrogram.shape[1], audiosr)
    nfolds = 10
    hop = int(spectrogram.shape[0]/nfolds)
    rec_audio = np.array([])
    for_reconstruction = mfb.fromLogMels(spectrogram)
    for w in range(0,spectrogram.shape[0],hop):
        spec = for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
        rec = rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        rec_audio = np.append(rec_audio,rec)
    scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
    return scaled


if __name__=="__main__":
    feat_path = r'./features'
    result_path = r'./results'
    pts = ['sub-%02d'%i for i in range(1,11)]

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000

    nfolds = 10
    kf = KFold(nfolds,shuffle=False)
    est = LinearRegression(n_jobs=5)
    pca = PCA()
    numComps = 50
    
    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((len(pts),nfolds,23))
    explainedVariance = np.zeros((len(pts),nfolds))
    numRands = 1000
    randomControl = np.zeros((len(pts),numRands, 23))

    for pNr, pt in enumerate(pts):
        #Load the data
        spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
        data = np.load(os.path.join(feat_path,f'{pt}_feat.npy'))
        labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
        featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
        
        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            trainData=(data[train,:]-mu)/std
            testData=(data[test,:]-mu)/std

            #Fit PCA to training data
            pca.fit(trainData)
            #Get percentage of explained variance by selected components
            explainedVariance[pNr,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
            #Tranform data into component space
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)
            
            #Fit the regression model
            est.fit(trainData, spectrogram[train, :])
            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = est.predict(testData)

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Estimate random baseline
        for randRound in range(numRands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(spectrogram.shape[0]*0.1),int(spectrogram.shape[0]*0.9)))
            #Swap the dataset on the splitting point 
            shuffled = np.concatenate((spectrogram[splitPoint:,:],spectrogram[:splitPoint,:]))
            #Calculate the correlations
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[:,specBin], shuffled[:,specBin])
                randomControl[pNr, randRound,specBin]=r


        #Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)
        np.save(os.path.join(result_path,f'{pt}_predicted_spec.npy'), rec_spec)
        
        #Synthesize waveform from spectrogram using Griffin-Lim
        reconstructedWav = createAudio(rec_spec,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}_predicted.wav'),int(audiosr),reconstructedWav)

        #For comparison synthesize the original spectrogram with Griffin-Lim
        origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)

    #Save results in numpy arrays          
    np.save(os.path.join(result_path,'linearResults.npy'),allRes)
    np.save(os.path.join(result_path,'randomResults.npy'),randomControl)
    np.save(os.path.join(result_path,'explainedVariance.npy'),explainedVariance)