import os

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import reconstructWave as rW
import MelFilterBank as mel


def createAudio(spectrogram,audiosr=16000,winLength=0.05,frameshift=0.01):
    mfb = mel.MelFilterBank(int((audiosr*winLength)/2+1), spectrogram.shape[1], audiosr)
    nfolds=10
    hop=int(spectrogram.shape[0]/nfolds)
    rec_audio=np.array([])
    for_reconstruction=mfb.fromLogMels(spectrogram)

    for w in range(0,spectrogram.shape[0],hop):
        spec=for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
        rec=rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        rec_audio=np.append(rec_audio,rec)
    scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
    return scaled

if __name__=="__main__":
    feat_path = r'./features'
    result_path = r'./results'
    pts = ['sub-%02d'%i for i in range(1,11)]

    winLength=0.05
    frameshift=0.01 #0.025
    audiosr=16000

    nfolds =10
    kf = KFold(nfolds,shuffle=False)
    est=LinearRegression(n_jobs=5)
    pca=PCA()
    numComps = 50
    
    allRes = np.zeros((len(pts),nfolds,23))
    for pNr, pt in enumerate(pts):
        #Load the data
        spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
        data = np.load(os.path.join(feat_path,f'{pt}_feat.npy'))
        labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
        featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
        
        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Saving the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            trainData=(data[train,:]-mu)/std
            testData=(data[test,:]-mu)/std

            #Fit PCA to training data
            pca.fit(trainData)
            # Tranform data into component space
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)
            
            #Fitting the regression model
            est.fit(trainData, spectrogram[train, :])
            #Predicting the reconstructed spectrogram for the test data
            rec_spec[test, :] = est.predict(testData)

            # Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in recontruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Save reconstructed spectrogram
        np.save(os.path.join(result_path,f'{pt}_predicted_spec.npy'), rec_spec)
        
        #Synthesize wavform from spectrogram using Griffin-Lim
        reconstructedWav =createAudio(rec_spec,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}_predicted.wav'),int(audiosr),reconstructedWav)

        #For comparison synthesize the original spectrogram with Griffin-Lim
        origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)
              
    np.save(os.path.join(result_path,'linearResults.npy'),allRes)