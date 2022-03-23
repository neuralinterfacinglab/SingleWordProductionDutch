import scipy, numpy as np
import scipy.io.wavfile as wavefile

def stft(x, fftsize=1024, overlap=4):
    """Returns short time fourier transform of a signal x
    """
    hop = int(fftsize / overlap)
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]
    return np.array([np.fft.rfft(w*x[i:i+int(fftsize)]) for i in range(0, len(x)-int(fftsize), hop)])

def istft(X, overlap=4):
    """Returns inverse short time fourier transform of a complex spectrum X
    """
    fftsize=(X.shape[1]-1)*2
    hop = int(fftsize / overlap)
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop)
    for n,i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    #Could be improved
    #pos = wsum != 0
    #x[pos] /= wsum[pos]
    return x

def reconstructWavFromSpectrogram(spec,lenWaveFile,fftsize=1024,overlap=4,numIterations=8):
    """Returns a reconstructed waveform from the spectrogram
    using the method in
    Griffin, Lim: Signal estimation from modified short-time Fourier transform,
    IEEE Transactions on Acoustics Speech and Signal Processing, 1984
    algo described here:
    Bayram, Ilker. "An analytic wavelet transform with a flexible time-frequency covering."
    Signal Processing, IEEE Transactions on 61.5 (2013): 1131-1142.
    """
    reconstructedWav = np.random.rand(lenWaveFile*2)
    #while(stft(reconstructedWav,fftsize=fftsize,overlap=overlap).shape[0]<spec.shape[0]):
    #    print(spec.shape[0]-stft(reconstructedWav,fftsize=fftsize,overlap=overlap).shape[0])
    #    lenWaveFile += 1
    #    reconstructedWav = np.random.rand(lenWaveFile)


    for i in range(numIterations):
        x=stft(reconstructedWav,fftsize=fftsize,overlap=overlap)
        #print(str(x.shape) + '  ' + str(spec.shape))
        z=spec*np.exp(1j*np.angle(x[:spec.shape[0],:])) #[:spec.shape[0],:spec.shape[1]]
        re=istft(z,overlap=overlap)
        reconstructedWav[:len(re)]=re
    reconstructedWav=reconstructedWav[:len(re)]
    return reconstructedWav


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    (sr,wf) = wavefile.read('something.wav')
    wf = wf[10*sr:20*sr]
    print('Length of wavefile in samples ' + str(wf.shape[0]))
    print('Sampling rate is ' + str(sr))
    x=stft(wf)
    print('Shape of complex spectrum is ' + str(x.shape))
    spectrogram=np.abs(x)
    plt.matshow(np.log(spectrogram),aspect='auto')
    plt.show()
    import time
    t=time.time()
    data = reconstructWavFromSpectrogram(spectrogram,len(wf))
    print(time.time()-t)
    #plt.plot(wf, 'b',data,'r')
    #plt.show()
    #Save reconstructed file
    wavefile.write('orig.wav',sr,wf)
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    wavefile.write('test.wav',sr,scaled)
    #wavefile.write('test_orig.wav',sr,wf)
