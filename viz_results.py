import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

if __name__=="__main__":
    
    result_path = r'./results'
    #Load correlation results
    allRes = np.load(os.path.join(result_path,'linearResults.npy'))

    colors = ['C' + str(i) for i in range(10)]

    meanCorrs = np.mean(allRes,axis=(1,2))
    stdCorrs = np.std(allRes, axis=(1,2))

    x = range(len(meanCorrs))
    fig, ax = plt.subplots(1,2,figsize=(14,7))
    #Barplot of average results
    ax[0].bar(x,meanCorrs,yerr=stdCorrs,alpha=0.5,color=colors)
    for p in range(allRes.shape[0]):
        #Add mean results of each patient as scatter points
        ax[0].scatter(np.zeros(allRes[p,:,:].shape[0])+p,np.mean(allRes[p,:,:],axis=1),color=colors[p])

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(['sub-' + "{:02d}".format(i+1) for i in x],rotation=45, ha='right',fontsize=20)
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Correlation')
    #Title
    ax[0].set_title('a',fontsize=20,fontweight="bold")
    # Make pretty
    plt.setp(ax[0].spines.values(), linewidth=2)
    #The ticks
    ax[0].xaxis.set_tick_params(width=2)
    ax[0].yaxis.set_tick_params(width=2)
    ax[0].xaxis.label.set_fontsize(20)
    ax[0].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[0].get_yticklabels()]

    #Despine
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    #Mean across folds over spectral bins
    specMean = np.mean(allRes,axis=1)
    specStd = np.std(allRes,axis=1)
    specBins = np.arange(allRes.shape[2])
    for p in range(allRes.shape[0]):
        ax[1].plot(specBins, specMean[p,:],color=colors[p])
        error = specStd[p,:]/np.sqrt(allRes.shape[1])
        #Shaded areas highlight standard error
        ax[1].fill_between(specBins,specMean[p,:]-error,specMean[p,:]+error,alpha=0.5,color=colors[p])
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,len(specBins))
    ax[1].set_xlabel('Spectral Bin')
    ax[1].set_ylabel('Correlation')
    #Title
    ax[1].set_title('b',fontsize=20,fontweight="bold")

    #Make pretty
    plt.setp(ax[1].spines.values(), linewidth=2)
    #The ticks
    ax[1].xaxis.set_tick_params(width=2)
    ax[1].yaxis.set_tick_params(width=2)
    ax[1].xaxis.label.set_fontsize(20)
    ax[1].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[1].get_yticklabels()]
    c = [a.set_fontsize(20) for a in ax[1].get_xticklabels()]
    #Despine
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path,'results.png'),dpi=600)
    plt.show()

    # Viz example spectrogram
    #Load words and spectrograms
    feat_path = r'./features'
    participant = 'sub-06'
    #Which timeframe to plot
    start_s = 5.5
    stop_s=19.5

    frameshift = 0.01
    #Load spectrograms
    rec_spec = np.load(os.path.join(result_path, f'{participant}_predicted_spec.npy'))
    spectrogram = np.load(os.path.join(feat_path, f'{participant}_spec.npy'))
    #Load prompted words
    eeg_sr= 1024
    words = np.load(os.path.join(feat_path,f'{participant}_procWords.npy'))[int(start_s*eeg_sr):int(stop_s*eeg_sr)]
    words = [words[w] for w in np.arange(1,len(words)) if words[w]!=words[w-1] and words[w]!='']
    
    cm='viridis'
    fig, ax = plt.subplots(2, sharex=True)
    #Plot spectrograms
    pSta=int(start_s*(1/frameshift));pSto=int(stop_s*(1/frameshift))
    ax[0].imshow(np.flipud(spectrogram[pSta:pSto, :].T), cmap=cm, interpolation=None,aspect='auto')
    ax[0].set_ylabel('Log Mel-Spec Bin')
    ax[1].imshow(np.flipud(rec_spec[pSta:pSto, :].T), cmap=cm, interpolation=None,aspect='auto')
    plt.setp(ax[1], xticks=np.arange(0,pSto-pSta,int(1/frameshift)), xticklabels=[str(x/int(1/frameshift)) for x in np.arange(0,pSto-pSta,int(1/frameshift))])
    plt.setp(ax[1], xticks=np.arange(int(1/frameshift),spectrogram[pSta:pSto, :].shape[0],3*int(1/frameshift)), xticklabels=words)
    ax[1].set_ylabel('Log Mel-Spec Bin')

    plt.savefig(os.path.join(result_path,'spec_example.png'),dpi=600)

    #Saving for use in Adobe Illustrator
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.savefig(os.path.join(result_path,'spec_example.pdf'),transparent=True)
    plt.show()


    # Viz waveforms
    #Load waveforms
    rate, audio = wavfile.read(os.path.join(result_path,f'{participant}_orig_synthesized.wav'))
    rate, recAudio = wavfile.read(os.path.join(result_path,f'{participant}_predicted.wav'))
    
    orig = audio[int(start_s*rate):int(stop_s*rate)]
    rec = recAudio[int(start_s*rate):int(stop_s*rate)]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(orig)
    axarr[1].plot(rec)

    #Axis
    xts = np.arange(rate,orig.shape[0]+1,3*rate)
    axarr[1].set_xticks(xts)
    axarr[1].set_xticklabels(words)
    axarr[0].set_xlim([0,orig.shape[0]])
    axarr[0].set_ylim([-np.max(np.abs(orig)),np.max(np.abs(orig))])
    axarr[1].set_ylim([-np.max(np.abs(rec)),np.max(np.abs(rec))])
    
    #Add line indicating 3 seconds
    axarr[1].annotate("",xy=(xts[0], 27000), xycoords='data',xytext=(xts[1],27000), textcoords='data',
                arrowprops=dict(arrowstyle="-",
                                connectionstyle="arc3"),
                )
    axarr[1].annotate("3 seconds",xy=((xts[0]+xts[1])/2, 22000), horizontalalignment='center')

    #Axis labels
    axarr[0].set_ylabel('Original')
    axarr[0].set_yticks([])
    axarr[1].set_yticks([])
    axarr[1].set_ylabel('Amplitude')
    axarr[0].set_ylabel('Amplitude')
    axarr[1].text(orig.shape[0],0,'Reconstruction',horizontalalignment='left',verticalalignment='center',rotation='vertical',)
    axarr[0].text(orig.shape[0],0,'Original',horizontalalignment='left',verticalalignment='center',rotation='vertical',)

    #Make Pretty
    ax[1].xaxis.set_tick_params(width=2)
    ax[1].yaxis.set_tick_params(width=2)
    ax[1].xaxis.label.set_fontsize(20)
    ax[1].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[1].get_yticklabels()]
    c = [a.set_fontsize(20) for a in ax[1].get_xticklabels()]
    #ax.get_yticklabels().set_fontsize(28)

    #Despine
    for axes in axarr:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        #axes.spines['top'].set_visible(False)

    plt.savefig(os.path.join(result_path,'wav_example.png'),dpi=600)
    #Saving for usage in Adobe Illustrator
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.savefig(os.path.join(result_path,'wav_example.pdf'),transparent = True)
    plt.show()



