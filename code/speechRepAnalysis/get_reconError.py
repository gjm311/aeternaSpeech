import sys
import os
import torch
import pywt
import pickle
import scaleogram as scg 
from scipy.io.wavfile import read
import scipy
import numpy as np
from AEspeech import AEspeech

from librosa.feature import melspectrogram

import toolbox.traintestsplit as tts
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    if len(sys.argv)!=4:
        print("python get_rep.py <CAE or RAE> <spec or wvlt> <path_speech>")
        sys.exit()
        
    #path_speech: "./tedx_spanish_corpus/speech/test/"
    
    if sys.argv[3][0] != '/':
        sys.argv[3] = '/'+sys.argv[3]
        
    if sys.argv[3][-1] != "/":
        sys.argv[3] = sys.argv[3]+'/'

    PATH=os.path.dirname(os.path.abspath(__file__))
    
    mod=sys.argv[1]
    rep=sys.argv[2]
    path_audio=PATH+sys.argv[3]
    wav_files=os.listdir(path_audio)
    num_files=len(os.listdir(path_audio))
    save_path=PATH+'/pts/'+mod+'_'+rep+'.pickle'
    unit=256
    if rep=='wvlt':
        time_steps=256
    else:
        time_steps=126
    
    data={'means':np.zeros((num_files,time_steps)), 'stds':np.zeros((num_files,time_steps))}
    
    for ii,wav_file in enumerate(wav_files):
        wav_file=path_audio+wav_file
        if ".wav" in wav_file:
            fs_in, signal=read(wav_file)
        else:
            continue
        # for mod in models:
        aespeech=AEspeech(model=mod,units=unit,rep=rep)
        if rep=='spec':
            mat=aespeech.compute_spectrograms(wav_file)
            mat=aespeech.standard(mat)
        if rep=='wvlt':
            mat,freqs=aespeech.compute_cwt(wav_file)

        if torch.cuda.is_available():
            mat=mat.cuda()
        to,bot=aespeech.AE.forward(mat)

        if rep=='spec':
            mat=aespeech.destandard(mat)
            to=aespeech.destandard(to)

        data['means'][ii,:]=np.mean(np.mean((mat[:,0,:,:].detach().numpy()-to[:,0,:,:].detach().numpy())**2,axis=1),axis=0)
        data['stds'][ii,:]=np.std(np.std((mat[:,0,:,:].detach().numpy()-to[:,0,:,:].detach().numpy())**2,axis=1),axis=0)
        
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)