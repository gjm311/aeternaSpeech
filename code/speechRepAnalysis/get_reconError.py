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



if __name__ == "__main__":

    if len(sys.argv)!=4:
        print("python get_reconError.py <CAE or RAE> <spec or wvlt> <path_speech>")
        sys.exit()
    
        
    #path_speech: "./pdSpanish/speech/"
    
    if sys.argv[3][0] != '/':
        sys.argv[3] = '/'+sys.argv[3]
        
    if sys.argv[3][-1] != "/":
        sys.argv[3] = sys.argv[3]+'/'

    PATH=os.path.dirname(os.path.abspath(__file__))
    
    mod=sys.argv[1]
    rep=sys.argv[2]
#     path_audio=PATH+sys.argv[3]
#     pd_path=path_audio+'pd/'
#     hc_path=path_audio+'hc/'
    

    save_path=PATH+'/pts/'+'/reconErrs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path=save_path+mod+'_'+rep+'.pickle'

    unit=256
    if rep=='wvlt':
        time_steps=256
    else:
        time_steps=126
    
    data={spk:{'means':[], 'stds':[]} for spk in ['pd','hc']}
    utters= os.listdir(PATH+sys.argv[3])
    
    for itr,utter in enumerate(utters):
        path_utter=PATH+sys.argv[3]+'/'+utter+'/'
    
        for spk in ['pd','hc']:
            path_audio=path_utter+spk+'/'
            dirNames=os.listdir(path_audio)
            wav_files=[name for name in dirNames if '.wav' in name]

            num_files=len(wav_files)
            data_curr=np.zeros((num_files,time_steps))
            data_curr=np.zeros((num_files,time_steps))

            for ii,wav_file in enumerate(wav_files):
                wav_file=path_audio+wav_file
                fs_in, signal=read(wav_file)

                # for mod in models:
                aespeech=AEspeech(model=mod,units=unit,rep=rep)
                if rep=='spec':
                    mat=aespeech.compute_spectrograms(wav_file)
                if rep=='wvlt':
                    mat,freqs=aespeech.compute_cwt(wav_file)

                if torch.cuda.is_available():
                    mat=mat.cuda()
                to,bot=aespeech.AE.forward(mat)

                data_curr[ii,:]=np.mean(np.mean((mat[:,0,:,:].cpu().detach().numpy()-to[:,0,:,:].cpu().detach().numpy())**2,axis=1),axis=0)
                data_curr[ii,:]=np.std(np.std((mat[:,0,:,:].cpu().detach().numpy()-to[:,0,:,:].cpu().detach().numpy())**2,axis=1),axis=0)
                
            if itr==0:
                data[spk]['means']=data_curr
                data[spk]['stds']=data_curr
            else:
                data[spk]['means']=np.concatenate((data[spk]['means'],data_curr),axis=0)
                data[spk]['stds']=np.concatenate((data[spk]['stds'],data_curr),axis=0)
               
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
