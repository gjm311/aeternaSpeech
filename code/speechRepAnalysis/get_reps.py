import sys
import os
import torch
import pywt
import pandas as pd
import scaleogram as scg 
from scipy.io.wavfile import read
import scipy
import numpy as np
import numpy.fft
import cv2

from librosa.feature import melspectrogram

import toolbox.traintestsplit as tts
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    if len(sys.argv)!=2:
        print("python get_rep.py <path_audios>")
        sys.exit()
    
    if sys.argv[1][0] != '/':
        sys.argv[1] = '/'+sys.argv[1]
        
    if sys.argv[1][-1] != "/":
        sys.argv[1] = sys.argv[1]+'/'

    PATH=os.path.dirname(os.path.abspath(__file__))
    PATH_AUDIO=PATH+sys.argv[1]
    PATH_SPEC=PATH_AUDIO+"/../reps/spec/"
    PATH_WVLT=PATH_AUDIO+"/../reps/wvlt/"
    
    if not os.path.exists(PATH_SPEC):
        os.makedirs(PATH_SPEC)
    if not os.path.exists(PATH_WVLT):
        os.makedirs(PATH_WVLT)
        
    if not os.path.exists(PATH_AUDIO+'/train/'):
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=0.1)
        split.fileTrTstSplit()
    elif len(os.listdir(PATH_AUDIO+'/train/'))<=2:
        split=tts.trainTestSplit(PATH_AUDIO, tst_perc=0.1)
        split.fileTrTstSplit()
    
    FS=16000
    NFFT=512
    FRAME_SIZE=0.5
    TIME_SHIFT=0.25
    HOP=64
    NMELS=128
    DIM=()
    SNIP_LEN=50#in mS
    
    minSpec_en = np.inf
    maxSpec_en = -np.inf
    minWvlt_en = np.inf
    maxWvlt_en = -np.inf
    enrgy = {'Min spec Scale': [], 'Max spec Scale': [], 'Min wvlt Scale': [], 'Max wvlt Scale': []}
    
    for trtst in ['/train/', '/test/']:
        audio_path=PATH_AUDIO+trtst
        spec_path=PATH_SPEC+trtst
        wvlt_path=PATH_WVLT+trtst
        
        hf=os.listdir(audio_path)
        hf.sort()
        if len(hf) == 0:
            print(audio_path+ " is empty...", len(hf))
            sys.exit()
        else:
            print(audio_path, len(hf))

        if not os.path.exists(spec_path):
            os.makedirs(spec_path)
        if not os.path.exists(wvlt_path):
            os.makedirs(wvlt_path)

        countbad=0
        countinf=0
        
        for j in range(len(hf)):
            print("Procesing audio", j+1, hf[j]+" of "+str(len(hf)))
            fs_in, data=read(audio_path+hf[j])
            if fs_in!=16000:
                raise ValueError(str(fs)+" is not a valid sampling frequency")
            
           
            if len(data.shape)>1:
                continue
            data=data-np.mean(data)
            data=data/np.max(np.abs(data))
            
            file_spec_out=spec_path+hf[j].replace(".wav", "")
            file_wvlt_out=wvlt_path+hf[j].replace(".wav", "")
            if os.path.isfile(file_wvlt_out) and os.path.isfile(file_spec_out):
                continue            
                
            NFR=int(data.shape[0]*1000/(FS*SNIP_LEN))
            WV_FRAME_SIZE=int(data.shape[0]/NFR)
            OVRLP=0.5
            SHIFT=int(WV_FRAME_SIZE*OVRLP)
            NBF=64
            TIME_STEPS=256
            DIM=(TIME_STEPS,NBF)
                
            init=0
            endi=WV_FRAME_SIZE
            wv_mat=np.zeros((1,NBF,TIME_STEPS),dtype=np.float32)
            
            for k in range(NFR):    
                frame=data[init:endi]                         
                init=init+int(SHIFT)
                endi=endi+int(SHIFT)
                cwtmatr,_ = pywt.cwt(frame, np.arange(1,NBF+1), 'morl')

                bicubic_img = cv2.resize(np.real(cwtmatr),DIM,interpolation=cv2.INTER_CUBIC)
                
                #Looking for min/max coefficients for standardization.
                max_curr = np.max(bicubic_img)
                min_curr = np.min(bicubic_img)
                if max_curr > maxWvlt_en:
                    maxWvlt_en = max_curr
                if min_curr < minWvlt_en:
                    minWvlt_en = min_curr    
                
                wv_mat[0,:,:]=bicubic_img
                np.save(file_wvlt_out+"_"+str(k)+".npy",wv_mat)
    
            init=0
            num_samples=int(FRAME_SIZE*FS)
            endi=num_samples
            
#             #Create wavelet basis and get necessary dimensions to scale to 128 time steps
#             [freqs,psi,phi]=create_wavelets(num_samples,nbf=NBF,dil=DIL)
            
            
            nf=int(len(data)/(TIME_SHIFT*FS))-1
            if nf>0:
                mat=np.zeros((1,NMELS,126), dtype=np.float32)
                for k in range(nf):
                    try:
                        frame=data[init:endi]
                        imag=melspectrogram(frame, sr=FS, n_fft=NFFT, hop_length=HOP, n_mels=NMELS, fmax=FS/2)

                        init=init+int(TIME_SHIFT*FS)
                        endi=endi+int(TIME_SHIFT*FS)
                        if np.min(np.min(imag))<=0:
                            countinf+=1
                            continue

                        imag=np.log(imag, dtype=np.float32)
                        mat[0,:,:]=imag
                        np.save(file_spec_out+"_"+str(k)+".npy",mat)

                        max_curr = np.max(imag)
                        min_curr = np.min(imag)
                        if max_curr > maxSpec_en:
                            maxSpec_en = max_curr
                        if min_curr < minSpec_en:
                            minSpec_en = min_curr    

                    except:
                        init=init+int(TIME_SHIFT*FS)
                        endi=endi+int(TIME_SHIFT*FS)
                        countinf+=1

            else:
                print("WARNING, audio too short", hf[j], len(data))
                countbad+=1
                
    enrgy['Min wvlt Scale'].append(minWvlt_en)
    enrgy['Max wvlt Scale'].append(maxWvlt_en)
    enrgy['Min spec Scale'].append(minSpec_en)
    enrgy['Max spec Scale'].append(maxSpec_en)
    df = pd.DataFrame(data=enrgy)
    df.to_csv(PATH+'/scales.csv')

    print(countbad)
    print(countinf)




        
        
