import sys
import os
import torch
import pywt
import pandas as pd
import scaleogram as scg 
from scipy.io.wavfile import read
import scipy
import librosa
from librosa.feature import melspectrogram
import numpy as np
import numpy.fft
import cv2
import json
from AEspeech import AEspeech 
import argparse
import torchaudio
from scipy.signal import butter, lfilter
import pdb

from diffwave.inference import predict as diffwave_predict
from librosa.feature import melspectrogram

import toolbox.traintestsplit as tts



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



if __name__ == "__main__":
    
    if len(sys.argv)!=4:
        print("python quick_infer.py <nb-(1) or bb-(0)> <ori or recon> <path_audio>")
        sys.exit()
#    "./tedx_spanish_corpus/speech/train/"
    if sys.argv[3][0] != '/':
        sys.argv[3] = '/'+sys.argv[3]
        
    if sys.argv[3][-1] != "/":
        sys.argv[3] = sys.argv[3]+'/'
        
    if sys.argv[2] not in ['ori','recon']:
        print("python quick_infer.py <nb-(1) or bb-(0)> <ori or recon> <path_audio>")
        sys.exit()
    else:
        ori=sys.argv[2]

    if int(sys.argv[1]) not in [0,1]:
        print("python quick_infer.py <nb-(1) or bb-(0)> <ori or recon> <path_audio>")
        sys.exit()
    else:
        nb=int(sys.argv[1])
    
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    
    FS=config['general']['FS']
    NFFT=config['mel_spec']['NFFT']
    units=config['general']['UNITS']
    TIME_STEPS=config['mel_spec']['TIME_STEPS']
    TIME_SHIFT=config['mel_spec']['TIME_SHIFT']
    FRAME_SIZE=config['mel_spec']['FRAME_SIZE']
    
    #binary narrowband: 1 yes, 0 no (i.e. broadband)
    if nb==0:
        #broadband: higher time resolution, less frequency resolution
        NMELS=config['mel_spec']['BB_NMELS']
        HOP=int(FS*config['mel_spec']['BB_HOP'])#3ms hop (48 SAMPLES)
        WIN_LEN=int(FS*config['mel_spec']['BB_TIME_WINDOW'])#5ms time window (60 SAMPLES)
        min_filter=50
        max_filter=7000
        rep='narrowband'
    elif nb==1:
        #narrowband: higher frequency resolution, less time resolution
        NMELS=config['mel_spec']['NB_NMELS']
        HOP=int(FS*config['mel_spec']['NB_HOP']) #10ms hop (160 SAMPLES)
        WIN_LEN=int(FS*config['mel_spec']['NB_TIME_WINDOW']) #30ms time window (480 SAMPLES)
        min_filter=300
        max_filter=5400
        rep='broadband'
        
    PATH=os.path.dirname(os.path.abspath(__file__)) 
    path_audio=PATH+sys.argv[3]
    save_path=PATH+"/diff_recon/"+rep+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_dir=PATH+"/diff_models/"+rep+"/" #'/path/to/model/dir'
    model_files=os.listdir(model_dir)
    model_files.sort()
    if ori=='ori':
        model_dir=model_dir+model_files[0]
    else:
        model_dir=model_dir+model_files[1]
    
    itr=0
    for iter in range(5):
        
        while '.npy' in os.listdir(path_audio)[itr]:
            itr+=1
            
        wav_file=path_audio+os.listdir(path_audio)[itr]
                
        if ori=='ori':
            fs_in, signal=read(wav_file)
            sig_len=len(signal)
            if fs_in!=FS:
                raise ValueError(str(fs)+" is not a valid sampling frequency")

            signal=signal-np.mean(signal)
            signal=signal/np.max(np.abs(signal))
            signal=butter_bandpass_filter(signal,min_filter,max_filter,FS)
            imag=melspectrogram(signal, sr=FS, n_fft=NFFT, win_length=WIN_LEN, hop_length=HOP, n_mels=NMELS, fmax=FS//2)
            imag=np.abs(imag)
            imag=np.log(imag, dtype=np.float32)
            spectrogram=torch.from_numpy(imag)
        else:
            aespeech=AEspeech(model='CAE',units=units,rep=rep)   
#             if torch.cuda.is_available():
            mat=aespeech.compute_spectrograms(wav_file, plosives_only=0,volta=0)
#             else:
#                 mat=aespeech.compute_spectrograms(wav_file, plosives_only=0,volta=0).float()
            if torch.cuda.is_available():
                mat=mat.cuda()
            to,bot=aespeech.AE.forward(mat)
            to=to.float()
            spectrogram=torch.zeros((to.shape[2],to.shape[0]*to.shape[3]))
            init=0
            endi=int(TIME_STEPS)
            shift=int(TIME_STEPS*(TIME_SHIFT/FRAME_SIZE))
            for fr in range(to.shape[0]):
                spectrogram[:,init:endi]=to[fr,:,:,:]
                init+=shift
                endi+=shift
        
        audio, sample_rate = diffwave_predict(spectrogram.float(), model_dir, ori=ori, rep=rep)
        torchaudio.save(save_path+ori+"_"+os.listdir(path_audio)[iter]+".wav", audio.cpu(),sample_rate=FS)
#         audio, sample_rate = diffwave_predict(spectrogram.float(), model_dir, device=torch.device('cpu'))
