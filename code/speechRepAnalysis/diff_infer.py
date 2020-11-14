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
import argparse
from scipy.signal import butter, lfilter
from pystoi import stoi
import pdb
from AEspeech import AEspeech

from diffwave.inference import predict as diffwave_predict
from librosa.feature import melspectrogram
from phonetGM2 import Phonet
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


#check similarity of specific phoneme groupings  that are supposedly more apparent for a given representation
def diff_analysis(fs, ori, new, audio_name, list_phonemes, pkl_path):    
    phon=Phonet()
    dic_keys=np.concatenate((['audio_names','STOI'],list_phonemes))
    dic={d:'' for d in dic_keys}
    ori_pkl_path=pkl_path+'/../'+audio_name+'.pkl'
    new_pkl_path=pkl_path+'/'+audio_name+'.pkl'
 
    dic['audio_names']=audio_name
    
    #get STOI (the output is expected to have a monotonic relation with the subjective speech-intelligibility, 
    #where a higher value denotes better intelligible speech).
    stoi_var=stoi(ori, new, FS, extended=False)   
    dic['STOI']=stoi_var
    
    #get probability of different phoneme groupings for original rep and newly constructed representation
    if os.path.isfile(ori_pkl_path):
        ori_wav_df=pd.read_pickle(ori_pkl_path)
    else:
        ori_wav_df=phon.get_phon_wav(audio_file=ori, phonclass="all", feat_file=ori_pkl_path)
    
    if os.path.isfile(ori_pkl_path):
        new_wav_df=pd.read_pickle(new_pkl_path)
    else:
        new_wav_df=phon.get_phon_wav(audio_file=new, phonclass="all", feat_file=new_pkl_path)
    
    #calculate difference across all time instances between original and recon file followed by mean. 
    phoneme_mean_diff=(ori_wav_df.filter(col for col in new_wav_df.columns if col not in ['time','phoneme'])-new_wav_df.filter(col for col in new_wav_df.columns if col not in ['time','phoneme'])).mean()
    #calculate difference in variance between original speech file and recon for all phoneme groupings.
    phoneme_var_diff=ori_wav_df.filter(col for col in new_wav_df.columns if col not in ['time','phoneme']).var()-new_wav_df.filter(col for col in new_wav_df.columns if col not in ['time','phoneme']).var() 
                                                                                                                                   
    for phon in list_phonemes:
        dic[phon]=np.array([phoneme_mean_diff[phon],phoneme_var_diff[phon]])

    series=pd.Series(dic)
    return series


if __name__ == "__main__":
    
    if len(sys.argv)!=4:
        print("python get_rep.py CAE/RAE ori/nb/bb <path_audio>")
        sys.exit()
#    "./tedx_spanish_corpus/speech/train/"
    if sys.argv[3][0] != '/':
        sys.argv[3] = '/'+sys.argv[3]
        
    if sys.argv[3][-1] != "/":
        sys.argv[3] = sys.argv[3]+'/'

    PATH=os.path.dirname(os.path.abspath(__file__)) 
    path_audio=PATH+sys.argv[3]
    save_path=path_audio+"/../diff_recon/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
       
    model=sys.argv[1]
    if model not in ['CAE', 'RAE']:
        print("Invalid model type: "+model+". Please try again..")
        sys.exit()
    
    rep_typ=sys.argv[2]
    if rep_typ not in ['bb', 'nb']:
        print("Invalid repr. type: "+rep_typ+". Please try again..")
        sys.exit()
    
    PATH_PHON_PROBS_PRE=path_audio+"/../../phonet_probs/"+rep_typ+'/pre_synth/'
    PATH_PHON_PROBS_POST=path_audio+"/../../phonet_probs/"+rep_typ+'/post_synth/'
    if not os.path.exists(PATH_PHON_PROBS_PRE):
        os.makedirs(PATH_PHON_PROBS_PRE)
    if not os.path.exists(PATH_PHON_PROBS_POST):
        os.makedirs(PATH_PHON_PROBS_POST)
    paths=[PATH_PHON_PROBS_PRE,PATH_PHON_PROBS_POST]
    
    
    with open("config.json") as f:
        data = f.read()
    config = json.loads(data)
    
    FS=config['general']['FS']
    unit=config['general']['UNITS']
    NFFT=config['mel_spec']['NFFT']
    
    list_phonemes=["a","e","i","o","u",
                        "b","d","f","x","k","l","m","n","p","r","rr","s","t",
                        "L","g","tS","sil"]
    
    
    INTERP_NMELS=config['mel_spec']['INTERP_NMELS']
    TIME_STEPS=config['mel_spec']['TIME_STEPS']
    TIME_SHIFT=config['mel_spec']['TIME_SHIFT']
    FRAME_SIZE=config['mel_spec']['FRAME_SIZE']

    #binary narrowband: 1 yes, 0 no (i.e. broadband)
    if rep_typ=='bb':
        #broadband: higher time resolution, less frequency resolution
        NMELS=config['mel_spec']['BB_NMELS']
        HOP=int(FS*config['mel_spec']['BB_HOP'])#3ms hop (48 SAMPLES)
        WIN_LEN=int(FS*config['mel_spec']['BB_TIME_WINDOW'])#5ms time window (60 SAMPLES)
        min_filter=50
        max_filter=7000
        rep='broadband'
    elif rep_typ=='nb':
        #narrowband: higher frequency resolution, less time resolution
        NMELS=config['mel_spec']['NB_NMELS']
        HOP=int(FS*config['mel_spec']['NB_HOP']) #10ms hop (160 SAMPLES)
        WIN_LEN=int(FS*config['mel_spec']['NB_TIME_WINDOW']) #30ms time window (480 SAMPLES)
        min_filter=300
        max_filter=5400
        rep='narrowband'
        
        
    model_dir=PATH+"/diff_models/"+rep+"/" #'/path/to/model/dir'
    model_files=os.listdir(model_dir).sort()
    pre_synth_model_dir=model_dir+os.listdir(model_dir)[0]
    post_synth_model_dir=model_dir+os.listdir(model_dir)[1]
    synth_models=[pre_synth_model_dir,post_synth_model_dir]
    
    if not os.path.isfile(ori_model_dir) or not os.path.isfile(recon_model_dir):
        print("make sure synthesizer models have been trained for both original and recon spec. rep. types")
        sys.exit()
    
    path_audio=PATH+"/"+sys.argv[3]+"/"
    hf=os.listdir(path_audio)
    hf.sort()
    if len(hf) == 0:
        print(path_audio+ " is empty...", len(hf))
        sys.exit()
    
    df_cols=np.concatenate((['audio_names','STOI'],list_phonemes))
    data_stor=pd.DataFrame(columns=df_cols)

    for itr,h in enumerate(hf):
        wav_file=path_audio+h

        fs_in, signal=read(wav_file)
        sig_len=len(signal)
        
        if fs_in!=FS:
            raise ValueError(str(fs)+" is not a valid sampling frequency")
    
        signal=signal-np.mean(signal)
        signal=signal/np.max(np.abs(signal))
        bp_signal=butter_bandpass_filter(signal,min_filter,max_filter,FS)
        
        
        #compute 'original' mel-spectrogram using librosa/rep params
        imag=melspectrogram(bp_signal, sr=FS, n_fft=NFFT, win_length=WIN_LEN, hop_length=HOP, n_mels=NMELS, fmax=FS//2)
        imag=np.abs(imag)
        imag=np.log(imag, dtype=np.float32)
        spectrogram=torch.from_numpy(imag)
        
        #compute 'reconstructed (recon)' mel-spectrogram using learned auto-encoder params
        aespeech=AEspeech(model=model,units=unit,rep=rep)
        mat,sig_len=aespeech.compute_spectrograms(wav_file, plosives_only=0,volta=1)
        to,bot=aespeech.AE.forward(mat)
        new_to=torch.zeros((to.shape[2],to.shape[0]*to.shape[3]))
        init=0
        endi=int(TIME_STEPS)
        shift=int(TIME_STEPS*(TIME_SHIFT/FRAME_SIZE))
        for fr in range(to.shape[0]):
            new_to[:,init:endi]=to[fr,:,:,:]
            init+=shift
            endi+=shift

        for pitr,pkl_path in enumerate(paths):
#             audio, sample_rate = diffwave_predict(spectrogram.float(), model_dir, device=torch.device('cpu'))
            if pitr==0:
                audio, sample_rate = diffwave_predict(spectrogram.float(), synth_models[pitr])
            else:
                audio, sample_rate = diffwave_predict(new_to.float(), synth_models[pitr])
        
            if len(audio)>len(signal):
                if (len(audio)-len(signal))%2==0:
                    signal=np.pad(signal,int((len(audio)-len(signal))/2))
                else:
                    audio=audio[:-1]
                    signal=np.pad(signal,int(np.floor((len(audio)-len(signal))/2)))
            else:
                if (len(signal)-len(audio))%2==0:
                    audio=np.pad(audio,int((len(signal)-len(audio))/2))
                else:
                    signal=signal[:-1]
                    audio=np.pad(audio,int(np.floor((len(signal)-len(audio))/2)))

            analysis_series=diff_analysis(fs=FS, ori=signal, new=audio, audio_name=h.replace(".wav", ""), list_phonemes=list_phonemes, pkl_path=pkl_path)
            data_stor=data_stor.append(analysis_series,ignore_index=True)
        
            data_stor.to_pickle(save_path+"/synthesisData.pkl")
# #         #IF desire is to save audio, uncomment:
# #         audio.to_pickle(save_path+rep+"_"+os.listdir(path_audio)[itr]+".pkl")


    
    
   