import numpy as np
import os
import pandas as pd
import scipy
import sys
import torch
from tqdm import tqdm
import traintestsplit as tts 
import warnings

sys.path.append(os.path.abspath('../'))
from AEspeech import AEspeech
from scipy.io.wavfile import read


if __name__=="__main__":
    
    PATH=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv)!=2:
        print("python get_spec_full.py <path_audio>")
        sys.exit()

    path_audio=PATH+sys.argv[1]
    
    if path_audio[-1] != '/':
        path_audio = path_audio+'/'

#     path_audio = PATH+'/../../../../../../../cluster/ag61iwyb/tedx_spanish_corpus/speech/'

    if os.path.exists(path_audio+'train/') or os.path.exists(path_audio+'test/'):
        if len(os.listdir(path_audio+'train/')) >= 1 or len(os.listdir(path_audio+'test/')) >= 1:  
            split = tts.trainTestSplit(path_audio)
            split.trTstReset()
            
    FS = 16000
    NMELS = 128
    UNITS = 1024
    MODEL = "CAE"

    d = {'fs': FS, 'Min Scale': [], 'Max Scale': []}
    
    
    aespeech=AEspeech(model=MODEL, units=UNITS, fs=FS, nmels=NMELS) # load the pretrained CAE with 1024 units
    min_en = np.inf
    max_en = -np.inf
    for i,file in enumerate(tqdm(os.listdir(path_audio))):
        if os.path.isfile(path_audio+file):
            wav_file=path_audio+file
            mat_spec=aespeech.compute_spectrograms(wav_file) # compute the decoded spectrograms from the autoencoder
            max_curr = float(torch.max(mat_spec))
            min_curr = float(torch.min(mat_spec))
            if max_curr > max_en:
                max_en = max_curr
            if min_curr < min_en:
                min_en = min_curr            
                    
    d['Min Scale'].append(min_en)
    d['Max Scale'].append(max_en)
                
    df = pd.DataFrame(data=d)
    df.to_csv(PATH+'/../scales.csv')
    
    
           