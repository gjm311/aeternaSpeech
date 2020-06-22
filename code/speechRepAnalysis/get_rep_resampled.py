import sys
import os

from scipy.io.wavfile import read
import scipy
import numpy as np

from librosa.feature import melspectrogram

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    if len(sys.argv)!=3:
        print("python get_spec_full.py <path_audios>")
        sys.exit()
   
    PATH = os.path.dirname(os.path.abspath(__file__))
    PATH_AUDIO=PATH+sys.argv[2]
    PATH_SPEC=PATH_AUDIO+"/../images/spec/"
    PATH_WVLT=PATH_AUDIO+"/../images/wvlt/"
    
    FS=16000
    NFFT=512
    FRAME_SIZE=0.5
    TIME_SHIFT=0.25
    HOP=64
    NMELS=128
    SAMPLE_PERIOD = 3200
    NUM_BANDS = SAMPLE_PERIOD//HOP
    
    hf=os.listdir(PATH_AUDIO)
    hf.sort()
    if len(hf) == 0:
        print(PATH_AUDIO+ " is empty...", len(hf))
        sys.exit()
    else:
        print(PATH_AUDIO, len(hf))
        
    if not os.path.exists(PATH_SPEC):
        os.makedirs(PATH_SPEC)
    if not os.path.exists(PATH_WVLT):
        os.makedirs(PATH_WVLT)
        
    countbad=0
    countinf=0
    for j in range(len(hf)):
        print("Procesing audio", j, hf[j], len(hf))
        fs_in, signal=read(PATH_AUDIO+hf[j])
        data = scipy.signal.resample(signal,FS)
        if len(data.shape)>1:
            continue
        data=data-np.mean(data)
        data=data/np.max(np.abs(data))
        file_spec_out=PATH_SPEC+hf[j].replace(".wav", "")
        file_wvlt_out=PATH_WVLT+hf[j].replace(".wav", "")
        if os.path.isfile(file_wvlt_out) and os.path.isfile(file_spec_out):
            continue
        
        init=0
        endi=int(FRAME_SIZE*FS)
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
                except:
                    init=init+int(TIME_SHIFT*FS)
                    endi=endi+int(TIME_SHIFT*FS)
                    countinf+=1

        else:
            print("WARNING, audio too short", hf[j], len(data))
            countbad+=1
            
        mat = torch.zeros(1,1,NUM_BANDS,FS)
        scales =  np.arange(1, SAMPLE_PERIOD, HOP)*pywt.central_frequency('morl')
        coefs, freqs = pywt.cwt(data, scales, 'morl')
        coefs=np.log(coefs, dtype=np.float32)
        np.save(file_wvlt_out+"_.npy",coefs)

        
    print(countbad)
    print(countinf)
        



        
        