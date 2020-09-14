import numpy as np
import scipy
import librosa
import torch
from librosa.feature import melspectrogram
import warnings
warnings.filterwarnings("ignore", message="WavFileWarning: Chunk (non-data) not understood, skipping it.")

def standard(tensor,minval,maxval):
    temp=tensor-minval
    return temp/(maxval-minval)


def compute_spectrograms(signal):
    MIN_SCALER=-41.0749297
    MAX_SCALER=6.720702
    """
    Compute the tensor of Mel-scale spectrograms to be used as input for the autoencoders from a wav file
    :param wav_file: *.wav file with a sampling frequency of 16kHz
    :returns: Pytorch tensor (N, C, F, T). N: batch of spectrograms extracted every 500ms, C: number of channels (1),  F: number of Mel frequencies (128), T: time steps (126)
    """        

    NFFT=512
    FRAME_SIZE=0.5
    TIME_SHIFT=0.25
    HOP=64
    NMELS=128
    FS=16000


#         signal = scipy.signal.resample(np.real(sig),FS)

#         signal=signal-np.mean(signal)
#         signal=signal/np.max(np.abs(signal))
    init=0
    endi=int(FRAME_SIZE*FS)
    nf=int(signal.shape[0]/(TIME_SHIFT*FS))-1
    signal=signal.numpy()

    if nf>0:
        mat=torch.zeros((nf,NMELS,126),dtype=torch.float)
        j=0
        for k in range(nf):
#                 try:
            frame=signal[init:endi]
            imag=melspectrogram(frame, sr=FS, n_fft=NFFT, hop_length=HOP, n_mels=NMELS, fmax=FS/2)
            init=init+int(TIME_SHIFT*FS)
            endi=endi+int(TIME_SHIFT*FS)
            if np.min(np.min(imag))<=0:
                warnings.warn("There is Inf values in the Mel spectrogram")
                continue
            imag=np.log(imag, dtype=np.float32)
            imagt=torch.from_numpy(imag)
            mat[k,:,:]=imagt
            j+=1
#                 except:
#                     init=init+int(TIME_SHIFT*FS)
#                     endi=endi+int(TIME_SHIFT*FS)
#                     warnings.warn("There is non valid values in the wav file")
    else:
        raise ValueError("WAV file is too short to compute the Mel spectrogram tensor")

#         mat=torch.squeeze(mat,dim=1)
    return standard(torch.reshape(mat,(NMELS,nf*126)),MIN_SCALER,MAX_SCALER)
    #         return mat[0:j,:,:,:]
