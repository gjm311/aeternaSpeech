
# -*- coding: utf-8 -*-
"""
Feature extraction from speech signals based on representation learning strategies
@author: J. C. Vasquez-Correa
        Pattern recognition Lab, University of Erlangen-Nuremberg
        Faculty of Engineering, University of Antioquia,
        juan.vasquez@fau.de
"""

import os
from CAE import CAEn
from RAE import RAEn
from scipy.io.wavfile import read
import scipy
import torch
from librosa.feature import melspectrogram
import pywt
import scaleogram as scg
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="WavFileWarning: Chunk (non-data) not understood, skipping it.")

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

class AEspeech:

    def __init__(self,model,units,fs=16000,rep='spec',nmels=128,waveletype='morl'):
        """
        Feature extraction from speech signals based on representation learning strategies using convolutional and recurrent autoencoders
        :param model: type of autoencoder to extract the features from ('CAE': convolutional autoencoders, 'RAE': recurrent autoencoder)
        :param units: number of hidden neurons in the bottleneck space (64, 128, 256, 512, 1024)
        :returns: AEspeech Object.
        """
        self.model_type=model
        self.units=units
        self.PATH=os.path.dirname(os.path.abspath(__file__))
        try:
            SCALERS = pd.read_csv("scales.csv")
            MIN_SCALER= float(SCALERS['Min Scale']) #MIN value of total energy.
            MAX_SCALER= float(SCALERS['Max Scale'])  #MAX value of total energy.
        except:
            print("Scalers not found..")
        self.fs=fs
        self.nmels=nmels
        self.waveletype = waveletype
        
        pt_path = self.PATH+"/pts/"+rep+"/"+str(fs)+'/'
        if os.path.isdir(pt_path):                          
            continue
        else:
            print("Inputs are wrong or 'pts' directory is incorect...")
            
        if model=="CAE":
            self.AE=CAEn(units)
            if torch.cuda.is_available():
                self.AE.load_state_dict(torch.load(pt_path+"/"+str(units)+'_CAE.pt'))
                self.AE.cuda()
            else:
                self.AE.load_state_dict(torch.load(pt_path+"/"+str(units)+'_CAE.pt', map_location='cpu'))
        elif model=="RAE":
            self.AE=RAEn(units)
            if torch.cuda.is_available():
                self.AE.load_state_dict(torch.load(pt_path+"/"+str(units)+'_RAE.pt'))
                self.AE.cuda()
            else:
                self.AE.load_state_dict(torch.load(pt_path+"/"+str(units)+'_RAE.pt', map_location='cpu'))

        else:
            raise ValueError("Model "+model+" is not valid. Please choose only CAE or RAE")


    
    def compute_spectrograms(self, wav_file):
        """
        Compute the tensor of Mel-scale spectrograms to be used as input for the autoencoders from a wav file
        :param wav_file: *.wav file with a sampling frequency of 16kHz
        :returns: Pytorch tensor (N, C, F, T). N: batch of spectrograms extracted every 500ms, C: number of channels (1),  F: number of Mel frequencies (128), T: time steps (126)
        """        
        
        NFFT=512
        FRAME_SIZE=0.5
        TIME_SHIFT=0.25
        HOP=64
            
        fs_in, signal=read(wav_file)
        signal = scipy.signal.resample(signal,self.fs)
        
        signal=signal-np.mean(signal)
        signal=signal/np.max(np.abs(signal))
        init=0
        endi=int(FRAME_SIZE*self.fs)
        nf=int(len(signal)/(TIME_SHIFT*self.fs))-1
        
        if nf>0:
            mat=torch.zeros(nf,1,self.nmels,126)
            j=0
            for k in range(nf):
                try:
                    frame=signal[init:endi]
                    imag=melspectrogram(frame, sr=self.fs, n_fft=NFFT, hop_length=HOP, n_mels=self.nmels, fmax=self.fs/2)
                    init=init+int(TIME_SHIFT*self.fs)
                    endi=endi+int(TIME_SHIFT*self.fs)
                    if np.min(np.min(imag))<=0:
                        warnings.warn("There is Inf values in the Mel spectrogram")
                        continue
                    imag=np.log(imag, dtype=np.float32)
                    imagt=torch.from_numpy(imag)
                    mat[j,:,:,:]=imagt
                    j+=1
                except:
                    init=init+int(TIME_SHIFT*self.fs)
                    endi=endi+int(TIME_SHIFT*self.fs)
                    warnings.warn("There is non valid values in the wav file")
        else:
            raise ValueError("WAV file is too short to compute the Mel spectrogram tensor")
        
        return mat[0:j,:,:,:]

    
    def compute_cwt(self, wav_file):
        """
        Compute the continuous wavelet transform to be used as input for the autoencoders from a wav file
        :param wav_file: *.wav file with a sampling frequency of 16kHz
        :returns: Pytorch tensor (N, C, F, T). N: batch of spectrograms extracted every 500ms, C: number of channels (1),  F: number of frequency (period) bands (128), T: time steps (126)
        """
        
        SAMPLE_PERIOD = 3200
        HOP = 20
        NUM_BANDS = SAMPLE_PERIOD//HOP
        self.fs = 16000
        
        if wav_file.find('.wav')==-1 and wav_file.find('.WAV')==-1:
            raise ValueError(wav_file+" is not a valid audio file")
        fs_in, signal=read(wav_file)
        signal = scipy.signal.resample(signal,self.fs)
        signal_new = signal_new - np.mean(signal_new)
        signal_new = signal_new/np.max(np.abs(signal_new))

        # range of scales to perform the transform
        mat = torch.zeros(1,1,NUM_BANDS,self.fs)
        scales =  np.arange(1, SAMPLE_PERIOD, HOP)*pywt.central_frequency(self.waveletype)
        coefs, freqs = pywt.cwt(signal_new, scales, self.waveletype)
        coefs=np.log(coefs, dtype=np.float32)
        mat[0,0,:,:]=torch.from_numpy(coefs)
        return mat

    
    def show_spectrograms(self, spectrograms):
        """
        Visualization of the computed tensor of Mel-scale spectrograms to be used as input for the autoencoders from a wav file
        :param spectrograms: tensor of spectrograms obtained from '''compute_spectrograms(wav-file)'''
        """
        mmax=2595*np.log10(1+8000/700)
        m=np.linspace(0,mmax,11)

        f=np.round(700*(10**(m/2595)-1))
        f=f[::-1]
        for k in range(spectrograms.shape[0]):
            fig,  ax=plt.subplots(1, 1)
            fig.set_size_inches(5, 5)
            mat=spectrograms.data.numpy()[k,0,:,:]
            ax.imshow(np.flipud(mat), cmap=plt.cm.viridis, vmax=mat.max())
            ax.set_yticks(np.linspace(0,128,11))
            ax.set_yticklabels(map(str, f))
            ax.set_xticks(np.linspace(0,126,6))
            ax.set_xticklabels(map(str, np.linspace(0,500,6, dtype=np.int)))
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (ms)")
            plt.tight_layout()
            plt.show()

    def showScalogram(self, wav_file):
        """
        Plot the scaleogram (used as input for the autoencoders).
        :param scalogram: tensor of scalogram obatined from '''compute_cwt(wav-file)'''
        """
        SAMPLE_PERIOD = 3200
        HOP = 20
        self.fs = 16000
        
        if wav_file.find('.wav')==-1 and wav_file.find('.WAV')==-1:
            raise ValueError(wav_file+" is not a valid audio file")
        fs_in, signal=read(wav_file)
        
        # choose default wavelet function 
        uns_signal_length = np.shape(signal)[0]
        signal_new = scipy.signal.resample(signal,self.fs)
        signal_new = signal_new - np.mean(signal_new)
        signal_new = signal_new/np.max(np.abs(signal_new))
        signal_new_length = np.shape(signal_new)[0]
        xtix = np.array((np.arange(uns_signal_length)/fs_in))

        # range of scales to perform the transform
        scales = np.arange(1, SAMPLE_PERIOD, HOP)*pywt.central_frequency(self.waveletype)

        ax2 = scg.cws(signal_new[:signal_new_length], scales=scales, figsize=(10, 4.0), yscale = 'log', coi = False, ylabel="Period", xlabel="Time (s)")
#         ax2.set_title=wav_file.split('/')[-1]
        ax2.set_xticks(np.arange(0,self.fs,signal_new_length//np.ceil(xtix[-1])))
        ax2.set_xticklabels(np.round(xtix[::int(uns_signal_length//int(np.ceil(xtix[-1])))],2))

            
    def standard(self, tensor):
        """
        standardize input tensor for the autoencoders
        :param tensor: input tensor for the AEs (N, 128,126)
        :returns:  standardize tensor for the AEs (N, 128,126)
        """
        temp=tensor-self.min_scaler
        temp/(self.max_scaler-self.min_scaler)
        return temp.float()

    def destandard(self, tensor):
        """
        destandardize input tensor from the autoencoders
        :param tensor: standardized input tensor for the AEs (N, 128,126)
        :returns:  destandardized tensor for the AEs (N, 128,126)
        """
        temp=tensor*(self.max_scaler-self.min_scaler)
        return temp+self.min_scaler

    def compute_bottleneck_features(self, wav_file, return_numpy=True):
        """
        Compute the the bottleneck features of the autoencoder
        :param wav_file: *.wav file with a sampling frequency of 16kHz
        :param return_numpy: return the features in a numpy array (True) or a Pytorch tensor (False)
        :returns: Pytorch tensor (nf, h) or numpy array (nf, h) with the extracted features. nf: number of frames, size of the bottleneck space
        """

        mat=self.compute_spectrograms(wav_file)
        mat=self.standard(mat)
        if torch.cuda.is_available():
            mat=mat.cuda()
        to, bot=self.AE.forward(mat)
        if return_numpy:
            return bot.data.numpy()
        else:
            return bot

    def compute_rec_error_features(self, wav_file, return_numpy=True):
        """
        Compute the  reconstruction error features from the autoencoder
        :param wav_file: *.wav file with a given sampling frequency
        :param return_numpy: return the features in a numpy array (True) or a Pytorch tensor (False)
        :returns: Pytorch tensor (nf, 128) or numpy array (nf, 128) with the extracted features. nf: number of frames
        """
        mat=self.compute_spectrograms(wav_file)
        mat=self.standard(mat)
        if torch.cuda.is_available():
            mat=mat.cuda()
        to, bot=self.AE.forward(mat)
        
        to=self.destandard(to)

        mat_error=(mat[:,0,:,:]-to[:,0,:,:])**2
        error=torch.mean(mat_error,2)
        if return_numpy:
            return error.data.numpy()
        else:
            return error



    def compute_rec_spectrogram(self, wav_file, return_numpy=True):
        """
        Compute the  reconstructed spectrogram from the autoencoder
        :param wav_file: *.wav file with a sampling frequency of 16kHz
        :param return_numpy: return the features in a numpy array (True) or a Pytorch tensor (False)
        :returns: Pytorch tensor (N, C, F, T). N: batch of spectrograms extracted every 500ms, C: number of channels (1),  F: number of Mel frequencies (128), T: time steps (126)
        """
        mat=self.compute_spectrograms(wav_file)
        mat=self.standard(mat)
        if torch.cuda.is_available():
            mat=mat.cuda()
        to, bot=self.AE.forward(mat)        
        to=self.destandard(to)

        if return_numpy:
            return to.data.numpy()
        else:
            return to

        
    def compute_dynamic_features(self, wav_directory):
        """
        Compute both the bottleneck and the reconstruction error features from the autoencoder for wav files inside a directory
        :param wav_directory: *.wav file with a sampling frequency of 16kHz
        :return: dictionary with the extracted bottleneck and error features, and with information about which frame coresponds to which wav file in the directory.
        """

        if os.path.isdir(wav_directory):
            hf=os.listdir(wav_directory)
            hf.sort()
        else:
            raise ValueError(wav_directory+" is not a valid directory")

        if wav_directory[-1]!='/':
            wav_directory=wav_directory+"/"

        total_bottle=[]
        total_error=[]
        metadata={"wav_file":[], "frame": [], "bottleneck": [], "error":[]}
        for wav_file in hf:
            bottle=self.compute_bottleneck_features(wav_directory+wav_file, True)
            error=self.compute_rec_error_features(wav_directory+wav_file, True)
            metadata["bottleneck"].append(bottle)
            metadata["error"].append(error)
            nframes=error.shape[0]
            list_wav=np.repeat(wav_file, nframes)
            metadata["wav_file"].append(list_wav)
            frames=np.arange(nframes)
            metadata["frame"].append(frames)

        metadata["bottleneck"]=np.concatenate(metadata["bottleneck"], 0)
        metadata["error"]=np.concatenate(metadata["error"], 0)
        metadata["wav_file"]=np.hstack(metadata["wav_file"])
        metadata["frame"]=np.hstack(metadata["frame"])
        return metadata


    def compute_global_features(self, wav_directory, stack_feat=False):
        """
        Compute global features (1 vector per utterance) both for the bottleneck and the reconstruction error features from the autoencoder for wav files inside a directory 
        :param wav_directory: *.wav file with a sampling frequency of 16kHz
        :param stack_feat: if True, returns also a feature matrix with the stack of the bottleneck and error features
        :return: pandas dataframes with the bottleneck and error features.
        """

        if os.path.isdir(wav_directory):
            hf=os.listdir(wav_directory)
            hf.sort()
        else:
            raise ValueError(wav_directory+" is not a valid directory")

        if wav_directory[-1]!='/':
            wav_directory=wav_directory+"/"

        total_bottle=[]
        total_error=[]
        feat_names_bottle=["bottleneck_"+str(k) for k in range(self.units)]
        feat_names_error=["error_"+str(k) for k in range(128)]

        stat_names=["avg", "std", "skewness", "kurtosis"]

        feat_names_bottle_all=[]
        feat_names_error_all=[]

        for k in stat_names:
            for j in feat_names_bottle:
                feat_names_bottle_all.append(k+"_"+j)
            for j in feat_names_error:
                feat_names_error_all.append(k+"_"+j)

        if stack_feat:
            feat_names_all=feat_names_bottle_all+feat_names_error_all

        metadata={"wav_file":[], "frame": [], "bottleneck": [], "error":[]}
        bottle_feat=np.zeros((len(hf), len(feat_names_bottle_all)))
        error_feat=np.zeros((len(hf), len(feat_names_error_all)))
        
        if stack_feat:
            feat_all=np.zeros((len(hf),len(feat_names_bottle_all)+len(feat_names_error_all) ))

        for i, wav_file in enumerate(hf):
            try:
                bottle=self.compute_bottleneck_features(wav_directory+wav_file, True)
                bottle_feat[i,:]=np.hstack((np.mean(bottle, 0), np.std(bottle, 0), st.skew(bottle, 0), st.kurtosis(bottle, 0)))
                error=self.compute_rec_error_features(wav_directory+wav_file, True)
                error_feat[i,:]=np.hstack((np.mean(error, 0), np.std(error, 0), st.skew(error, 0), st.kurtosis(error, 0)))
            except:
                warnings.warn("ERROR WITH "+wav_file)
                continue

        dict_feat_bottle={}
        dict_feat_bottle["ID"]=hf
        for j in range(bottle_feat.shape[1]):
            dict_feat_bottle[feat_names_bottle_all[j]]=bottle_feat[:,j]

        dict_feat_error={}
        dict_feat_error["ID"]=hf
        for j in range(error_feat.shape[1]):
            dict_feat_error[feat_names_error_all[j]]=error_feat[:,j]

        df1=pd.DataFrame(dict_feat_bottle)
        df2=pd.DataFrame(dict_feat_error)

        if stack_feat:
            feat_all=np.concatenate((bottle_feat, error_feat), axis=1)

            dict_feat_all={}
            dict_feat_all["ID"]=hf
            for j in range(feat_all.shape[1]):
                dict_feat_all[feat_names_all[j]]=feat_all[:,j] 

            df3=pd.DataFrame(dict_feat_all)

            return df1, df2, df3

        else:

            return df1, df2

