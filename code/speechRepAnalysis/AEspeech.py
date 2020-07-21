
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
from wvCAE import wvCAEn
from scipy.io.wavfile import read
import scipy
import torch
import librosa
from librosa.feature import melspectrogram
import pywt
import scaleogram as scg
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="WavFileWarning: Chunk (non-data) not understood, skipping it.")

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

CBAR_DEFAULTS = {
    'vertical'   : { 'aspect':30, 'pad':0.03, 'fraction':0.05 },
    'horizontal' : { 'aspect':40, 'pad':0.12, 'fraction':0.05 }
}

COI_DEFAULTS = {
    'alpha':'0.5',
    'hatch':'/',
}



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
#         try:
        scalers = pd.read_csv(self.PATH+"/scales.csv")
        self.min_scaler = float(scalers['Min Scale']) #MIN value of total energy.
        self.max_scaler = float(scalers['Max Scale'])  #MAX value of total energy.
#         except:
#             print("Scalers not found..")
            
        self.fs=fs
        self.nmels=nmels
        self.waveletype = waveletype
        
        pt_path = self.PATH+"/pts/"+rep+"/"
        if not os.path.isdir(pt_path):                          
            print("Inputs are wrong or 'pts' directory is incorect...")
        
        if model=="CAE":
            if rep=='spec':
                self.AE=CAEn(units)
                if torch.cuda.is_available():
                    self.AE.load_state_dict(torch.load(pt_path+"/"+str(units)+'_CAE.pt'))
                    self.AE.cuda()
                else:
                    self.AE.load_state_dict(torch.load(pt_path+"/"+str(units)+'_CAE.pt', map_location='cpu'))
            elif rep=='wvlt':
                self.AE=wvCAEn(units)
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
#         signal = scipy.signal.resample(signal,self.fs)
        
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

    
    def compute_cwt(self, wav_file,volta=1):
        """
        Compute the continuous wavelet transform to be used as input for the autoencoders from a wav file
        :param wav_file: *.wav file with a sampling frequency of 16kHz
        :returns: Pytorch tensor (N, C, F, T). N: one file per method call C: number of channels (1),  F: number of bands, T: Number of Samples
        """
        
        SAMPLE_PERIOD = 3200
        HOP = 50
        NUM_BANDS = SAMPLE_PERIOD//HOP
        self.fs = 16000
        
        if wav_file.find('.wav')==-1 and wav_file.find('.WAV')==-1:
            raise ValueError(wav_file+" is not a valid audio file")
        fs_in, signal=read(wav_file)
#         signal = scipy.signal.resample(signal,self.fs)
        signal_new = signal - np.mean(signal)
        signal_new = signal_new/np.max(np.abs(signal_new))
        time=np.arange(self.fs)

        # range of scales to perform the transform
        mat = torch.zeros(1,1,NUM_BANDS,self.fs)
        
        scales =  np.arange(1, SAMPLE_PERIOD, HOP)*pywt.central_frequency(self.waveletype)
        cwt= scg.CWT(time,signal_new,scales)
        coefs=cwt.coefs
        freqs=cwt.scales_freq
        coefs= np.log(np.abs(coefs))
        mat[0,0,:,:]=torch.from_numpy(np.abs(coefs))
        if volta==1:
            return mat
        else:
            return coefs,freqs

    
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
#             mat=spectrograms.data[k,0,:,:]
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


    def show_scalogram(self, time, coefs=None, freqs=None, spectrum='amp', ax=None, cscale='linear', cmap='jet', clim=None,
                      cbar='vertical', cbarlabel=None,cbarkw=None,figsize=(10, 4.0),yaxis='period', xlim=None,
                      ylim=None, yscale='log',coi=False, ylabel="Period", xlabel="Time (s)", title=None):

        SAMPLE_PERIOD = 3200
        dt=time[1]-time[0]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # adjust y axis ticks
        scales_period = 1./freqs  # needed also for COI mask
        xmesh = np.append(time,time[-1]+dt)
        if yaxis == 'period':
            ymesh = np.concatenate([scales_period, [scales_period[-1]+dt]])
            ylim  = ymesh[[-1,0]] if ylim is None else ylim
            ax.set_ylabel("Period" if ylabel is None else ylabel)
        elif yaxis == 'frequency':
            df    = freqs[-1]/freqs[-2]
            ymesh = np.concatenate([freqs, [freqs[-1]*df]])
            # set a useful yscale default: the scale freqs appears evenly in logscale
            yscale = 'log' if yscale is None else yscale
            ylim   = ymesh[[-1, 0]] if ylim is None else ylim
            ax.set_ylabel("Frequency" if ylabel is None else ylabel)
            #ax.invert_yaxis()
        else:
            raise ValueError("yaxis must be one of 'frequency' or 'period', found "
                              + str(yaxis)+" instead")

        # limit of visual range
        xr = [time.min(), time.max()]
        if xlim is None:
            xlim = xr
        else:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # adjust logarithmic scales on request (set automatically in Frequency mode)
        if yscale is not None:
            ax.set_yscale(yscale)

        # choose the correct spectrum display function and name
        if spectrum == 'amp':
            values = np.abs(coefs)
            sp_title = "Amplitude"
            cbarlabel= "abs(CWT)" if cbarlabel is None else cbarlabel
        elif spectrum == 'real':
            values = np.real(coefs)
            sp_title = "Real"
            cbarlabel= "real(CWT)" if cbarlabel is None else cbarlabel
        elif spectrum == 'imag':
            values = np.imag(coefs)
            sp_title = "Imaginary"
            cbarlabel= "imaginary(CWT)" if cbarlabel is None else cbarlabel
        elif spectrum == 'power':
            sp_title = "Power"
            cbarlabel= "abs(CWT)$^2$" if cbarlabel is None else cbarlabel
            values = np.power(np.abs(coefs),2)
        elif hasattr(spectrum, '__call__'):
            sp_title = "Custom"
            values = spectrum(coefs)
        else:
            raise ValueError("The spectrum parameter must be one of 'amp', 'real', 'imag',"+
                             "'power' or a lambda() expression")

        # labels and titles
        ax.set_title("Continuous Wavelet Transform "+sp_title+" Spectrum"
                     if title is None else title)
        ax.set_xlabel("Time/spatial domain" if xlabel is None else xlabel )
        if max(time)/FS>1:
            ax.set_xticklabels(map(str, np.linspace(0,max(time)/FS,6, dtype=np.int)))
        else:
            ax.set_xticklabels(map(str, np.linspace(0,(max(time)/FS)*1000,6, dtype=np.int)))
            ax.set_xlabel('Time (ms)')

        if cscale == 'log':
            isvalid = (values > 0)
            cnorm = LogNorm(values[isvalid].min(), values[isvalid].max())
        elif cscale == 'linear':
            cnorm = None
        else:
            raise ValueError("Color bar cscale should be 'linear' or 'log', got:"+
                             str(cscale))

        # plot the 2D spectrum using a pcolormesh to specify the correct Y axis
        # location at each scale
        qmesh = ax.pcolormesh(xmesh, ymesh, values, cmap=cmap, norm=cnorm)

        if clim:
            qmesh.set_clim(*clim)

        # fill visually the Cone Of Influence
        # (locations subject to invalid coefficients near the borders of data)
        if coi:
            # convert the wavelet scales frequency into time domain periodicity
            scales_coi = scales_period
            max_coi  = scales_coi[-1]

            # produce the line and the curve delimiting the COI masked area
            mid = int(len(xmesh)/2)
            time0 = np.abs(xmesh[0:mid+1]-xmesh[0])
            ymask = np.zeros(len(xmesh), dtype=np.float16)
            ymhalf= ymask[0:mid+1]  # compute the left part of the mask
            ws    = np.argsort(scales_period) # ensure np.interp() works
            minscale, maxscale = sorted(ax.get_ylim())
            if yaxis == 'period':
                ymhalf[:] = np.interp(time0,
                      scales_period[ws], scales_coi[ws])
                yborder = np.zeros(len(xmesh)) + maxscale
                ymhalf[time0 > max_coi]   = maxscale
            elif yaxis == 'frequency':
                ymhalf[:] = np.interp(time0,
                      scales_period[ws], 1./scales_coi[ws])
                yborder = np.zeros(len(xmesh)) + minscale
                ymhalf[time0 > max_coi]   = minscale
            elif yaxis == 'scale':
                ymhalf[:] = np.interp(time0, scales_coi, scales)
                yborder = np.zeros(len(xmesh)) + maxscale
                ymhalf[time0 > max_coi]   = maxscale
            else:
                raise ValueError("yaxis="+str(yaxis))

            # complete the right part of the mask by symmetry
            ymask[-mid:] = ymhalf[0:mid][::-1]

            # plot the mask and forward user parameters
            plt.plot(xmesh, ymask)
            coikw = COI_DEFAULTS if coikw is None else coikw
            ax.fill_between(xmesh, yborder, ymask, **coikw )

        # color bar stuff
        if cbar:
            cbarkw   = CBAR_DEFAULTS[cbar] if cbarkw is None else cbarkw
            colorbar = plt.colorbar(qmesh, orientation=cbar, ax=ax, **cbarkw)
            if cbarlabel:
                colorbar.set_label(cbarlabel)

        return ax

            
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

        
    def mel2speech(self,spec_signal):
        """
        Return speech file given a tensor to numpy version of a spectrogram. Input expected is of shape: 
        (N, C, F, T). N: batch of spectrograms extracted every 500ms, C: number of channels (1),  F: number of Mel frequencies (128), T: time steps (126)
        
        :spec_signal: tensor of shape as described above.
        :return: numpy array of the speech represetatiion  of the input spectrogam
        """
        FS=16000
        NFFT=512  
        HOP=64

        spec_signal=np.squeeze(spec_signal,axis=1)
        speech_signal=np.array([])
        num_batches=np.shape(spec_signal)[0]

        for k in range(num_batches):
            if k==0:
                speech_signal=librosa.feature.inverse.mel_to_audio((spec_signal[k,:,:]),sr=FS,n_fft=NFFT, hop_length=HOP)
            else:
                np.append(speech_signal,librosa.feature.inverse.mel_to_audio((spec_signal[k,:,:]),sr=FS,n_fft=NFFT, hop_length=HOP))

        return speech_signal
