#GOAL: Get correlation scores for all speech files (reconstructed vs. original).
#Reconstructed versions are obtained via autoencoders.
# -*- coding: utf-8 -*-


from AEspeech import AEspeech
import os
import sys
from phonetGM2 import Phonet

if __name__=="__main__":

    if len(sys.argv)!=2:
        print("python get_spec_full.py <'spec' or 'wvlt'> <path_speech>")
        sys.exit()
   
    rep=sys.argv[1]
    end_path = sys.argv[2]
    if end_path[0]!='/':
        end_path='/'+end_path
        
    PATH=os.path.dirname(os.path.abspath(__file__))
    path_audio=PATH+end_path
    path_image=path_audio+'../images/'+rep+'/'
    phon = Phonet()
    
    
    #set loop parameters    
    try:
        fss=os.listdir(path_image)
        fss=list(map(int, fss))
    except:
        print("Sampling rate file directory incorrect...")

    models=['CAE','RAE']
    units=2**np.arange(6,12)
    min_files=0
    max_files=len(os.listdir(path_audio))
    save_path=PATH+'/phonCSVs/'+rep+'/'
    
    #loop through different models and possible units
    for mod in models:
        for unit_curr in units:
            
            #for each wav_file, resample (handled in aespeech)
            for j,wav_file in enumerate(os.listdir(path_audio)):
                for fs_curr in fss:
                    save_path_curr=save_path+str(fs_curr)+'/'
                
                    #compute the decoded spectrograms from the autoencoder and standardize or get coeffs for wvlt representation
                    aespeech=AEspeech(model=mod,unit=unit_curr,fs=fs_curr,rep=rep)
                    if rep=='spec':
                        mat=aespeech.compute_spectrograms(wav_file)
                        mat=aespeech.standard(mat)
                    if rep=='wvlt':
                        mat=aespeech.compute_cwt(wav_file)

                    if torch.cuda.is_available():
                        mat=mat.cuda()
                    to,bot=aespeech.AE.forward(mat)
                    to=aespeech.destandard(to)

                    recon=to.data.numpy()
                    ori=mat.data.numpy()
                    
                    reconPath_phonSave=save_path_curr+'/'+str(unit_curr)+'_'+mod+'_recon.csv'
                    oriPath_phonSave=save_path_curr+'/'+str(unit_curr)+'_'+mod+'_original.csv'
                    
                    phon.get_phon_wav(signal=recon,reconPath_phonSave,fs=fs_curr,phonclass="all")
                    phon.get_phon_wav(signal=ori,oriPath_phonSave,fs=fs_curr,phonclass="all")
                    
#                     print("processing file ", j+1, " from ", str(max_files), " ", hf[j])
         
    