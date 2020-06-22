#GOAL: Get correlation scores for all speech files (reconstructed vs. original).
#Reconstructed versions are obtained via autoencoders.
# -*- coding: utf-8 -*-


from AEspeech import AEspeech
import os
import sys
from phonetGM2 import Phonet

if __name__=="__main__":

    if len(sys.argv)!=2:
        print("python get_spec_full.py <path_speech>")
        sys.exit()
   
    end_path = sys.argv[1]
    if end_path[0]!='/':
        end_path='/'+end_path
        
    PATH=os.path.dirname(os.path.abspath(__file__))
    path_audio=PATH+end_path
    phon=Phonet()
    
    #set loop parameters
    reps=['spec','wvlt']
    models=['CAE','RAE']
    units=256
    num_files=len(os.listdir(path_audio))
      
    #for each wav_file, resample (handled in aespeech)
    for j,wav_file in enumerate(os.listdir(path_audio)):
        
        #loop through different models and possible units
        for rep in reps:
            save_path=PATH+'/phonCSVs/'+rep+'/'
        
            for mod in models:
                for unit in units:
                
                    #compute the decoded spectrograms from the autoencoder and standardize or get coeffs for wvlt representation
                    aespeech=AEspeech(model=mod,unit=unit,rep=rep)
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
                    reconPath_phonSave=save_path+'/'+str(unit)+'_'+mod+'_recon.csv'
                    oriPath_phonSave=save_path+'/'+str(unit)+'_'+mod+'_original.csv'    
                    phon.get_phon_wav(signal=recon,reconPath_phonSave,phonclass="all")
                    phon.get_phon_wav(signal=ori,oriPath_phonSave,phonclass="all")
                    
                    print("processing file ", j+1, " from ", str(num_files), " ", hf[j])
         
    