import sys
import os
import numpy as np
import torch
#import pandas as pd
#from apex.apex.amp import amp
from apex import amp
from denoiser import Denoiser


if __name__ == "__main__":
    
    if len(sys.argv)!=2:
        print("python get_rep.py <path_rep>")
        sys.exit()
#     "/../tedx_spanish_corpus/reps/spec/test/"
 
    PATH=os.path.dirname(os.path.abspath(__file__))
    waveglow_path=PATH+"/checkpoints/waveglow_9000"
    output_dir=PATH+"/audio_outs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    
    if sys.argv[1][0] != '/':
        sys.argv[1] = '/'+sys.argv[1]
        
    if sys.argv[1][-1] != "/":
        sys.argv[1] = sys.argv[1]+'/'
        
    PATH_SPEC=PATH+sys.argv[1]
    SIGMA=0.6
    DENOISER_STRENGTH=0.1
    SCALERS = pd.read_csv(PATH+"/../scales.csv")
#     MIN_SCALER=float(scalers['Min Scale']) #MIN value of total energy.
    MAX_SCALER=float(SCALERS['Max spec Scale'])  #MAX value of total energy.
    FS=16000
    
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    
    waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")
    denoiser = Denoiser(waveglow).cuda()
    
    hf=np.asarray([name for name in os.listdir(PATH_SPEC) if '.npy' in name])
    spk_ids=[int(h.split('_')[2]) for h in hf]
    spk_ids_niq=np.unique(spk_ids)
    spk_idxs=[np.where(spk_ids==niq) for niq in spk_ids_niq]
    
    for itr,spk in enumerate(spk_ids_niq):
        #Get subset of speakers 
        spk_sub=hf[spk_idxs[itr]]
        spch_ids=[int(sp.split('_')[4]) for sp in spk_sub]
        spch_ids_niq=np.unique(spch_ids)
        spch_idxs=[np.where(spch_ids==niq) for niq in spch_ids_niq]
        
        for spch in range(len(spch_idxs)):
            spch_sub=spk_sub[spch_idxs[spch]]
            
            for window in spch_sub:
                mel_path=PATH_SPEC+window
                mel=torch.from_numpy(np.load(mel_path))
                
                mel = torch.autograd.Variable(mel.cuda())
                mel = torch.unsqueeze(mel, 0)
                mel = mel.half() 
                with torch.no_grad():
                    audio = waveglow.infer(mel, sigma=SIGMA)
                    if denoiser_strength > 0:
                        audio = denoiser(audio, DENOISER_STRENGTH)
                    audio = audio * MAX_SCALER
                    
                audio = audio.squeeze()
                audio = audio.cpu().numpy()
                audio = audio.astype('int16')
                audio_path = os.path.join(
                    output_dir, "{}_synthesis.wav".format(window))
                
                write(audio_path, FS, audio)
                print(audio_path)
                break
                
                
        
    
