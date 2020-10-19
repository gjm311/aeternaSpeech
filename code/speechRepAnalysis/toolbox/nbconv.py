import numpy as np
import os
import sys
from pydub import AudioSegment
import pdb

if __name__=="__main__":
    PATH=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) != 2:
        print("python nbconv.py <audio path>")
        sys.exit()
    #rep_path: "../tedx_spanish_corpus/speech/"
    
    if sys.argv[1][0] !='/':
        sys.argv[1] = '/'+sys.argv[1]
    if sys.argv[1][-1] !='/':
        sys.argv[1] = sys.argv[1]+'/'
    
    audio_path=PATH+sys.argv[1]
    
    nb_path=audio_path+'/narrowband/'
    if not os.path.isdir(nb_path):
        os.mkdir(nb_path)
        
    #If audio already split into train/test, narrowband conversion output into corrresponding train/test files    
    if os.path.isdir(audio_path+'/train/'):
        if not os.path.isdir(nb_path+'train/'):
            os.mkdir(nb_path+'train/')
        if not os.path.isdir(nb_path+'test/'):
            os.mkdir(nb_path+'test/')

        for t in ['train/','test/']:
            audio_path_curr=audio_path+t
            hf=os.listdir(audio_path_curr)
            for h in hf:
                wav=AudioSegment.from_file(audio_path_curr+h, format='wav')
                wav=wav.set_frame_rate(8000)
                wav.export(nb_path+t+h[:-4]+'.amr', format='amr')
#                 amr=AudioSegment.from_file(nb_path+t+h[:-4]+'.amr', format='amrnb')
# #                         amr=amr.set_frame_rate(5900)
#                 amr.export(nb_path+t+h[:-4]+'.wav', format='wav')
#                 os.remove(nb_path+t+h[:-4]+'.amr')
#                 wav=AudioSegment.from_file(nb_path+t+h[:-4]+'.wav', format='wav')
                wav=wav.set_frame_rate(16000)
                wav.export(nb_path+t+h[:-4]+'.wav', format='wav')
    else:
        audio_path_curr=audio_path
        hf=os.listdir(audio_path)
        for h in hf:
            wav=AudioSegment.from_file(audio_path+h, format='wav')
            wav=wav.set_frame_rate(8000)
            wav.export(nb_path+h[:-4]+'.amr', format='amr')
            amr=AudioSegment.from_file(nb_path+h[:-4]+'.amr', format='amrnb')
            amr.export(nb_path+h[:-4]+'.wav', format='wav')
            os.remove(nb_path+h[:-4]+'.amr')
            wav=AudioSegment.from_file(nb_path+h[:-4]+'.wav', format='wav')
            wav=wav.set_frame_rate(16000)
            wav.export(nb_path+h[:-4]+'.wav', format='wav')

    
    
