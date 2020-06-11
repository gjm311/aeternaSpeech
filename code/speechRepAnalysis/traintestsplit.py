import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys
import stat
import pandas as pd
import pickle
import numpy as np 
import shutil
import random


class trainTestSplit:
    
    def __init__(self, path_audio, tst_perc=.1):

        self.PATH = os.getcwd()
        self.path_audio = path_audio
        self.tst_perc = tst_perc
        
        if path_audio[-1] != '/':
            self.path_audio = path_audio+'/'
            
        #Make train/test paths for audio files    
        self.makeTrTstPath()
        
        
    def makeTrTstPath(self):
        self.tr_path = self.path_audio+'/train/'
        self.tst_path = self.path_audio+'/test/'
        if not os.path.exists(self.tr_path):
            os.makedirs(self.tr_path)
        if not os.path.exists(self.tst_path):
            os.makedirs(self.tst_path)
        
        
    def audioTrTstSplit(self):
        if len(os.listdir(self.tr_path))>0 or len(os.listdir(self.tst_path))>0:
            print("Files already in train or test folder. Reset folders with trTstReset and try again.")
        else:
            num_audios = len([name for name in os.listdir(self.path_audio) if os.path.isfile(name)])
            num_tst = round(num_audios*self.tst_perc)
            num_tr = num_audios-num_tst
            idxs = np.arange(num_audios)
            random.shuffle(idxs)
            tr_idxs = idxs[0:num_tr]
            tst_idxs = idxs[num_tr:]
            for itr, file in enumerate(os.listdir(self.path_audio)):
                if os.path.isfile(self.path_audio+file):
                    if itr in tr_idxs:
                        shutil.move(self.path_audio+file, self.tr_path+file)
                    elif itr in tst_idxs:
                        shutil.move(self.path_audio+file, self.tst_path+file)
        
    def trTstReset(self): 
        if len(os.listdir(self.tr_path))==0 and len(os.listdir(self.tst_path))==0:
            print("Directory is empty. No files to reset...")
        else:
            for file in os.listdir(self.tr_path):
                shutil.move(self.tr_path+file, self.path_audio+'/'+file)
            for file in os.listdir(self.tst_path):
                shutil.move(self.tst_path+file, self.path_audio+'/'+file)

        
                            
#     def getSpkIDs(self, spktyp = 'pd', trtyp = 'tst'):
#         spkids = {lang:[] for lang in self.lang_utter_dict.keys()}
#         for lang in self.lang_utter_dict:
#             for utter in self.lang_utter_dict[lang]:
#                 audio_path = self.path_audio+lang+"/"+"/"+utter+"/"
#                 if not os.path.exists(audio_path+'pd/') or not os.path.exists(audio_path+'hc/'):
#                     print('Please split tr/tst files with audioTrTestSplit()...')
#                     return
#                 else:
#                     spktyp = '/'+spktyp+'/'
#                     trtyp = '/'+trtyp+'/'
#                     for itr,file in enumerate(os.listdir(audio_path+spktyp+trtyp)):
#                         if lang == 'Czech':
#                             spkids[lang].append(file[:file.find('aDDK')])
#                         else:
#                             id_num = file[:file.find('_')]
#                             if id_num[0] == str(0) and id_num[1] == str(0):
#                                 id_num = id_num[2]
#                             elif id_num[0] == str(0):
#                                 id_num = id_num[1:]
#                             spkids[lang].append(id_num)
#         return spkids