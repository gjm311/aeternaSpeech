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
import sys
sys.path.append("/../")


class trainTestSplit:
    
    def __init__(self, file_path, tst_perc=.1):

        self.file_path = file_path
        self.tst_perc = tst_perc
        
        if file_path[-1] != '/':
            self.file_path = file_path+'/'
            
        #Make train/test paths for files    
        self.makeTrTstPath()
        
        
    def makeTrTstPath(self):
        self.tr_path = self.file_path+'/train/'
        self.tst_path = self.file_path+'/test/'
        if not os.path.exists(self.tr_path):
            os.makedirs(self.tr_path)
        if not os.path.exists(self.tst_path):
            os.makedirs(self.tst_path)
        
        
    def fileTrTstSplit(self):
        if len(os.listdir(self.tr_path))>0 or len(os.listdir(self.tst_path))>0:
            print("Files already in train or test folder. Reset folders with trTstReset and try again.")
        else:
            num_files = len([name for name in os.listdir(self.file_path) if os.path.isfile(self.file_path+'/'+name)])
            num_tst = round(num_files*self.tst_perc)
            num_tr = num_files-num_tst
            idxs = np.arange(num_files)
            random.shuffle(idxs)
            tr_idxs = idxs[0:num_tr]
            tst_idxs = idxs[num_tr:]
            for itr, file in enumerate(os.listdir(self.file_path)):
                if os.path.isfile(self.file_path+file):
                    if itr in tr_idxs:
                        shutil.move(self.file_path+file, self.tr_path+file)
                    elif itr in tst_idxs:
                        shutil.move(self.file_path+file, self.tst_path+file)
        
    def trTstReset(self): 
        if len(os.listdir(self.tr_path))==0 and len(os.listdir(self.tst_path))==0:
            print("Directory is empty. No files to reset...")
        else:
            for file in os.listdir(self.tr_path):
                shutil.move(self.tr_path+file, self.file_path+'/'+file)
            for file in os.listdir(self.tst_path):
                shutil.move(self.tst_path+file, self.file_path+'/'+file)

        
                            
#     def getSpkIDs(self, spktyp = 'pd', trtyp = 'tst'):
#         spkids = {lang:[] for lang in self.lang_utter_dict.keys()}
#         for lang in self.lang_utter_dict:
#             for utter in self.lang_utter_dict[lang]:
#                 audio_path = self.file_path+lang+"/"+"/"+utter+"/"
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