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
import pdb
 
sys.path.append("/../")
PATH=os.path.dirname(os.path.abspath(__file__))

class trainTestSplit:
    
    def __init__(self,file_path,file_type='.wav', tst_perc=.2):
        
        self.dir_path=PATH+'/../'
        self.file_path=file_path
        self.file_type=file_type
        self.tst_perc=tst_perc
        self.ids_path=self.file_path+'/trTst_ids'+self.file_type[1:].upper()+'.pkl'
        
        if file_path[-1] != '/':
            self.file_path = file_path+'/'
            
        if self.file_type[0] != '.':
            self.file_type='.'+self.file_type
            
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
            
            hf=[name for name in os.listdir(self.file_path) if self.file_type in name]
            hf.sort()
            num_files=len(hf)
#             if self.file_type=='.npy':
            max_idx=int(hf[-1].split('_')[2])
            idxs=np.arange(max_idx)
#             else:
#                 idxs = np.arange(num_files)
            num_tst = int(np.ceil(max_idx*self.tst_perc))
            num_tr = max_idx-num_tst
            
            random.shuffle(idxs)

            tr_idxs = idxs[0:num_tr]
            tst_idxs = idxs[num_tr:]
    
            self.saveAssignments(hf,tr_idxs,tst_idxs)
            for itr, file in enumerate(os.listdir(self.file_path)):
                if self.file_type in file:
                    itr=int(file.split('_')[2])
#                     print(itr)
                    if itr-1 in tr_idxs:
                        shutil.move(self.file_path+file, self.tr_path+file)
                    elif itr-1 in tst_idxs:
                        shutil.move(self.file_path+file, self.tst_path+file)
                
                
    def imTrTstSplit(self):
        
        train,test=self.fetchNames()
        
        im_path=self.file_path
        
        if not os.path.exists(im_path+'/train/') or not os.path.exists(im_path+'/test/'):
            os.mkdir(im_path+'/train/')
            os.mkdir(im_path+'/test/')          

        files=os.listdir(im_path)
        files.sort()
        files=[name for name in files if os.path.isfile(im_path+'/'+name)]

        for file in files:
            file_id=file.split('.')[0]
            file_id='_'.join(file_id.split('_')[:-1])            
            for tr in train:
                trId_curr=tr.split('.')[0]
                if 'npy' not in tr:
                    next
                if trId_curr==file_id:
                    shutil.move(im_path+file, im_path+'/train/'+file)
                    continue
            
            for tst in test:
                tstId_curr=tst.split('.')[0]
                if 'npy' not in tst:
                    next
                if tstId_curr==file_id:
                    shutil.move(im_path+file, im_path+'/test/'+file)
                    continue
                    
        
    def wavReset(self): 
        if len(os.listdir(self.tr_path))==0 and len(os.listdir(self.tst_path))==0:
            print("Directory is empty. No files to reset...")
        else:
            for file in os.listdir(self.tr_path):
                shutil.move(self.tr_path+file, self.file_path+'/'+file)
            for file in os.listdir(self.tst_path):
                shutil.move(self.tst_path+file, self.file_path+'/'+file)
            os.rmtree(self.tr_path)
            os.rmtree(self.tst_path)
                            
    def saveAssignments(self,hf,tr_ids,tst_ids):
        tr_names=[hf[lnk] for lnk in tr_ids]
        tst_names=[hf[lnk] for lnk in tst_ids]
        self.assignments = {'trIds':tr_ids,'trNames':tr_names,'tstIds':tst_ids,'tstNames':tst_names}        
        with open(self.ids_path, 'wb') as f:
            pickle.dump(self.assignments, f)
        
    def fetchIds(self):
        if not os.path.exists(self.ids_path):
            print("No train/test assignments exist... try saveAssignments()")
        else:
            with open(self.ids_path, 'rb') as f:
                info_dict=pickle.load(f)
            trIds=info_dict['trIds']
            tstIds=info_dict['tstIds']
            return trIds,tstIds
        
    def fetchNames(self):
        if not os.path.exists(self.ids_path):
            print("No train/test assignments exist... try saveAssignments()")
        else:
            with open(self.ids_path, 'rb') as f:
                info_dict=pickle.load(f)   
            trNames=info_dict['trNames']
            tstNames=info_dict['tstNames']
            return trNames,tstNames

        
#     def trTstReset(self): 
#         if not os.path.exists(self.trTstPath):
#             print("No train/test assignments exist... try trTstAssign()")
#         else:
#             os.unlink(self.trTst_path)
            

#     def fetchIds(self):
#         if not os.path.exists(self.trTstPath):
#             print("No train/test assignments exist... try trTstAssign()")
#         else:
#             df=pd.read_csv(self.trTstPath)
#             trIds=df['trIds'].values.tolist()
#             tstIds=df['tstIds'].values.tolist()
#             return trIds,tstIds