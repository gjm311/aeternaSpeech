import sys
import os
import numpy as np

PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH+"/toolbox/")
import traintestsplit as tts
import matplotlib.pyplot as plt
import shutil

if __name__ == "__main__":
    
    if len(sys.argv)!=2:
        print("python get_rep.py <path_image>")
        sys.exit()
#     "/tedx_spanish_corpus/images/spec/"

    PATH_SPEC=PATH+sys.argv[1]
#     PATH_SPEC=PATH+"/../images/spec/"
#     PATH_WVLT=PATH_AUDIO+"/../images/wvlt/"
    

    split=tts.trainTestSplit(PATH_SPEC,file_type='.wav', tst_perc=0.2)
#     split.fileTrTstSplit()
    train,test=split.fetchNames()
    if not os.path.exists(PATH_SPEC+'/train/'):       
        os.mkdir(PATH_SPEC+'/train/')
    
    if not os.path.exists(PATH_SPEC+'/test/'):
        os.mkdir(PATH_SPEC+'/test/')
#         os.mkdir(PATH_WVLT+'/train/')
#         os.mkdir(PATH_WVLT+'/test/')
        
    spcs=os.listdir(PATH_SPEC)
#     wvs=os.listdir(PATH_WVLT)
    spcs.sort()
#     wvs.sort()
    files=[name for name in spcs if os.path.isfile(PATH_SPEC+'/'+name)]
#     wvlts=[name for name in wvs if os.path.isfile(PATH_WVLT+'/'+name)]
    
    for file in files:
        file_id=file.split('.')[0]
        file_id='_'.join(file_id.split('_')[:-1])            
        for tr in train:
            trId_curr=tr.split('.')[0]
            if 'npy' not in tr:
                next
            if trId_curr==file_id:
                shutil.move(PATH_SPEC+file, PATH_SPEC+'/train/'+file)
                continue

        for tst in test:
            tstId_curr=tst.split('.')[0]
            if 'npy' not in tst:
                next
            if tstId_curr==file_id:
                shutil.move(PATH_SPEC+file, PATH_SPEC+'/test/'+file)
                continue

    