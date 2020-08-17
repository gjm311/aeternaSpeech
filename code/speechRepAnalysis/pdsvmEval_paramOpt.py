import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb

from AEspeech import AEspeech
from scipy.stats import kurtosis, skew
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

MODELS=["CAE","RAE","ALL"]
REPS=['spec','wvlt']    
UNITS=256
UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']
PATH=os.path.dirname(os.path.abspath(__file__))

def saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms (mel-freq) or 50 ms (wvlt) frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
    with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return feat_vecs
            

def getFeats(model,units,rep,wav_path,utter,spk_typ):
    global PATH
    save_path=PATH+"/"+"pdSpanish/feats/"+utter+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
        
    if os.path.isfile(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle'):
        with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle', 'rb') as handle:
            feat_vecs = pickle.load(handle)
    else:
        feat_vecs=saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ)
    
    return feat_vecs
       

if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/'    
    
    if sys.argv[1] in MODELS:
        model=sys.argv[1]
    else:
        print("python pdsvmTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()
    
    if sys.argv[2] in REPS:
        rep=sys.argv[2]
    else:
        print("python pdsvmTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()    
          
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
        
    save_path=PATH+"/pdSpanish/classResults/svm/params"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    n_epochs=25
    if rep=='wvlt':
        num_feats=64+256
    else:
        num_feats=128+256
           
        
    for uIdx,utter in enumerate(UTTERS):
        pd_path=PATH+sys.argv[3]+'/'+utter+"/pd/"
        hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/"   
        pds=[name for name in os.listdir(pd_path) if '.wav' in name]
        hcs=[name for name in os.listdir(hc_path) if '.wav' in name]
        pds.sort()
        hcs.sort()
        spks=pds+hcs
        num_pd=len(pds)
        num_hc=len(hcs)
        pdFeats=getFeats(model,UNITS,rep,pd_path,utter,'pd')
        hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')

        #getting second,third and fourth order stats of bottle neck features and reconstruction errors
        pdTr=np.concatenate((pdFeats['bottleneck'],pdFeats['error']),axis=1)
        pdTrs=np.array([np.mean(pdTr,axis=0),np.std(pdTr,axis=0),skew(pdTr,axis=0),kurtosis(pdTr,axis=0)]).T
        hcTr=np.concatenate((hcFeats['bottleneck'],hcFeats['error']),axis=1)
        hcTrs=np.array([np.mean(hcTr,axis=0),np.std(hcTr,axis=0),skew(hcTr,axis=0),kurtosis(hcTr,axis=0)]).T

        pdYTrain=np.ones((pdTrs.shape[0])).T
        hcYTrain=np.zeros((hcTrs.shape[0])).T
        
        if uIdx==0:
            xTrain=np.concatenate((pdTrs,hcTrs),axis=0)
            yTrain=np.concatenate((pdYTrain,hcYTrain),axis=0)
        else:
            xTrain_curr=np.concatenate((pdTrs,hcTrs),axis=0)
            yTrain_curr=np.concatenate((pdYTrain,hcYTrain),axis=0)
            xTrain=np.concatenate((xTrain,xTrain_curr),axis=0)
            yTrain=np.concatenate((yTrain,yTrain_curr),axis=0)

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(xTrain, yTrain)

#     print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

    joblib.dump(grid, save_path+model+'_'+rep+'Gs_object.pkl')
            
    
    
    
