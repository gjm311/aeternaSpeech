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
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MODELS=["CAE","RAE","ALL"]
REPS=['spec','wvlt']    
UNITS=256
# UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']
UTTERS=['pataka']
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
        
    save_path=PATH+"/pdSpanish/classResults/svm/params/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    mfda_path=PATH+"/pdSpanish/"
    mfdas=pd.read_csv(mfda_path+"metadata-Spanish_All.csv")['M-FDA'].values
    pd_mfdas=mfdas[0:50]
    hc_mfdas=mfdas[50:]

    if rep=='wvlt':
        num_feats=64+256
    else:
        num_feats=128+256
    comp_range=np.arange(1,5)
    
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
        scores=[]
#         for n_comps in comp_range:
        n_comps=2
        pca = PCA(n_components=n_comps)
        curr_best=0
        pdTrainees=np.unique(pdFeats['wav_file'])
        hcTrainees=np.unique(hcFeats['wav_file'])
        pdTrainIds=np.arange(50)
        hcTrainIds=np.arange(50,100)
        pds=np.zeros((len(pdTrainees),n_comps,4))
        hcs=np.zeros((len(hcTrainees),n_comps,4))
        #getting bottle neck features and reconstruction error for training
        for ii,tr in enumerate(pdTrainees):
            tritr=pdTrainIds[ii]
            pdTrBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTrErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTr=np.concatenate((pdTrBns,pdTrErrs),axis=1)
            st_pdFeats=StandardScaler().fit_transform(pd.DataFrame(pdTr))
            pdPCs=pca.fit_transform(st_pdFeats)
            pds[ii,:,:]=np.array([np.mean(pdPCs,axis=0),np.std(pdPCs,axis=0),skew(pdPCs,axis=0),kurtosis(pdPCs,axis=0)]).T
        for ii,tr in enumerate(hcTrainees):
            tritr=hcTrainIds[ii]
            hcTrBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTrErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTr=np.concatenate((hcTrBns,hcTrErrs),axis=1)
            st_hcFeats=StandardScaler().fit_transform(pd.DataFrame(hcTr))
            hcPCs=pca.fit_transform(st_hcFeats)
            hcs[ii,:,:]=np.array([np.mean(hcPCs,axis=0),np.std(hcPCs,axis=0),skew(hcPCs,axis=0),kurtosis(hcPCs,axis=0)]).T

        pdXTrain=np.reshape(pds,(pds.shape[0]*n_comps,4))
        hcXTrain=np.reshape(hcs,(hcs.shape[0]*n_comps,4))      
        pdYTrain=np.ones((pdXTrain.shape[0])).T
        hcYTrain=np.zeros((hcXTrain.shape[0])).T
        
        xTrain=np.concatenate((pdXTrain,hcXTrain),axis=0)
        yTrain=np.concatenate((pdYTrain,hcYTrain),axis=0)
        where_are_NaNs = np.isnan(xTrain)
        xTrain[where_are_NaNs] = 0
        
#         C_range = np.logspace(1, 5, 20)
#         gamma_range = np.logspace(-6,-1, 20)
#         param_grid = dict(gamma=gamma_range, C=C_range)
        param_grid = [
          {'C':np.logspace(3, 8, 20), 'gamma':np.logspace(-3,5, 20), 'degree':[2],'kernel': ['poly']},
        ]

        cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(xTrain, yTrain)

        scores.append(grid.best_score_)
        if grid.best_score_ > curr_best:
            filename = save_path+model+'_'+utter+'_'+rep+'Grid.pkl'
            with open(filename, 'wb') as file:
                joblib.dump(grid, filename)
                    
#         scores_df=pd.DataFrame({n_c:score for n_c,score in zip(comp_range,scores)},index=np.arange(1))
#         scores_df.to_pickle(save_path+model+'_'+utter+'_'+rep+'GridCompData.pkl')



