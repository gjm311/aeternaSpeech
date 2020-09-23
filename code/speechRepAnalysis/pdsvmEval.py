import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb
# import xgboost as xgb
from AEspeech import AEspeech
from scipy.stats import kurtosis, skew
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

MODELS=["CAE","RAE","ALL"]
REPS=['spec','wvlt']    
UNITS=256
UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']
# UTTERS=['pataka']
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
       
    
# define scoring function 
# def custom_auc(ground_truth, predictions):
#     fpr, tpr, _ = roc_curve(ground_truth, predictions, pos_label=1)    
#     return auc(fpr, tpr)
    
    
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
        
    save_path=PATH+"/pdSpanish/classResults/svm/"
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
#     num_feats=256
    
    scores=[]
    results=pd.DataFrame({utter:{'train_acc':0,'test_acc':0,'bin_class':{},'class_report':{}} for utter in UTTERS})
    
    for uIdx,utter in enumerate(UTTERS):
        curr_best=0
        pd_path=PATH+sys.argv[3]+'/'+utter+"/pd/"
        hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/"   
        pdNames=[name for name in os.listdir(pd_path) if '.wav' in name]
        hcNames=[name for name in os.listdir(hc_path) if '.wav' in name]
        pdNames.sort()
        hcNames.sort()
        spks=pdNames+hcNames
        num_spks=len(spks)
        num_pd=len(pdNames)
        num_hc=len(hcNames)
        pdFeats=getFeats(model,UNITS,rep,pd_path,utter,'pd')
        hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
        pdAll=np.unique(pdFeats['wav_file'])
        hcAll=np.unique(hcFeats['wav_file'])
        pdIds=np.arange(50)
        hcIds=np.arange(50,100)
        pds=np.zeros((len(pdAll),num_feats,4))
        hcs=np.zeros((len(hcAll),num_feats,4))
        #getting bottle neck features and reconstruction error for training
        for ii,tr in enumerate(pdAll):
            tritr=pdIds[ii]
            pdTrBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTrErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[tritr])]
            pdTr=np.concatenate((pdTrBns,pdTrErrs),axis=1)
            pds[ii,:,:]=np.array([np.mean(pdTr,axis=0),np.std(pdTr,axis=0),skew(pdTr,axis=0),kurtosis(pdTr,axis=0)]).T
        for ii,tr in enumerate(hcAll):
            tritr=hcIds[ii]
            hcTrBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTrErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[tritr])]
            hcTr=np.concatenate((hcTrBns,hcTrErrs),axis=1)
            hcs[ii,:,:]=np.array([np.mean(hcTr,axis=0),np.std(hcTr,axis=0),skew(hcTr,axis=0),kurtosis(hcTr,axis=0)]).T

        pdXAll=np.reshape(pds,(pds.shape[0],num_feats*4))
        hcXAll=np.reshape(hcs,(hcs.shape[0],num_feats*4))  
        xAll=np.concatenate((pdXAll,hcXAll),axis=0)
        st_xAll=StandardScaler().fit_transform(pd.DataFrame(xAll))
        
        
        pca = PCA(n_components=min(xAll.shape[0],xAll.shape[1]))
        pca.fit_transform(st_xAll)
        variance = pca.explained_variance_ratio_ #calculate variance ratios
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
        ncs=np.count_nonzero(var>90)
        pca = PCA(n_components=ncs)
        pca_xAll=pca.fit_transform(st_xAll)
        
        #split data into training and test with multiple iterations (90 training, 10 test per iter and evenly split PD:HC)
        pd_files=pdNames
        hc_files=hcNames
        num_pdHc_tests=4 #must be even (same # of test pds and hcs per iter)
        for itr in range(int(num_spks/num_pdHc_tests)):
            rand_range=np.arange(num_spks)
            random.shuffle(rand_range)
            
            pdCurrs=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),int(num_pdHc_tests/2))]
            hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),int(num_pdHc_tests/2))]
            pd_files=[pd for pd in pd_files if pd not in pdCurrs]
            hc_files=[hc for hc in hc_files if hc not in hcCurrs]

            pdIds=[spks.index(pdCurr) for pdCurr in pdCurrs]
            hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]
        
#             testDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}
            pdTest=np.zeros((num_pdHc_tests,ncs))
            hcTest=np.zeros((num_pdHc_tests,ncs))
            for ii,pdItr in enumerate(pdIds):
                pdTest[ii,:]=pca_xAll[pdItr,:]
            for ii,hcItr in enumerate(hcIds):
                hcTest[ii,:]=pca_xAll[hcItr,:]
                
            pdTrainees=[spk for idx,spk in enumerate(pdNames) if spk not in pdCurrs]
            hcTrainees=[spk for idx,spk in enumerate(hcNames) if spk not in hcCurrs]
            pdTrainIds=[spks.index(tr) for tr in pdTrainees]
            hcTrainIds=[spks.index(tr) for tr in hcTrainees]
            
            pdTrain=np.zeros((num_pd-int(num_pdHc_tests/2),ncs))
            hcTrain=np.zeros((num_hc-int(num_pdHc_tests/2),ncs))
            for ii,pdItr in enumerate(pdTrainIds):
                pdTrain[ii,:]=pca_xAll[pdItr,:]
            for ii,hcItr in enumerate(hcTrainIds):
                pdTrain[ii,:]=pca_xAll[hcItr,:]
            
            xTrain=np.concatenate((pdTrain,hcTrain),axis=0)
            pdYTrain=np.ones((pdTrain.shape[0])).T
            hcYTrain=np.zeros((pdTrain.shape[0])).T
            yTrain=np.concatenate((pdYTrain,hcYTrain),axis=0)
            
            xTest=np.concatenate((pdTest,hcTest),axis=0)
            pdYTest=np.ones((pdTest.shape[0])).T
            hcYTest=np.zeros((pdTest.shape[0])).T
            yTest=np.concatenate((pdYTest,hcYTest),axis=0)
            
            grid=joblib.load(PATH+"/pdSpanish/classResults/svm/params/"+model+'_'+utter+'_'+rep+'Grid.pkl')
#             grid=svm.SVC(C=grid.best_params_['C'],degree=grid.best_params_['degree'],gamma=grid.best_params_['gamma'],
#                                 kernel=grid.best_params_['kernel'], probability=True)
#             grid.fit(xTrain,yTrain)
            
            train_acc=grid.score(xTrain,yTrain)
            test_acc=grid.score(xTest,yTest)
            bin_class=grid.predict_proba(xTest)
#             avg_precision=average_precision_score(yTest, grid.decision_function(xTest))
            class_report=classification_report(yTest,grid.predict(xTest))
            results[utter]['train_acc']+=train_acc*(num_spks-num_pdHc_tests)*.01
            results[utter]['test_acc']+=test_acc*num_pdHc_tests*.01
            results[utter]['class_report'][itr]=class_report  
            for cpi,(pdId,hcId) in enumerate(zip(pdIds,hcIds)):          
                results[utter]['bin_class'][pdId]=bin_class[cpi]     
                results[utter]['bin_class'][hcId]=bin_class[cpi+int(num_pdHc_tests/2)]

    results.to_pickle(save_path+model+'_'+rep+"Results.pkl")
    


