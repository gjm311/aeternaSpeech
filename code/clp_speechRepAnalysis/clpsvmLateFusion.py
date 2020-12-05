import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb
import itertools
from AEspeech import AEspeech
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import json
import argparse


PATH=os.path.dirname(os.path.abspath(__file__))
#LOAD CONFIG.JSON INFO
with open("config.json") as f:
    data = f.read()
config = json.loads(data)
UNITS=config['general']['UNITS']
UTTERS=['bola','choza','chuzo','coco','gato','jugo','mano','papa','susi']
MODELS=["CAE","RAE","ALL"]
REPS=['narrowband','broadband','wvlt']


def saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ):
    global UNITS    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
    with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return feat_vecs
            

def getFeats(model,units,rep,wav_path,utter,spk_typ):
    global PATH
    save_path=PATH+"/"+"clpSpanish/feats/"+utter+"/"
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
        print("python clpsvmLateFusion.py <'CAE','RAE', or 'ALL'> <nreps - 2 (nb/bb) or 3 (nb/bb/wvlt)> <clp path>")
        sys.exit()        
    #TRAIN_PATH: './clpSpanish/speech/'    
    
    if sys.argv[1] in MODELS:
        model=sys.argv[1]
    else:
        print("python clpsvmLateFusion.py <'CAE','RAE', or 'ALL'> <nreps - 2 (nb/bb) or 3 (nb/bb/wvlt)> <clp path>")
        sys.exit() 
    
    if int(sys.argv[2]) not in [2,3]:
        print("python clpsvmLateFusion.py <'CAE','RAE', or 'ALL'> <nreps - 2 (nb/bb) or 3 (nb/bb/wvlt)> <clp path>")
        sys.exit()
    else:
        nreps=int(sys.argv[2])
    
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
        
    save_path=PATH+"/clpSpanish/classResults/svm/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    reps=REPS[:nreps]   
    num_utters=len(UTTERS)
    
    mfda_path=PATH+"/clpSpanish/"
    mfdas=pd.read_csv(mfda_path+"metadata-Spanish_All.csv")['M-FDA'].values
    clp_mfdas=mfdas[0:50]
    hc_mfdas=mfdas[50:]

    for nrep,rep in enumerate(reps):
        if rep=='wvlt':
            num_feats=config['wavelet']['NBF']+UNITS
        else:
            num_feats=config['mel_spec']['INTERP_NMELS']+UNITS
        clps=np.zeros((50*num_utters,num_feats,4))
        hcs=np.zeros((50*num_utters,num_feats,4))
        for uIdx,utter in enumerate(UTTERS):
            curr_best=0
            clp_path=PATH+sys.argv[3]+'/'+utter+"/clp/"
            hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/"   
            clpNames=[name for name in os.listdir(clp_path) if '.wav' in name]
            hcNames=[name for name in os.listdir(hc_path) if '.wav' in name]
            clpNames.sort()
            hcNames.sort()
            spks=clpNames+hcNames
            num_spks=len(spks)
            num_clp=len(clpNames)
            num_hc=len(hcNames)
            clpFeats=getFeats(model,UNITS,rep,clp_path,utter,'clp')
            hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
            clpAll=np.unique(clpFeats['wav_file'])
            hcAll=np.unique(hcFeats['wav_file'])
            clpIds=np.arange(50)
            hcIds=np.arange(50,100)

            #getting bottle neck features and reconstruction error for training
            for ii,tr in enumerate(clpAll):
                tritr=clpIds[ii]
                clpTrBns=clpFeats['bottleneck'][np.where(clpFeats['wav_file']==spks[tritr])]
                clpTrBns=np.array([np.mean(clpTrBns,axis=0),np.std(clpTrBns,axis=0),skew(clpTrBns,axis=0),kurtosis(clpTrBns,axis=0)])
                clpTrErrs=clpFeats['error'][np.where(clpFeats['wav_file']==spks[tritr])]
                clpTrErrs=np.array([np.mean(clpTrErrs,axis=0),np.std(clpTrErrs,axis=0),skew(clpTrErrs,axis=0),kurtosis(clpTrErrs,axis=0)])
                clps[(ii*num_utters)+uIdx,:,:]=np.concatenate((clpTrBns,clpTrErrs),axis=1).T
            for ii,tr in enumerate(hcAll):
                tritr=hcIds[ii]
                hcTrBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[tritr])]
                hcTrBns=np.array([np.mean(hcTrBns,axis=0),np.std(hcTrBns,axis=0),skew(hcTrBns,axis=0),kurtosis(hcTrBns,axis=0)])
                hcTrErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[tritr])]
                hcTrErrs=np.array([np.mean(hcTrErrs,axis=0),np.std(hcTrErrs,axis=0),skew(hcTrErrs,axis=0),kurtosis(hcTrErrs,axis=0)])
                hcs[(ii*num_utters)+uIdx,:,:]=np.concatenate((hcTrBns,hcTrErrs),axis=1).T

        clpXAll=np.reshape(clps,(clps.shape[0],num_feats*4))
        hcXAll=np.reshape(hcs,(hcs.shape[0],num_feats*4))  
        xAll=np.concatenate((clpXAll,hcXAll),axis=0)
        st_xAll=StandardScaler().fit_transform(pd.DataFrame(xAll))

        pca = PCA(n_components=min(xAll.shape[0],xAll.shape[1]))
        pca.fit_transform(st_xAll)
        variance = pca.explained_variance_ratio_ #calculate variance ratios
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
        if rep=='narrowband':
            nb_ncs=np.count_nonzero(var<90)
            pca = PCA(n_components=nb_ncs)
            nb_pca_xAll=pca.fit_transform(st_xAll)
        elif rep=='broadband':
            bb_ncs=np.count_nonzero(var<90)
            pca = PCA(n_components=bb_ncs)
            bb_pca_xAll=pca.fit_transform(st_xAll)
        elif rep=='wvlt':
            wvlt_ncs=np.count_nonzero(var<90)
            pca = PCA(n_components=wvlt_ncs)
            wvlt_pca_xAll=pca.fit_transform(st_xAll)
    
    if 'wvlt' in reps:
        rncs=[nb_ncs,bb_ncs,wvlt_ncs]
        pca_xAlls=[nb_pca_xAll,bb_pca_xAll,wvlt_pca_xAll]
    else:
        rncs=[nb_ncs,bb_ncs]
        pca_xAlls=[nb_pca_xAll,bb_pca_xAll]
    
    #split data into training and test with multiple iterations
    num_clpHc_tests=config['svm']['tst_spks']#must be even (same # of test clps and hcs per iter)
    nv=config['svm']['val_size']#number of validation speakers per split#must be even and a divisor of num_spks (same # of test clps and hcs per iter)
    if  np.mod(num_clpHc_tests,2)!=0:
        print("number of test spks must be even...")
        sys.exit()
    if  np.mod(100,num_clpHc_tests)!=0:
        print("number of test spks must be a divisor of 100...")
        sys.exit()
    
    total_itrs=config['svm']['iterations']
    results=pd.DataFrame({'Data':{'train_acc':0,'test_acc':0, 'mFDA_spear_corr':{itr:{idx:{utter:0 for utter in UTTERS} for idx in np.arange(100)} for itr in range(total_itrs)},'bin_class':{itr:{} for itr in range(total_itrs)},'class_report':{itr:{} for itr in range(total_itrs)}}})
    threshes=np.zeros((nreps,total_itrs*int(num_spks/num_clpHc_tests)))
                                  
    for o_itr in range(total_itrs):
        clp_files=clpNames
        hc_files=hcNames
        predictions=pd.DataFrame(index=np.arange(100), columns=['predictions'])
        
        for itr in range(int(num_spks/num_clpHc_tests)):
            clpCurrs=[clp_files[idx] for idx in random.sample(range(0,len(clp_files)),int(num_clpHc_tests/2))]
            hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),int(num_clpHc_tests/2))]
            clp_files=[clp for clp in clp_files if clp not in clpCurrs]
            hc_files=[hc for hc in hc_files if hc not in hcCurrs]

            clpIds=[spks.index(clpCurr) for clpCurr in clpCurrs]
            hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]
                                  
            diffs=np.zeros((nreps,(num_spks-num_clpHc_tests)*num_utters))
            tst_diffs=np.zeros((nreps,(num_clpHc_tests)*num_utters))
            preds=np.zeros((nreps,(num_spks-num_clpHc_tests)*num_utters))
            tst_preds=np.zeros((nreps,(num_clpHc_tests)*num_utters))
            
            for nrep,(pca_xAll,ncs) in enumerate(zip(pca_xAlls,rncs)):
                clpTest=np.zeros(((num_clpHc_tests//2)*num_utters,ncs))
                hcTest=np.zeros(((num_clpHc_tests//2)*num_utters,ncs))
                for ii,clpItr in enumerate(clpIds):
                    clpTest[ii*num_utters:(ii+1)*num_utters,:]=pca_xAll[clpItr*num_utters:(clpItr+1)*num_utters,:]
                for ii,hcItr in enumerate(hcIds):
                    hcTest[ii*num_utters:(ii+1)*num_utters,:]=pca_xAll[hcItr*num_utters:(hcItr+1)*num_utters,:]

                clpTrainees=[spk for idx,spk in enumerate(clpNames) if spk not in clpCurrs]
                hcTrainees=[spk for idx,spk in enumerate(hcNames) if spk not in hcCurrs]
                clpTrainIds=[spks.index(tr) for tr in clpTrainees]
                hcTrainIds=[spks.index(tr) for tr in hcTrainees]

                clpTrain=np.zeros(((num_clp-int(num_clpHc_tests/2))*num_utters,ncs))
                hcTrain=np.zeros(((num_hc-int(num_clpHc_tests/2))*num_utters,ncs))
                for ii,clpItr in enumerate(clpTrainIds):
                    clpTrain[ii*num_utters:(ii+1)*num_utters,:]=pca_xAll[clpItr*num_utters:(clpItr+1)*num_utters,:]
                for ii,hcItr in enumerate(hcTrainIds):
                    hcTrain[ii*num_utters:(ii+1)*num_utters,:]=pca_xAll[hcItr*num_utters:(hcItr+1)*num_utters,:]
                xTrain=np.concatenate((clpTrain,hcTrain),axis=0)
                clpYTrain=np.ones((clpTrain.shape[0])).T
                hcYTrain=np.zeros((hcTrain.shape[0])).T
                yTrain=np.concatenate((clpYTrain,hcYTrain),axis=0)
                
                xTest=np.concatenate((clpTest,hcTest),axis=0)
                clpYTest=np.ones((clpTest.shape[0])).T
                hcYTest=np.zeros((clpTest.shape[0])).T
                yTest=np.concatenate((clpYTest,hcYTest),axis=0)
                
                #repeat m-fda score of a given speaker for every segment/utterance associated with said speaker.
                mfda_yTrain=list(itertools.chain.from_iterable(itertools.repeat(x, num_utters) for x in mfdas[clpTrainIds+hcTrainIds]))
                mfda_yTest=list(itertools.chain.from_iterable(itertools.repeat(x, num_utters) for x in mfdas[clpIds+hcIds]))
                param_grid = [
                  {'C':np.logspace(0,5,25), 'gamma':np.logspace(-8,-4,25), 'degree':[1],'kernel': ['rbf']},
                    ]

                cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
                grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
                mfda_grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
                grid.fit(xTrain, yTrain)
                mfda_grid.fit(xTrain,mfda_yTrain)
                grid=svm.SVC(C=grid.best_params_['C'],degree=grid.best_params_['degree'],gamma=grid.best_params_['gamma'],
                                    kernel=grid.best_params_['kernel'], probability=True)
                mfda_grid=svm.SVC(C=mfda_grid.best_params_['C'],degree=mfda_grid.best_params_['degree'],gamma=mfda_grid.best_params_['gamma'],
                                  kernel=mfda_grid.best_params_['kernel'], probability=True)

                grid.fit(xTrain,yTrain)
                mfda_grid.fit(xTrain,mfda_yTrain)
                
                preds[nrep,:]=mfda_grid.predict(xTrain)
                tst_preds[nrep,:]=mfda_grid.predict(xTest)
                
                tr_bin_class=grid.predict_proba(xTrain)
                diffs[nrep,:]=tr_bin_class[:,0]-tr_bin_class[:,1]
                tst_bin_class=grid.predict_proba(xTest)
                tst_diffs[nrep,:]=tst_bin_class[:,0]-tst_bin_class[:,1]
            
            
            diff_tpls=tuple((x1,x2) for x1,x2 in zip(diffs[0,:],diffs[1,:]))
            clf = SGDClassifier(loss="hinge", penalty="l2")
            clf.fit(diff_tpls, yTrain)
            tr_acc=sum(clf.predict(diff_tpls[0:len(clpYTrain)]))+sum(np.mod(clf.predict(diff_tpls)[len(clpYTrain):]+1,2))/len(yTrain)
            calibrator=CalibratedClassifierCV(clf, cv='prefit')
            modCal=calibrator.fit(diff_tpls, yTrain)

            tst_diff_tpls=tuple((x1,x2) for x1,x2 in zip(tst_diffs[0,:],tst_diffs[1,:]))
            tst_acc=sum(clf.predict(tst_diff_tpls[0:len(clpYTest)]))+sum(np.mod(clf.predict(tst_diff_tpls)[len(clpYTest):]+1,2))/len(yTest)
            bin_class=modCal.predict_proba(tst_diff_tpls)
            
            #predict m-fdas
            pred_tpls=tuple((x1,x2) for x1,x2 in zip(preds[0,:],preds[1,:]))
            mfda_clf = SGDClassifier(loss="hinge", penalty="l2")
            mfda_clf.fit(pred_tpls, mfda_yTrain)       
            
            tst_pred_tpls=tuple((x1,x2) for x1,x2 in zip(tst_preds[0,:],tst_preds[1,:]))
            #predict speaker mfdas for each utterance (will average over after predictions of all spks made).
            for idItr,curr_id in enumerate(clpIds+hcIds):
                results['Data']['mFDA_spear_corr'][o_itr][curr_id]=mfda_clf.predict(np.array(tst_pred_tpls[idItr]).reshape(1,-1))

            class_report=classification_report(yTest,grid.predict(xTest))
            results['Data']['train_acc']+=tr_acc*(1/(int(num_spks/num_clpHc_tests)*total_itrs))
            results['Data']['test_acc']+=tst_acc*(1/(int(num_spks/num_clpHc_tests)*total_itrs))
            results['Data']['class_report'][o_itr][itr]=class_report  
            for cpi,(clpId,hcId) in enumerate(zip(clpIds,hcIds)):          
                results['Data']['bin_class'][o_itr][clpId]=bin_class[cpi*num_utters:(cpi+1)*num_utters]     
                results['Data']['bin_class'][o_itr][hcId]=bin_class[(cpi+num_clpHc_tests//2)*num_utters:(cpi+(num_clpHc_tests//2)+1)*num_utters]
        
#         results['Data']['mFDA_spear_corr']+=stats.spearmanr(predictions['predictions'].values,mfdas)[0]/total_itrs     
    
    if 'wvlt' in reps:
        results.to_pickle(save_path+model+"_wvlt_lateFusionResults.pkl")
    else:
        results.to_pickle(save_path+model+"_lateFusionResults.pkl")



