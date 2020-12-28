import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import pdb
import itertools
from clpAEspeech import AEspeech
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import joblib
import collections
from sklearn.preprocessing import StandardScalerclp
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import json
import argparse


PATH=os.path.dirname(os.path.abspath(__file__))
#LOAD CONFIG.JSON INFO
with open("clpConfig.json") as f:
    data = f.read()
config = json.loads(data)
UNITS=config['general']['UNITS']
UTTERS=['bola','choza','chuzo','coco','gato','jugo','mano','papa','susi']
MODELS=["CAE","RAE","ALL"]


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
       
def featAgg(model,rep,spk_path,ef):
    global UTTERS
    
    allClpNames=[]
    allHcNames=[]
    for u_idx,utter in enumerate(UTTERS):
        clp_path=spk_path+'/'+utter+"/clp/"
        hc_path=spk_path+'/'+utter+"/hc/"   
        clpNames=[name for name in os.listdir(clp_path) if '.wav' in name]
        hcNames=[name for name in os.listdir(hc_path) if '.wav' in name]
        allClpNames.extend([clpName for clpName in clpNames if clpName not in allClpNames])
        allHcNames.extend([hcName for hcName in hcNames if hcName not in allHcNames])
    
    clpNames_count=dict(collections.Counter([name.split('_')[0] for name in allClpNames]))
    hcNames_count=dict(collections.Counter([name.split('_')[0] for name in allHcNames]))    
    spkdict={spk:[] for spk in ['clp','hc']}
    spkdict['clp']={clp:[] for clp in clpNames_count.keys()}
    spkdict['hc']={hc:[] for hc in hcNames_count.keys()}
    for itr,name in enumerate(clpNames_count.keys()):
        spkdict['clp'][name]=[]
    for itr,name in enumerate(hcNames_count.keys()):
        spkdict['hc'][name]=[]
    clpNames=list(np.unique([name.split('_')[0] for name in allClpNames]))
    hcNames=list(np.unique([name.split('_')[0] for name in allHcNames]))
    clpKeys=np.zeros(len(clpNames))
    hcKeys=np.zeros(len(hcNames))
    
    reps=rep
    for uIdx,utter in enumerate(UTTERS):
        clp_path=spk_path+'/'+utter+"/clp/"
        hc_path=spk_path+'/'+utter+"/hc/" 
        if ef==0:           
            clpFeats=getFeats(model,UNITS,rep,clp_path,utter,'clp')
            hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
            clpAll=[c.split('_')[0] for c in np.unique(clpFeats['wav_file'])]
            hcAll=[h.split('_')[0] for h in np.unique(hcFeats['wav_file'])]
            
            for ii,tr in enumerate(clpAll):
                inner_ks=np.array([c.split('_')[0] for c in clpFeats['wav_file']])
                clpTrBns=clpFeats['bottleneck'][np.where(inner_ks==tr.split('_')[0])]
                clpTrBns=np.array([np.mean(clpTrBns,axis=0),np.std(clpTrBns,axis=0),skew(clpTrBns,axis=0),kurtosis(clpTrBns,axis=0)])
                clpTrErrs=clpFeats['error'][np.where(inner_ks==tr.split('_')[0])]
                clpTrErrs=np.array([np.mean(clpTrErrs,axis=0),np.std(clpTrErrs,axis=0),skew(clpTrErrs,axis=0),kurtosis(clpTrErrs,axis=0)])
                ovrall_idx=clpNames.index(tr.split('_')[0])
                if clpKeys[ovrall_idx]==0:
                    spkdict['clp'][tr.split('_')[0]]=np.expand_dims(np.concatenate((clpTrBns,clpTrErrs),axis=1).T,axis=0)
                    clpKeys[ovrall_idx]=1
                else:
                    new=np.concatenate((clpTrBns,clpTrErrs),axis=1).T
                    spkdict['clp'][tr.split('_')[0]]=np.concatenate((spkdict['clp'][tr.split('_')[0]],np.expand_dims(new,axis=0)),axis=0)
                    
            for ii,tr in enumerate(hcAll):
                inner_ks=np.array([h.split('_')[0] for h in hcFeats['wav_file']])
                hcTrBns=hcFeats['bottleneck'][np.where(inner_ks==tr.split('_')[0])]
                hcTrBns=np.array([np.mean(hcTrBns,axis=0),np.std(hcTrBns,axis=0),skew(hcTrBns,axis=0),kurtosis(hcTrBns,axis=0)])
                hcTrErrs=hcFeats['error'][np.where(inner_ks==tr.split('_')[0])]
                hcTrErrs=np.array([np.mean(hcTrErrs,axis=0),np.std(hcTrErrs,axis=0),skew(hcTrErrs,axis=0),kurtosis(hcTrErrs,axis=0)])
                ovrall_idx=hcNames.index(tr.split('_')[0])
                if hcKeys[ovrall_idx]==0:
                    spkdict['hc'][tr.split('_')[0]]=np.expand_dims(np.concatenate((hcTrBns,hcTrErrs),axis=1).T,axis=0)
                    hcKeys[ovrall_idx]=1
                else:
                    new=np.concatenate((hcTrBns,hcTrErrs),axis=1).T
                    spkdict['hc'][tr.split('_')[0]]=np.concatenate((spkdict['hc'][tr.split('_')[0]],np.expand_dims(new,axis=0)),axis=0)
        else:
            for rIdx,rep in enumerate(reps):
                clpFeats=getFeats(model,UNITS,rep,clp_path,utter,'clp')
                hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
                clpAll=[c.split('_')[0] for c in np.unique(clpFeats['wav_file'])]
                hcAll=[h.split('_')[0] for h in np.unique(hcFeats['wav_file'])]
                for ii,tr in enumerate(clpAll):
                    inner_ks=np.array([c.split('_')[0] for c in clpFeats['wav_file']])
                    clpTrBns=clpFeats['bottleneck'][np.where(inner_ks==tr.split('_')[0])]
                    clpTrBns=np.array([np.mean(clpTrBns,axis=0),np.std(clpTrBns,axis=0),skew(clpTrBns,axis=0),kurtosis(clpTrBns,axis=0)])
                    clpTrErrs=clpFeats['error'][np.where(inner_ks==tr.split('_')[0])]
                    clpTrErrs=np.array([np.mean(clpTrErrs,axis=0),np.std(clpTrErrs,axis=0),skew(clpTrErrs,axis=0),kurtosis(clpTrErrs,axis=0)])
                    if rIdx==0:
                        clpR1s[ii,:,:]=np.concatenate((clpTrBns,clpTrErrs),axis=1).T
                    elif len(reps)==3 and rIdx==1:
                        clpR2s[ii,:,:]==np.concatenate((clpTrBns,clpTrErrs),axis=1).T
                    else:
                        if len(reps)==3:
                            old=np.concatenate((clpR1s[ii,:,:],clpR2s[ii,:,:]),axis=0)
                        else:
                            old=clpR1s[ii,:,:]
                        new=np.concatenate((old,np.concatenate((clpTrBns,clpTrErrs),axis=1).T),axis=0)
                        ovrall_idx=clpNames.index(tr.split('_')[0])
                        if clpKeys[ovrall_idx]==0:
                            spkdict['clp'][tr.split('_')[0]]=np.expand_dims(new,axis=0)
                            clpKeys[ovrall_idx]=1
                        else:
                            spkdict['clp'][tr.split('_')[0]]=np.concatenate((spkdict['clp'][tr.split('_')[0]],np.expand_dims(new,axis=0)),axis=0)

                for ii,tr in enumerate(hcAll):
                    inner_ks=np.array([h.split('_')[0] for h in hcFeats['wav_file']])
                    hcTrBns=hcFeats['bottleneck'][np.where(inner_ks==tr.split('_')[0])]
                    hcTrBns=np.array([np.mean(hcTrBns,axis=0),np.std(hcTrBns,axis=0),skew(hcTrBns,axis=0),kurtosis(hcTrBns,axis=0)])
                    hcTrErrs=hcFeats['error'][np.where(inner_ks==tr.split('_')[0])]
                    hcTrErrs=np.array([np.mean(hcTrErrs,axis=0),np.std(hcTrErrs,axis=0),skew(hcTrErrs,axis=0),kurtosis(hcTrErrs,axis=0)])
                    if rIdx==0:
                        hcR1s[ii,:,:]=np.concatenate((hcTrBns,hcTrErrs),axis=1).T
                    elif len(reps)==3 and rIdx==1:
                        hcR2s[ii,:,:]==np.concatenate((hcTrBns,hcTrErrs),axis=1).T
                    else:
                        if len(reps)==3:
                            old=np.concatenate((hcR1s[ii,:,:],hcR2s[ii,:,:]),axis=0)
                        else:
                            old=hcR1s[ii,:,:]
                        new=np.concatenate((old,np.concatenate((hcTrBns,hcTrErrs),axis=1).T),axis=0)
                        ovrall_idx=hcNames.index(tr.split('_')[0])
                        if hcKeys[ovrall_idx]==0:
                            spkdict['hc'][tr.split('_')[0]]=np.expand_dims(new,axis=0)
                            hcKeys[ovrall_idx]=1
                        else:
                            spkdict['hc'][tr.split('_')[0]]=np.concatenate((spkdict['hc'][tr.split('_')[0]],np.expand_dims(new,axis=0)),axis=0)
                        
    return spkdict,clpNames,hcNames,{**clpNames_count,**hcNames_count}



def getDataset(spkdict,clpNames,hcNames):
    num_spks=len(clpNames)    
    
    for ni,name in enumerate(clpNames):
        if ni==0:
            clpFeats=spkdict['clp'][name]
        else:
            clpFeats=np.concatenate((clpFeats,spkdict['clp'][name]),axis=0)
    for ni,name in enumerate(hcNames):
        if ni==0:
            hcFeats=spkdict['hc'][name]
        else:
            hcFeats=np.concatenate((hcFeats,spkdict['hc'][name]),axis=0)
    
    pca_x,ncs=pcaFeats(clpFeats,hcFeats)       
    return pca_x,ncs
        
def pcaFeats(clps,hcs): 
    clpXAll=np.reshape(clps,(clps.shape[0],clps.shape[1]*4))
    hcXAll=np.reshape(hcs,(hcs.shape[0],clps.shape[1]*4))  
    xAll=np.concatenate((clpXAll,hcXAll),axis=0)
    st_xAll=StandardScaler().fit_transform(pd.DataFrame(xAll))
    
    pca = PCA(n_components=min(xAll.shape[0],xAll.shape[1]))
    pca.fit_transform(st_xAll)
    variance = pca.explained_variance_ratio_ #calculate variance ratios
    var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
    ncs=np.count_nonzero(var<90)
    pca = PCA(n_components=ncs)
    pca_xAll=pca.fit_transform(st_xAll)
    return pca_xAll,ncs


    
if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python clpSvmEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse2,early_fuse3, mc_fuse> <clp path>")
        sys.exit()        
    #TRAIN_PATH: './clpSpanish/speech/'
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python clpSvmEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse2,early_fuse3, mc_fuse> <clp path>")
        sys.exit()
    
    if sys.argv[2] not in ['broadband', 'narrowband', 'wvlt', 'early_fuse2', 'early_fuse3', 'mc_fuse']:
        print("python clpSvmEval.py <'CAE','RAE', or 'ALL'> <broadband, narrowband, wvlt, early_fuse2,early_fuse3, mc_fuse> <clp path>")
        sys.exit()
    else:
        rep_typ=sys.argv[2]
    
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
    spk_path=PATH+sys.argv[3]
        
    save_path=PATH+"/clpSpanish/classResults/svm/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    num_utters=len(UTTERS)
    
    ef=0
    if rep_typ in ['broadband','narrowband','wvlt']:
        rep=rep_typ
    elif rep_typ=='mc_fuse':
        feats=[]
        rep='mc_fuse'
    elif rep_typ=='early_fuse2':
        rep=['broadband','narrowband']
        ef=1
    elif rep_typ=='early_fuse3':
        rep=['broadband','narrowband','wvlt']
        ef=1
        
    #get compressed data, n_components, and file_name list 
    spk_dict,clpNames,hcNames,name_count=featAgg(mod,rep,spk_path,ef)
    pca_xAll,ncs=getDataset(spk_dict,clpNames,hcNames)
    spks=clpNames+hcNames
    num_spks=len(spks)
    num_clps=len(clpNames)
    num_hcs=len(hcNames)
    
    #get num preceding utterances (due to each spk not performing each utter).
    prevsAll={nm:0 for nm in spks}
    for nItr,name in enumerate(spks):
        if nItr==0:
            prevsAll[name]=0
        else:
            for prev in spks[:nItr]:
                prevsAll[name]+=name_count[prev]
    
    #split data into training and test with multiple iterations
    num_clpHc_tests=config['svm']['tst_spks']#must be even (same # of test clps and hcs per iter)
    nv=config['svm']['val_size']#number of validation speakers per split#must be even and a divisor of num_spks
    in_iters=config['svm']['in_iters']
    total_itrs=config['svm']['iterations']
    results=pd.DataFrame({'Data':{'train_acc':0,'test_acc':0,'bin_class':{itr:{} for itr in range(total_itrs)},'class_report':{itr:{} for itr in range(total_itrs)}}})
                                      
    for o_itr in range(total_itrs):
        clp_files=clpNames
        hc_files=hcNames
        
        for itr in range(in_iters):
            clpCurrs=[clp_files[idx] for idx in random.sample(range(0,len(clp_files)),num_clpHc_tests//2)]
            hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),num_clpHc_tests//2)]
            spkCurrs=clpCurrs+hcCurrs
            clp_files=[clp for clp in clp_files if clp not in clpCurrs]
            hc_files=[hc for hc in hc_files if hc not in hcCurrs]
            prevs={nm:0 for nm in spkCurrs}
            for nItr,name in enumerate(spkCurrs):
                if nItr==0:
                    prevs[name]=0
                else:
                    for prev in spkCurrs[:nItr]:
                        prevs[name]+=name_count[prev]
            
            if not hc_files:
                hc_files=hcNames
            if len(hc_files)<num_clpHc_tests//2:
                left_ids=[lid for lid in np.arange(len(hcNames)) if hcNames[lid] not in hc_files]
                add_ids=random.sample(left_ids, (num_clpHc_tests//2)-len(hc_files))
                hc_files.extend(np.array(hcNames)[add_ids])
            
            ntst=prevs[spkCurrs[-1]]+name_count[spkCurrs[-1]]
            ntr=pca_xAll.shape[0]-ntst
            
            clpTrainees=[spk for spk in clpNames if spk not in clpCurrs]
            hcTrainees=[spk for spk in hcNames if spk not in hcCurrs]
            trainees=clpTrainees+hcTrainees
            
            trPrevs={nm:0 for nm in trainees}
            ntr_clp=0
            ntr_hc=0
            for nItr,name in enumerate(trainees):
                if nItr==0:
                    trPrevs[name]=0
                    ntr_clp+=name_count[name]
                else:
                    if 'CLP' in name:
                        ntr_clp+=name_count[name]
                    elif 'HC' in name:
                        ntr_hc+=name_count[name]
                    for prev in trainees[:nItr]:
                        trPrevs[name]+=name_count[prev] 
            
            xTest=np.zeros((ntst,ncs))
            yTest=np.concatenate((np.ones(ntst), np.zeros(ntst)))
            xTrain=np.zeros((ntr,ncs))
            yTrain=np.concatenate((np.ones((ntr_clp)), np.zeros((ntr_hc))))
            for ii,tstName in enumerate(spkCurrs):
                if ii==len(spkCurrs)-1:
                    xTest[prevs[tstName]:]=pca_xAll[prevsAll[tstName]:prevsAll[tstName]+name_count[tstName]]
                else:
                    xTest[prevs[tstName]:prevs[spkCurrs[ii+1]]]=pca_xAll[prevsAll[tstName]:prevsAll[tstName]+name_count[tstName]]
            for ii,trName in enumerate(trainees):
                if ii==len(trainees)-1:
                    xTrain[trPrevs[trName]:]=pca_xAll[prevsAll[trName]:prevsAll[trName]+name_count[trName]]
                else:
                    xTrain[trPrevs[trName]:trPrevs[trainees[ii+1]]]=pca_xAll[prevsAll[trName]:prevsAll[trName]+name_count[trName]]
            
            param_grid = [
              {'C':np.logspace(0,5,25), 'gamma':np.logspace(-8,-4,25), 'degree':[1],'kernel': ['rbf']},
                ]

            cv = StratifiedShuffleSplit(n_splits=4, test_size=nv, random_state=42)
            
            grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
            grid.fit(xTrain, yTrain)
            grid=svm.SVC(C=grid.best_params_['C'],degree=grid.best_params_['degree'],gamma=grid.best_params_['gamma'],
                                kernel=grid.best_params_['kernel'], probability=True)
            grid.fit(xTrain,yTrain)
            
            #predict probability of training, get differences and fit sgd class to find optimal split.
            tr_bin_class=grid.predict_proba(xTrain)
            diffs=tr_bin_class[:,0]-tr_bin_class[:,1]
            clf = SGDClassifier(loss="hinge", penalty="l2")
            diffs=np.array(diffs).reshape(-1,1)
            clf.fit(diffs, yTrain)
            train_acc=clf.score(diffs,yTrain)
            calibrator=CalibratedClassifierCV(clf, cv='prefit')
            modCal=calibrator.fit(diffs, yTrain)
            
            #predict probability of test speakers using optimal thresh
            tst_bin_class=grid.predict_proba(xTest)
            tst_diffs=tst_bin_class[:,0]-tst_bin_class[:,1]
            tst_diffs=np.array(tst_diffs).reshape(-1,1)
            test_acc=clf.score(tst_diffs,yTest)
            bin_class=calibrator.predict_proba(tst_diffs)
            
            class_report=classification_report(yTest,grid.predict(xTest))
            results['Data']['train_acc']+=train_acc*(1/(int(num_spks/num_clpHc_tests)*total_itrs))
            results['Data']['test_acc']+=test_acc*(1/(int(num_spks/num_clpHc_tests)*total_itrs))
            results['Data']['class_report'][o_itr][itr]=class_report  
            for cpi,(clpName,hcName) in enumerate(zip(clpCurrs,hcCurrs)):   
                if cpi == len(clpNames)-1:
                    results['Data']['bin_class'][o_itr][clpName]=bin_class[prevs[clpName]:]     
                    results['Data']['bin_class'][o_itr][hcName]=bin_class[prevs[hcName]:]
                else:
                    results['Data']['bin_class'][o_itr][clpName]=bin_class[prevs[clpName]:prevs[clpCurrs[cpi+1]]]     
                    results['Data']['bin_class'][o_itr][hcName]=bin_class[prevs[hcName]:prevs[hcCurrs[cpi+1]]]
                
                
    if rep_typ=='mc_fuse':
        results.to_pickle(save_path+mod+"_mcFusionResults.pkl")
    if rep_typ in ['broadband','narrowband','wvlt']:
        results.to_pickle(save_path+mod+'_'+rep_typ+"_aggResults.pkl")
    if rep_typ=='early_fuse':
        if 'wvlt' in reps:
            results.to_pickle(save_path+mod+"_wvlt_earlyFusionResults.pkl")
        else:
            results.to_pickle(save_path+mod+"_earlyFusionResults.pkl")