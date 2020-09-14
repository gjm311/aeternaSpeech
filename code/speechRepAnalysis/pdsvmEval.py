import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
from sklearn import svm
from sklearn.metrics import hinge_loss
from AEspeech import AEspeech
import pdb
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        
    
    mfda_path=PATH+"/pdSpanish/"
        
    save_path=PATH+"/pdSpanish/classResults/svm/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
        
    mfdas=pd.read_csv(mfda_path+"metadata-Spanish_All.csv")['M-FDA'].values
    pd_mfdas=mfdas[0:50]
    hc_mfdas=mfdas[50:]
    
    n_comps=4
    n_trspks=15
    if rep=='wvlt':
        num_feats=64+256
    else:
        num_feats=128+256
    
    results=pd.DataFrame({utter:{'train_loss':0,'train_acc':0,'test_loss':0,'test_acc':0, 
            'tstSpk_data':{}} for utter in UTTERS})
    for utter in UTTERS:
        pd_path=PATH+sys.argv[3]+'/'+utter+"/pd/"
        hc_path=PATH+sys.argv[3]+'/'+utter+"/hc/"   
        pds=[name for name in os.listdir(pd_path) if '.wav' in name]
        hcs=[name for name in os.listdir(hc_path) if '.wav' in name]
        pd_files=pds
        hc_files=hcs
        pds.sort()
        hcs.sort()
        spks=pds+hcs
        num_pd=len(pds)
        num_hc=len(hcs)
        results[utter]['tstSpk_data']={key:0 for key in range(len(pds+hcs))}
      
        pdFeats=getFeats(model,UNITS,rep,pd_path,utter,'pd')
        hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
        pca = PCA(n_components=n_comps)
        #iterate through all pd and hc speakers for a given utterance (see UTTERS for options) and using leave ten out, train an SVM
        #and classify if PD or HC.
        for itr in range(int(np.max([num_pd//10,num_hc//10]))):

            #Get test speaker features, load test
            pdCurrs=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),5)]
            hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),5)]
            pd_files=[pd for pd in pd_files if pd not in pdCurrs]
            hc_files=[hc for hc in hc_files if hc not in hcCurrs]

            pdIds=[spks.index(pdCurr) for pdCurr in pdCurrs]
            hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]
            testDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}
            pdTests=np.zeros((len(pdCurrs),n_comps,4))
            hcTests=np.zeros((len(hcCurrs),n_comps,4))

            for ii,pdItr in enumerate(pdIds):
                pdBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[pdItr])]
                pdErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[pdItr])]
                pdTest=np.concatenate((pdBns,pdErrs),axis=1)
                pdTest=StandardScaler().fit_transform(pd.DataFrame(pdTest))
                pdPCs=pca.fit_transform(pdTest)
                pdTests[ii,:,:]=np.array([np.mean(pdPCs,axis=0),np.std(pdPCs,axis=0),skew(pdPCs,axis=0),kurtosis(pdPCs,axis=0)]).T
                testDict['pd'][pdItr]=pdTests[ii,:,:]
            for ii,hcItr in enumerate(hcIds):
                hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[hcItr])]
                hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[hcItr])]
                hcTest=np.concatenate((hcBns,hcErrs),axis=1)
                hcTest=StandardScaler().fit_transform(pd.DataFrame(hcTest))
                hcPCs=pca.fit_transform(hcTest)
                hcTests[ii,:,:]=np.array([np.mean(hcPCs,axis=0),np.std(hcPCs,axis=0),skew(hcPCs,axis=0),kurtosis(hcPCs,axis=0)]).T
                testDict['hc'][hcItr]=hcTests[ii,:,:]
                        
            pdTests=np.reshape(pdTests,(len(pdCurrs)*n_comps,4))
            hcTests=np.reshape(hcTests,(len(hcCurrs)*n_comps,4))
            pdYTest=np.ones((pdTests.shape[0])).T
            hcYTest=np.zeros((hcTests.shape[0])).T
            yTest=np.concatenate((pdYTest,hcYTest),axis=0)


#                 pd_notTestSpks=[spk for idx,spk in enumerate(spks) if spk not in pdCurrs+hcCurrs and idx<num_pd]
#                 hc_notTestSpks=[spk for idx,spk in enumerate(spks) if spk not in hcCurrs+pdCurrs and idx>=num_pd]

#                 for valItr in range(10):
#                     new_pdTr=[]
#                     new_hcTr=[]

#                     #Separate Validation with IDs
#                     pdVals=[pd_notTestSpks[idx] for idx in random.sample(range(0,len(pd_notTestSpks)),4)]
#                     hcVals=[hc_notTestSpks[idx] for idx in random.sample(range(0,len(hc_notTestSpks)),4)]
#                     pd_notTestSpks=[new for new in pd_notTestSpks if new not in pdVals]
#                     hc_notTestSpks=[new for new in hc_notTestSpks if new not in hcVals]
#                     pdValIds=[spks.index(v) for v in pdVals]
#                     hcValIds=[spks.index(v) for v in hcVals]
#                     valDict={spk:{num[i]:{'feats':[]} for num in zip(pdValIds,hcValIds)} for i,spk in enumerate(['pd','hc'])}

#                     #getting bottle neck features and reconstruction error for validation
#                     for ii,v in enumerate(pdVals):
#                         vitr=pdValIds[ii]
#                         pdValBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[vitr])]
#                         pdValErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[vitr])]
#                         pdVFeats=np.concatenate((pdValBns,pdValErrs),axis=1)
#                         valDict['pd'][vitr]=pdVFeats
#                     for ii,v in enumerate(hcVals):
#                         vitr=hcValIds[ii]
#                         hcValBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[vitr])]
#                         hcValErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[vitr])]
#                         hcVFeats=np.concatenate((hcValBns,hcValErrs),axis=1)
#                         valDict['hc'][vitr]=hcVFeats

            pdVals=[]
            hcVals=[]
            #Aggregate training features for each speaker.  
            pdTrainees=[spk for idx,spk in enumerate(pds) if spk not in (pdCurrs+pdVals)]
            hcTrainees=[spk for idx,spk in enumerate(hcs) if spk not in (hcCurrs+hcVals)]
            pdTrainIds=[spks.index(tr) for tr in pdTrainees]
            hcTrainIds=[spks.index(tr) for tr in hcTrainees]
#             pd_mfdasExt=[mfda for idx,mfda in enumerate(np.sort(pd_mfdas)) if idx in pdTrainIds][-n_trspks:]
#             hc_mfdasExt=[mfda for idx,mfda in enumerate(np.sort(hc_mfdas)) if idx+50 in hcTrainIds][0:n_trspks]
#             pd_mfdasIdx=[np.where(mfda==pd_mfdas) for mfda in np.unique(pd_mfdasExt)]
#             hc_mfdasIdx=[np.where(mfda==hc_mfdas) for mfda in np.unique(hc_mfdasExt)]
#             pdTrainIds=[val for v in pd_mfdasIdx for val in v[0]][-n_trspks:]
#             hcTrainIds=[val+50 for v in hc_mfdasIdx for val in v[0]][0:n_trspks]
            
#             pdTrainees=[spk for idx,spk in enumerate(spks) if idx in pdTrainIds]
#             hcTrainees=[spk for idx,spk in enumerate(spks) if idx in hcTrainIds]
            pdTrs=np.zeros((len(pdTrainees),n_comps,4))
            hcTrs=np.zeros((len(hcTrainees),n_comps,4))
            
            #getting bottle neck features and reconstruction error for training
            for ii,tr in enumerate(pdTrainees):
                tritr=pdTrainIds[ii]
                pdTrBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[tritr])]
                pdTrErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[tritr])]
                pdTr=np.concatenate((pdTrBns,pdTrErrs),axis=1)
                pdTr=StandardScaler().fit_transform(pd.DataFrame(pdTr))
                pdPCs=pca.fit_transform(pdTr)
                pdTrs[ii,:,:]=np.array([np.mean(pdPCs,axis=0),np.std(pdPCs,axis=0),skew(pdPCs,axis=0),kurtosis(pdPCs,axis=0)]).T
            for ii,tr in enumerate(hcTrainees):
                tritr=hcTrainIds[ii]
                hcTrBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[tritr])]
                hcTrErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[tritr])]
                hcTr=np.concatenate((hcTrBns,hcTrErrs),axis=1)
                hcTr=StandardScaler().fit_transform(pd.DataFrame(hcTr))
                hcPCs=pca.fit_transform(hcTr)
                hcTrs[ii,:,:]=np.array([np.mean(hcPCs,axis=0),np.std(hcPCs,axis=0),skew(hcPCs,axis=0),kurtosis(hcPCs,axis=0)]).T
                
            pdTrs=np.reshape(pdTrs,(pdTrs.shape[0]*n_comps,4))
            hcTrs=np.reshape(hcTrs,(hcTrs.shape[0]*n_comps,4))
            xTrain=np.concatenate((pdTrs,hcTrs),axis=0)
            pdYTrain=np.ones((pdTrs.shape[0])).T
            hcYTrain=np.zeros((hcTrs.shape[0])).T
            yTrain=np.concatenate((pdYTrain,hcYTrain),axis=0)
#                     support = svm.SVC(gamma='scale',kernel='poly',degree=3, probability = True)
#                     support.fit(xTrain,yTrain)

#                     for spk in ['pd','hc']:
#                         if spk=='pd':
#                             y=np.ones((pdVFeats.shape[0])).T
#                             inter_pred=support.predict(pdVFeats)
#                             new_pdTr.append([p for idx,p in enumerate(pdVFeats) if np.round(inter_pred[idx])==y[idx]])
#                         elif spk=='hc':
#                             y=np.zeros((hcVFeats.shape[0])).T
#                             inter_pred=support.predict(hcVFeats)
#                             new_hcTr.append([p for idx,p in enumerate(hcVFeats) if np.round(inter_pred[idx])==y[idx]])


            #With training features that were best at identifying class, train model
#                 new_pdTr=np.squeeze(new_pdTr,axis=0)
#                 new_hcTr=np.squeeze(new_hcTr,axis=0)
#                 new_pdYTrain=np.ones((new_pdTr.shape[0])).T
#                 new_hcYTrain=np.zeros((new_hcTr.shape[0])).T
#                 try:
#                     xTrain=np.concatenate((new_pdTr,new_hcTr),axis=0)
#                     yTrain=np.concatenate((new_pdYTrain,new_hcYTrain))
#                 except:
#                     continue
            
    
            grid=joblib.load(PATH+"/pdSpanish/classResults/svm/params/"+model+'_'+utter+'_'+rep+'Grid.pkl')
            support=grid
#             support = svm.SVC(gamma='scale',kernel='poly',degree=3, probability = True)
#             support.fit(xTrain,yTrain)
            pdTr_pred = support.predict(pdTrs)
            hcTr_pred = support.predict(hcTrs)
            pdTst_pred = support.predict(pdTests)
            hcTst_pred = support.predict(hcTests)
            for spkItr,spk in enumerate(['pd','hc']):
                dic=testDict[spk]
                for tstId in dic.keys(): 
                    xTest=dic[tstId]
                    pred=support.predict(xTest)
                    yTst=np.ones((xTest.shape[0]))*np.mod(spkItr+1,2)
                    results[utter]['test_acc']+=sum(np.mod((np.round(pred)-yTst)+1,2))/xTest.shape[0]*(1/(num_pd+num_hc))
#                         results[utter]['tstSpk_data'][tstId]=sum(np.round(pred)/xTest.shape[0])
                    results[utter]['tstSpk_data'][tstId]=min(np.mean(pred),1)

            pdTr_hit=sum(np.mod((pdTr_pred-pdYTrain)+1,2))
            hcTr_hit=sum(np.mod((hcTr_pred-hcYTrain)+1,2))
            pdTst_hit=sum(np.mod((pdTst_pred-pdYTest)+1,2))
            hcTst_hit=sum(np.mod((hcTst_pred-hcYTest)+1,2))
            results[utter]['train_acc']+=(pdTr_hit+hcTr_hit)/(pdYTrain.shape[0]+hcYTrain.shape[0])
            results[utter]['train_loss']+=hinge_loss(yTrain,np.concatenate((pdTr_pred,hcTr_pred),axis=0),)
            results[utter]['test_loss']+=hinge_loss(yTest,np.concatenate((pdTst_pred,hcTst_pred)))

        results[utter]['train_acc']=results[utter]['train_acc']/int(np.max([num_pd//5,num_hc//5]))
        results[utter]['train_loss']=results[utter]['train_loss']/int(np.max([num_pd//5,num_hc//5]))
        results[utter]['test_loss']=results[utter]['test_loss']/int(np.max([num_pd//5,num_hc//5]))

    results.to_pickle(save_path+model+'_'+rep+"Results.pkl")
    
    
    
