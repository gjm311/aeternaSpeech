import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import shutil
import random
import torch.utils.data as data
from pdnn import pdn
import toolbox.traintestsplit as tts
from AEspeech import AEspeech
import json
import argparse
import pdb
from sklearn import metrics

PATH=os.path.dirname(os.path.abspath(__file__))
#LOAD CONFIG.JSON INFO
with open("config.json") as f:
    info = f.read()
config = json.loads(info)
UNITS=config['general']['UNITS']
UTTERS=['pataka','kakaka','pakata','papapa','petaka','tatata']
MODELS=["CAE","RAE","ALL"]
REPS=['broadband','narrowband','wvlt']



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
    save_path=PATH+"/"+"pdSpanish/feats/"+utter+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.isfile(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle'):
        with open(save_path+'/'+rep+'_'+model+'_'+spk_typ+'Feats.pickle', 'rb') as handle:
            feat_vecs = pickle.load(handle)
    else:
        feat_vecs=saveFeats(model,units,rep,wav_path,utter,save_path, spk_typ)
    
    return feat_vecs
        
    
class trainData(data.Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class testData(data.Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)   
    
    
    

if __name__=="__main__":

    PATH=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv)!=3:
        print("python pdnnEvalAgg.py <'CAE','RAE', or 'ALL'> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/<UTTER>/'
    
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python pdnnEvalAgg.py <'CAE','RAE', or 'ALL'> <pd path>")
        sys.exit()

    if sys.argv[2][0] !='/':
        sys.argv[2] = '/'+sys.argv[2]
    if sys.argv[2][-1] !='/':
        sys.argv[2] = sys.argv[3]+'/'
        
    reps=['broadband','narrowband']
    
    LR=config['dnn']['LR']
    BATCH_SIZE=config['dnn']['BATCH_SIZE']
    NUM_W=config['dnn']['NUM_W']
    N_EPOCHS=config['dnn']['N_EPOCHS']
    num_pdHc_tests=config['dnn']['tst_spks']#must be even (same # of test pds and hcs per iter)
    nv=config['dnn']['val_spks']#number of validation speakers per split

#     LRs=[10**-ex for ex in np.linspace(4,7,6)]

    
    NBF=config['mel_spec']['INTERP_NMELS']
    
    save_path=PATH+"/pdSpanish/classResults/dnn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ntr=100-(num_pdHc_tests+nv)
    testResults=pd.DataFrame({splItr:{'test_loss':0, 'test_acc':0, 'tstSpk_data':{}} for splItr in range(100//num_pdHc_tests)})     
    train_res=[]
        
    #iterate through all pd and hc speakers for a given utterance (see UTTERS for options) and using leave ten out, train a DNN
    #(see pdnn.py) and classify one by one if PD or HC.
#     lr_score_opt=0
#     lr_scores=pd.DataFrame(columns=LRs,index=np.arange(1))
#     for lrItr,LR in enumerate(LRs):

    #aggregate all utterance data per speaker together.
    pdIds=np.arange(50)
    hcIds=np.arange(50,100)
    spkDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}
    for u_idx,utter in enumerate(UTTERS):
        for rIdx,rep in enumerate(reps):
            pd_path=PATH+sys.argv[2]+'/'+utter+"/pd/"
            hc_path=PATH+sys.argv[2]+'/'+utter+"/hc/"   
            pdNames=[name for name in os.listdir(pd_path) if '.wav' in name]
            hcNames=[name for name in os.listdir(hc_path) if '.wav' in name]
            pdNames.sort()
            hcNames.sort()
            spks=pdNames+hcNames
            num_spks=len(spks)
            num_pd=len(pdNames)
            num_hc=len(hcNames)
            pdFeats=getFeats(mod,UNITS,rep,pd_path,utter,'pd')
            hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')
            pdAll=np.unique(pdFeats['wav_file'])
            hcAll=np.unique(hcFeats['wav_file'])

            for p in pdIds:
                pdBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[p])]
                pdErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[p])]
                if u_idx==0 and rIdx==0:
                    spkDict['pd'][p]=np.concatenate((pdBns,pdErrs),axis=1)
                else:
                    spkDict['pd'][p]=np.concatenate((spkDict['pd'][p],np.concatenate((pdBns,pdErrs),axis=1)),axis=0)

            for h in hcIds:
                hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[h])]
                hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[h])]
                if u_idx==0 and rIdx==0:
                    spkDict['hc'][h]=np.concatenate((hcBns,hcErrs),axis=1)
                else:
                    spkDict['hc'][h]=np.concatenate((spkDict['hc'][h],np.concatenate((hcBns,hcErrs),axis=1)),axis=0)
    
    #split data into training and test with multiple iterations (evenly split PD:HC)
    pd_files=pdNames
    hc_files=hcNames
    for itr in range(100//num_pdHc_tests):
        #RESET model
        model=pdn(UNITS+NBF)
        criterion=nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = LR)

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            model.cuda()
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        
        #Get test speaker features
        pdCurrs=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),num_pdHc_tests//2)]
        hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),num_pdHc_tests//2)]
        pd_files=[pd for pd in pd_files if pd not in pdCurrs]
        hc_files=[hc for hc in hc_files if hc not in hcCurrs]

        pdIds=[spks.index(pdCurr) for pdCurr in pdCurrs]
        hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]

        testDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}
        for pdItr in pdIds:
            testDict['pd'][pdItr]=spkDict['pd'][pdItr]
        for hcItr in hcIds:
            testDict['hc'][hcItr]=spkDict['hc'][hcItr]
        
        #Separate 'nv' (equal number of pd/hc) Validation speakers and get features
        notTestSpksPD=[spk for spk in pdNames if spk not in pdCurrs]
        notTestSpksHC=[spk for spk in hcNames if spk not in hcCurrs]
        validsPD=[notTestSpksPD[idx] for idx in random.sample(range(0,len(notTestSpksPD)),nv//2)]
        validsHC=[notTestSpksHC[idx] for idx in random.sample(range(0,len(notTestSpksHC)),nv//2)]
        valids=validsPD+validsHC
        valIds=[spks.index(valid) for valid in valids]
        valDict={num:{'feats':[]} for num in valIds}

        #getting bottle neck features and reconstruction error for validation speakers
        for ii,val in enumerate(valids):
            vitr=valIds[ii]
            if vitr<num_pd:
                spk_typ='pd'
            else:
                spk_typ='hc'
            valDict[vitr]=spkDict[spk_typ][vitr]

        trainResults_epo= pd.DataFrame({'train_loss':0.0, 'train_acc':0.0,'val_loss':0.0, 'val_acc':0.0}, index=np.arange(N_EPOCHS))

        for epoch in range(N_EPOCHS):  
            train_loss=0.0
            train_acc=0.0
            rand_range=np.arange(num_spks)
            random.shuffle(rand_range)
            
            #TRAIN dnn for each speaker individually.             
            for trainItr in rand_range:   
                if trainItr in np.concatenate((pdIds,hcIds,valIds)):
                    continue

                if trainItr<num_pd:
                    trainIndc=1
                    trainOpp=0
                    spk_typ='pd'
                else:
                    trainIndc=0
                    trainOpp=1
                    spk_typ='hc'

                #getting bottle neck features and reconstruction error for particular training speaker
                xTrain=spkDict[spk_typ][trainItr]
                xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T
                train_data=trainData(torch.FloatTensor(xTrain), torch.FloatTensor(yTrain))
                train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)
                start=time.time()

                train_loss_curr=0.0
                if len(train_loader)>0:
                    #TRAINING ON SNIPPETS OF SPEAKER UTTERANCES
                    model.train() # prep model for training
                    for X_train, y_train in train_loader:

                        #clear the gradients of all optimized variables
                        optimizer.zero_grad()

                        X_train=X_train.float()                
                        y_train=y_train.float()

                        if torch.cuda.is_available():
                            X_train,y_train=X_train.cuda(),y_train.cuda()

                        y_pred=model.forward(X_train)
                        #Find difference in probability of PD v. HC for all segments.
                        if torch.cuda.is_available():
                            y_pred=y_pred.cuda()

                        loss=criterion(y_pred, y_train)

                        loss.backward()
                        optimizer.step()
                        train_loss_curr += loss.item()*y_train.size(0)

                    #tally up train loss total for given speaker
                    train_loss+=train_loss_curr/len(train_loader.dataset)                           

            #Record train loss at end of each epoch (divide by number of train patients - ntr).
            trainResults_epo.iloc[epoch]['train_loss']=train_loss/ntr
    

            #Iterate through thresholds and choose one that yields best validation acc.
            #Iterate through all num_tr training patients and classify based off difference in probability of PD/HC
            max_val_acc=0
            tr_acc=0
            num_tr=0
            if epoch==N_EPOCHS-1:
                threshes=np.arange(-100,100)
            else:
                threshes=[0]

            for thresh in threshes:
                thresh=thresh/100
                for trainItr in rand_range:   
                    if trainItr in np.concatenate((pdIds,hcIds,valIds)):
                        continue

                    if trainItr<num_pd:
                        trainIndc=1
                        trainOpp=0
                        trainFeats=pdFeats
                    else:
                        trainIndc=0
                        trainOpp=1
                        trainFeats=hcFeats

                    #getting bottle neck features and reconstruction error for particular training speaker
                    bns=trainFeats['bottleneck'][np.where(trainFeats['wav_file']==spks[trainItr])]
                    errs=trainFeats['error'][np.where(trainFeats['wav_file']==spks[trainItr])]
                    xTrain=np.concatenate((bns,errs),axis=1)
                    xTrain=(xTrain-np.min(xTrain))/(np.max(xTrain)-np.min(xTrain))
                    yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T

                    train_data=testData(torch.FloatTensor(xTrain))
                    train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)
                    y_pred_tag=[]
                    model.eval()
                    with torch.no_grad():
                        for X_tr in train_loader:
                            yTr=np.vstack((np.ones((X_tr.shape[0]))*trainIndc,np.ones((X_tr.shape[0]))*trainOpp)).T
                            if torch.cuda.is_available():
                                X_tr=X_tr.cuda()

                            y_tr_pred = model.forward(X_tr)

                            #Find difference in probability of PD v. HC for all segments. 
                            y_pred_tag.extend((y_tr_pred[:,0]-y_tr_pred[:,1]).cpu().detach().numpy())

                    #Wlog, if difference greater than 0 occurs more and speaker is PD, than identification is correct (1=PD,0=HC).
                    y_pred_tag=np.array(y_pred_tag)
                    if len(y_pred_tag)>0:
                        num_tr+=1

                        """THREE CLASSIFIERS BELOW"""
#                         #Classification correct if (wlog) median prob difference indicates correct spk type.
#                         if trainIndc==1 and np.median(y_pred_tag)>0:
#                             tr_acc+=1
#                         elif trainIndc==0 and np.median(y_pred_tag)<0:
#                             tr_acc+=1               

    #                    #Classification correct if more frame probability differences indicate correct spk type. 
    #                     if trainIndc==1 and (len(y_pred_tag[np.where(y_pred_tag>0)]) >= len(y_pred_tag[np.where(y_pred_tag<0)])):
    #                         tr_acc+=1
    #                     elif trainIndc==0 and (len(y_pred_tag[np.where(y_pred_tag<0)]) >= len(y_pred_tag[np.where(y_pred_tag>0)])):
    #                         tr_acc+=1

                       #Classification is based off percent of frames classified correctly.
                        if trainIndc==1 :
                            tr_acc+=len(y_pred_tag[np.where(y_pred_tag>0)])/len(y_pred_tag)
                        elif trainIndc==0:
                            tr_acc+=len(y_pred_tag[np.where(y_pred_tag<0)])/len(y_pred_tag)
                    else:
                        continue

#                     trainResults_epo.iloc[epoch]['train_acc']=tr_acc/num_tr

                if np.mod(epoch,1)==0:
                    #Validate at end of each 1 epochs for nv speakers
                    val_loss=0.0
                    val_acc=0
                    num_val=0
                    for vid in valDict.keys():
                        if vid<num_pd:
                            indc=1
                            opp=0
                        else:
                            indc=0
                            opp=1
                        y_pred_tag=[]
                        xVal=valDict[vid]
                        xVal=(xVal-np.min(xVal))/(np.max(xVal)-np.min(xVal))
                        test_data=testData(torch.FloatTensor(xVal))
                        test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True) 

                        model.eval()
                        with torch.no_grad():
                            for X_test in test_loader:
                                yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                                if torch.cuda.is_available():
                                    X_test=X_test.cuda()

                                y_test_pred = model.forward(X_test)

                                #Find difference in probability of PD v. HC for all segments. 
                                y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())
                                if torch.cuda.is_available():
                                    loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                                else:
                                    loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                                val_loss+=loss.item()*X_test.size(0)


                        val_loss=val_loss/len(test_loader.dataset)
                        y_pred_tag=np.array(y_pred_tag)
                        if len(y_pred_tag)>0:
                            num_val+=1

                        """THREE CLASSIFIERS BELOW"""
        #                     #Classification correct if (wlog) median prob difference indicates correct spk type.
        #                     if indc==1 and np.median(y_pred_tag)>=0:
        #                         val_acc+=1
        #                     elif indc==0 and np.median(y_pred_tag)<=0:
        #                         val_acc+=1

        #                    #Classification correct if more frame probability differences indicate correct spk type. 
        #                     if indc==1 and (len(y_pred_tag[np.where(y_pred_tag>0)]) >= len(y_pred_tag[np.where(y_pred_tag<0)])):
        #                         val_acc+=1
        #                     elif indc==0 and (len(y_pred_tag[np.where(y_pred_tag<0)]) >= len(y_pred_tag[np.where(y_pred_tag>0)])):
        #                         val_acc+=1

                       #Classification is based off percent of frames classified correctly.
                        if indc==1 :
                            val_acc+=len(y_pred_tag[np.where(y_pred_tag>thresh)])/len(y_pred_tag)
                        elif indc==0:
                            val_acc+=len(y_pred_tag[np.where(y_pred_tag<thresh)])/len(y_pred_tag)
                        else:
                            continue

                if epoch==(N_EPOCHS-1):
                    if val_acc/num_val>max_val_acc:
                        max_val_acc=val_acc/num_val
                        opt_thresh=thresh
                        trainResults_epo.iloc[epoch]['train_acc']=tr_acc/num_tr
                        trainResults_epo.iloc[epoch]['val_loss']=val_loss/num_val
                        trainResults_epo.iloc[epoch]['val_acc']=val_acc/num_val
                else:
                    trainResults_epo.iloc[epoch]['train_acc']=tr_acc/num_tr
                    trainResults_epo.iloc[epoch]['val_loss']=val_loss/num_val
                    trainResults_epo.iloc[epoch]['val_acc']=val_acc/num_val

            print('Train Loss: {:.6f} Train Accuracy: {}\nValidation Loss: {:.6f} Validation Accuracy: {}\n'.format(
            train_loss/num_tr,  
            tr_acc/num_tr,
            val_loss/num_val,
            val_acc/num_val,
            ))      

        #AFTER MODEL TRAINED (FOR ALL SPEAKERS AND OVER NUM_EPOCHS), TEST MODEL ON LEFT OUT SPEAKERS  
        test_loss=0.0
        test_acc=0.0
        num_tst
        for spkItr,spk in enumerate(['pd','hc']):
            dic=testDict[spk]

            for tstId in dic.keys():
                if tstId<num_pd:
                    indc=1
                    opp=0
                else:
                    indc=0
                    opp=1
                y_pred_tag=[]  
                test_loss_curr=0
                xTest=dic[tstId]
                xTest=(xTest-np.min(xTest))/(np.max(xTest)-np.min(xTest))
                test_data=testData(torch.FloatTensor(xTest))
                test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=False, shuffle=True)  
                model.eval()
                with torch.no_grad():
                    for X_test in test_loader:
                        yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                        if torch.cuda.is_available():
                            X_test=X_test.cuda()

                        y_test_pred = model.forward(X_test)

                        #Find difference in probability of PD v. HC for all segments. 
                        y_pred_tag.extend((y_test_pred[:,0]-y_test_pred[:,1]).cpu().detach().numpy())

                        if torch.cuda.is_available():
                            loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                        else:
                            loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                        test_loss_curr+=loss.item()*X_test.size(0)

                #accuracy determined on majority rule (wlog, if more frames yield probability differences greater than 0,
                #and if speaker is PD than classification assessment is correct (1=>PD,0=>HC).
                test_loss+=test_loss_curr/len(test_loader.dataset)
                y_pred_tag=np.array(y_pred_tag)
                
                #THREE CLASSIFICATION TYPES:
                #1. Classification correct if (wlog) median prob difference indicates correct spk type.
#                 if indc==1 and np.median(y_pred_tag)>0:
#                     test_acc+=1
#                 elif indc==0 and np.median(y_pred_tag)<0:
#                     test_acc+=1
                    
#                #2. Classification correct if more frame probability differences indicate correct spk type. 
#                 if indc==1:
#                     if (len(y_pred_tag[np.where(y_pred_tag>0)]) >= len(y_pred_tag[np.where(y_pred_tag<0)])):
#                         test_acc+=1
#                 if indc==0:
#                     if (len(y_pred_tag[np.where(y_pred_tag<0)]) >= len(y_pred_tag[np.where(y_pred_tag>0)])):
#                         test_acc+=1
                
                if len(y_pred_tag)>0:
                    num_tst+=1
                #3. Classification is based off percent of frames classified correctly.
                if indc==1 :
                    test_acc+=len(y_pred_tag[np.where(y_pred_tag>0)])/len(y_pred_tag)
                elif indc==0:
                    test_acc+=len(y_pred_tag[np.where(y_pred_tag<0)])/len(y_pred_tag)

                #Store raw scores for each test speaker (probability of PD and HC as output by dnn) for ROC.
                testResults[itr]['tstSpk_data'][tstId]=y_test_pred.cpu().detach().numpy()


        #Store and report loss and accuracy for batch of test speakers.            
        testResults[itr]['test_loss'],testResults[itr]['test_acc']=test_loss/num_pdHc_tests,test_acc/num_pdHc_tests
        print('\nTest Loss: {:.3f} \tTest Acc: {:.3f} '.format(
                    test_loss/num_pdHc_tests,
                    test_acc/num_pdHc_tests
            ))
          
        train_res.append(trainResults_epo)


    trainResults=pd.concat(train_res,keys=(np.arange(itr+1)))

#         #compare acc for all lrs and save highest
#         lr_score=0
#         for item in testResults:
#             for index in testResults.index:
#                 if index[1] == 'test_acc':
#                     lr_score+=testResults[item][index]/(10*len(UTTERS))
#         if lr_score>lr_score_opt:
#             trainResults.to_pickle(save_path+mod+'_'+rep+"TrainResults.pkl")
#             testResults.to_pickle(save_path+mod+'_'+rep+"TestResults.pkl") 
#             lr_score_opt=lr_score
#         lr_scores.iloc[0][LRs[lrItr]]=lr_score
#         lr_scores.to_csv(save_path+mod+'_'+rep+"lrResults.csv")

    trainResults.to_pickle(save_path+mod+'_earlyFusion_trainResults.pkl')
    testResults.to_pickle(save_path+mod+'_earlyFusion_testResults.pkl')






