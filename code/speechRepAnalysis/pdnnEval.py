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
import pdb

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
    if len(sys.argv)!=4:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()        
    #TRAIN_PATH: './pdSpanish/speech/<UTTER>/'
    
    
    if sys.argv[1] in MODELS:
        mod=sys.argv[1]
    else:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()
    
    if sys.argv[2] in REPS:
        rep=sys.argv[2]
    else:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()    
        
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
        
  
    BATCH_SIZE=10
    NUM_W=0
    N_EPOCHS=100
    LR=0.0001
    if rep=='spec':
        NBF=128
    else:
        NBF=64
        
    save_path=PATH+"/pdSpanish/classResults/dnn/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #iterate through all pd and hc speakers for a given utterance (see UTTERS for options) and using leave ten out, train a DNN
    #(see pdnn.py) and classify one by one if PD or HC.
    
    train_res=[]
    test_res=[]
    for itr in range(5):
#         trainResults_curr=pd.DataFrame(index=UTTERS)
        trainResultsEpo_curr=[]
        testResults_curr=pd.DataFrame({utter:{'test_loss':0, 'test_acc':0, 'tstSpk_data':{}} for utter in UTTERS})
        
        for u_idx,utter in enumerate(UTTERS):
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
            total_spks=num_pd+num_hc
            rand_range=np.arange(total_spks)
            random.shuffle(rand_range)

            pdFeats=getFeats(mod,UNITS,rep,pd_path,utter,'pd')
            hcFeats=getFeats(mod,UNITS,rep,hc_path,utter,'hc')

            #RESET model that will be trained on all speakers but ten for test (must reset so test speaker data not used in model).
            if rep=='spec':
                model=pdn(UNITS+NBF)
            elif rep=='wvlt':
                model=pdn(UNITS+NBF)
            criterion=nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = LR)

            if torch.cuda.is_available():
                print(torch.cuda.get_device_name(0))
                model.cuda()
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

            #Get test speaker features, load test
            pdCurrs=[pd_files[idx] for idx in random.sample(range(0,len(pd_files)),5)]
            hcCurrs=[hc_files[idx] for idx in random.sample(range(0,len(hc_files)),5)]
            pd_files=[pd for pd in pd_files if pd not in pdCurrs]
            hc_files=[hc for hc in hc_files if hc not in hcCurrs]

            pdIds=[spks.index(pdCurr) for pdCurr in pdCurrs]
            hcIds=[spks.index(hcCurr) for hcCurr in hcCurrs]

            testDict={spk:{num[i]:{'feats':[]} for num in zip(pdIds,hcIds)} for i,spk in enumerate(['pd','hc'])}

            for pdItr in pdIds:
                pdBns=pdFeats['bottleneck'][np.where(pdFeats['wav_file']==spks[pdItr])]
                pdErrs=pdFeats['error'][np.where(pdFeats['wav_file']==spks[pdItr])]
                pdTest=np.concatenate((pdBns,pdErrs),axis=1)
                testDict['pd'][pdItr]=pdTest
            for hcItr in hcIds:
                hcBns=hcFeats['bottleneck'][np.where(hcFeats['wav_file']==spks[hcItr])]
                hcErrs=hcFeats['error'][np.where(hcFeats['wav_file']==spks[hcItr])]
                hcTest=np.concatenate((hcBns,hcErrs),axis=1)
                testDict['hc'][hcItr]=hcTest

            trainResults_epo=pd.DataFrame({'train_loss':0, 'train_acc':0, 'val_loss':0,'val_acc':0},index=np.arange(N_EPOCHS))
            for epoch in range(N_EPOCHS):   
                train_loss=0.0
                train_acc=0.0

                #Separate Validation with IDs
                notTestSpks=[spk for spk in spks if spk not in pdCurrs+hcCurrs]
                valids=[notTestSpks[idx] for idx in random.sample(range(0,len(notTestSpks)),5)]
                valIds=[spks.index(valid) for valid in valids]
                valDict={num:{'feats':[]} for num in valIds}

                #getting bottle neck features and reconstruction error for validation
                for ii,val in enumerate(valids):
                    vitr=valIds[ii]
                    if vitr<num_pd:
                        feats=pdFeats
                    else:
                        feats=hcFeats
                    valBns=feats['bottleneck'][np.where(feats['wav_file']==spks[vitr])]
                    valErrs=feats['error'][np.where(feats['wav_file']==spks[vitr])]
                    valTest=np.concatenate((valBns,valErrs),axis=1)
                    valDict[vitr]=valTest

                #TRAIN dnn for each speaker individually for given number of epochs (order is random between pd, hc patients).              
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
                    yTrain=np.vstack((np.ones((xTrain.shape[0]))*trainIndc,np.ones((xTrain.shape[0]))*trainOpp)).T

                    train_data=trainData(torch.FloatTensor(xTrain), torch.FloatTensor(yTrain))
                    train_loader=torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_W)
                    start=time.time()
                    train_loss_curr=0.0
                    y_pred_tag=[]
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
                            y_pred_tag.append(np.mod(torch.argmax(y_pred).cpu().numpy()+1,2))

                            if torch.cuda.is_available():
                                y_pred=y_pred.cuda()

                            loss=criterion(y_pred, y_train)

                            loss.backward()
                            optimizer.step()
                            train_loss_curr += loss.item()*y_train.size(0)

                        #tally up total train loss for given speaker
                        if train_loss_curr<1:
                            train_loss+=train_loss_curr/len(train_loader.dataset)
                        else:
                            train_loss+=np.log(train_loss_curr)/len(train_loader.dataset)
                        #calculate number batch frames correctly identified as pd/hc
                        acc=np.count_nonzero(np.array(y_pred_tag)==trainIndc)/len(y_pred_tag)
                        train_acc+=acc
    #                     train_acc+=np.count_nonzero(np.array(final_y_preds.detach().numpy())==trainIndc)/y_pred.shape[0]

                #Record train results at end of each epoch
                trainResults_epo.iloc[epoch]['train_loss'],trainResults_epo.iloc[epoch]['train_acc']=train_loss/85,train_acc/85
                print('Epoch: {} \nTraining Loss: {:.6f} Training Accuracy: {:.2f} \tTime: {:.6f}\n'.format(
                epoch+1,
                train_loss/85,
                train_acc/85,
                time.time()-start
                ))         

                #Validate at end of each epoch for 5 speakers
                y_pred_tag=[]
                val_loss=0.0
                val_acc=0

                for vid in valDict.keys():
                    if vid<num_pd:
                        indc=1
                        opp=0
                    else:
                        indc=0
                        opp=1

                    xVal=valDict[vid]
                    test_data=testData(torch.FloatTensor(xVal))
                    test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=True, shuffle=True) 

                    model.eval()
                    with torch.no_grad():
                        for X_test in test_loader:
                            yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                            if torch.cuda.is_available():
                                X_test=X_test.cuda()

                            y_test_pred = model.forward(X_test)
                            y_pred_tag.append(np.mod(torch.argmax(y_test_pred).cpu().numpy()+1,2))
                            loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                            val_loss+=loss.item()*X_test.size(0)
                    if len(y_pred_tag)>0:
                        if val_loss<1:
                            val_loss=val_loss/len(test_loader.dataset)
                        else:
                            val_loss=np.log(val_loss)/len(test_loader.dataset)

                        acc=np.count_nonzero(np.array(y_pred_tag)==indc)/len(y_pred_tag)
                        val_acc+=acc
#     #                     val_acc=np.count_nonzero(np.array(y_pred_tag)==indc)/len(y_pred_tag)
#                         trainResults_epo['valSpk_data'][vid]=y_test_pred[:,0].cpu().numpy()
                        print('Validation Spk ID (<51 => pd, >50 => hc): {} Spk Frame Accuracy: {:.2f}'.format(
                vid,
                acc,
                ))    
                trainResults_epo.iloc[epoch]['val_loss'],trainResults_epo.iloc[epoch]['val_acc']=val_loss/5.0,val_acc/5.0
                print('\nValidation Loss: {:.6f} Validation Accuracy: {}\n'.format(
                val_loss/5.0,
                val_acc/5.0,
                ))         
            
                trainResultsEpo_curr.append(trainResults_epo)
            
            trainResults_curr=pd.concat((trainResultsEpo_curr),keys=(UTTERS))      

            #AFTER MODEL TRAINED FOR ALL SPEAKERS TEST MODEL ON LEFT OUT SPEAKERS
            y_pred_tag=[]            
            y_pred_ind=[]
            test_loss=0.0
            test_acc=0.0
            testResults_curr[utter]['tstSpk_data']={key:{} for spk in ['pd','hc'] for key in testDict[spk].keys()}

            for spkItr,spk in enumerate(['pd','hc']):
                dic=testDict[spk]

                for tstId in dic.keys():
                    if tstId<num_pd:
                        indc=1
                        opp=0
                    else:
                        indc=0
                        opp=1

                    test_loss_curr=0
                    xTest=dic[tstId]
                    test_data=testData(torch.FloatTensor(xTest))
                    test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=True, shuffle=True)  

                    model.eval()
                    with torch.no_grad():
                        for X_test in test_loader:
                            yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                            if torch.cuda.is_available():
                                X_test=X_test.cuda()
                            
                            y_test_pred = model.forward(X_test)
                            y_pred_tag.append(np.mod(torch.argmax(y_test_pred).cpu().numpy()+1,2))
                            y_pred_ind.append((y_test_pred.cpu().numpy())[:,0])
                            loss = criterion(y_test_pred, torch.from_numpy(yTest).cuda().float())
                            test_loss_curr+=loss.item()*X_test.size(0)

                    if len(y_pred_tag)>0:
                        if test_loss<1:
                            test_loss+=test_loss_curr/len(test_loader.dataset)
                        else:
                            test_loss+=np.log(test_loss_curr)/len(test_loader.dataset)
                        acc=np.count_nonzero(np.array(y_pred_tag)==indc)/len(y_pred_tag)
                        test_acc+=acc
                        
                        ind_acc=np.mean(y_pred_ind)
                        #Store and report percent of speech frames classified correctly (>=50% implies correct)
                        testResults_curr[utter]['tstSpk_data'][spkItr]=acc
                        print('Test Speaker ID# (<51 => pd, >50 => hc): {} \tPD Percent: {:.2f} '.format(
                        tstId+1,
                        ind_acc
                    ))

            #Store and report loss and accuracy for batch of test speakers.            
            testResults_curr[utter]['test_loss'],testResults_curr[utter]['test_acc']=test_loss/10,test_acc/10
            print('\nTest Loss: {:.3f} \tTest Acc: {:.3f} '.format(
                        test_loss/10,
                        test_acc/10
                ))

        train_res.append(trainResults_curr)
        test_res.append(testResults_curr)

            
        trainResults=pd.concat(train_res,keys=(np.arange(itr+1)))
        testResults=pd.concat(test_res,keys=(np.arange(itr+1)))
        trainResults.to_pickle(save_path+mod+'_'+rep+"TrainResults.pkl")
        testResults.to_pickle(save_path+mod+'_'+rep+"TestResults.pkl")
    
    
    
