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
    
    with open(save_path+'/'+utter+'_'+spk_typ+'Feats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return feat_vecs
            

def getFeats(model,units,rep,wav_path,utter,spk_typ):
    global PATH
    save_path=PATH+"/"+"pdSpanish/feats/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
        
    if os.path.isfile(save_path+'/'+utter+'_'+spk_typ+'Feats.pickle'):
        with open(save_path+'/'+utter+'_'+spk_typ+'Feats.pickle', 'rb') as handle:
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
    #TRAIN_PATH: './pdSpanish/<UTTER>/'
    
    
    if sys.argv[1] in MODELS:
        model=sys.argv[1]
    else:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()
    
    if sys.argv[2] in REPS:
        rep=sys.argv[2]
    else:
        print("python dnnTrain.py <'CAE','RAE', or 'ALL'> <'spec' or 'wvlt'> <pd path>")
        sys.exit()    
    
#     print(sys.argv[3].split('/'))
    if sys.argv[3].split('/')[-2] in UTTERS:
        utter=sys.argv[3].split('/')[-2]   
    else:
        print("Please correct directory path input... './pdSpanish/<UTTERS>'")
        sys.exit()     
        
    if sys.argv[3][0] !='/':
        sys.argv[3] = '/'+sys.argv[3]
    if sys.argv[3][-1] !='/':
        sys.argv[3] = sys.argv[3]+'/'
        
  
    BATCH_SIZE=10
    NUM_W=0
    N_EPOCHS=1000
    LR=0.1
    if rep=='spec':
        NBF=128
    else:
        NBF=64
    
    pd_path=PATH+sys.argv[3]+"/pd/"
    hc_path=PATH+sys.argv[3]+"/hc/"   
    pds=[name for name in os.listdir(pd_path) if '.wav' in name]
    hcs=[name for name in os.listdir(hc_path) if '.wav' in name]
    pds.sort()
    hcs.sort()
    spks=pds+hcs
    num_pd=len(pds)
    num_hc=len(hcs)
    total_spks=num_pd+num_hc
    
    pdFeats=getFeats(model,UNITS,rep,pd_path,utter,'pd')
    hcFeats=getFeats(model,UNITS,rep,hc_path,utter,'hc')
    
    trainResults=pd.DataFrame(index=np.arange(total_spks),columns=np.arange(total_spks-1))
    testResults=pd.DataFrame(index=np.arange(total_spks))
    rand_range=np.arange(total_spks)
    random.shuffle(rand_range)
    
    #iterate through all pd and hc speakers for a given utterance (see UTTERS for options) and using leave one out, train a DNN
    #(see pdnn.py) and classify one by one if PD or HC.
    for itr in range(total_spks):
        if itr<num_pd:
            indc=1
            opp=0
            feats=pdFeats
        else:
            indc=0
            opp=1
            feats=hcFeats
            
        #RESET model that will be trained on all speakers but one (must reset so test speaker data not used in model).
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
        bns=feats['bottleneck'][np.where(feats['wav_file']==spks[itr])]
        errs=feats['error'][np.where(feats['wav_file']==spks[itr])]
        xTest=np.concatenate((bns,errs),axis=1)
        
        test_data=testData(torch.FloatTensor(xTest))
        test_loader=torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_W, drop_last=True, shuffle=True)  
        trainCntr=0
        
        #TRAIN dnn for each speaker individually for given number of epochs (order is random between pd, hc patients).              
        for trainItr in rand_range:                       
            if trainItr==itr:
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
            train_loss=0.0
#             val_loss=0.0

            for epoch in range(N_EPOCHS):    
                model.train() # prep model for training
                ind_train_loss=0
                if len(train_loader)>0:
                    #TRAINING ON SNIPPETS OF SPEAKER UTTERANCES
                    for X_train, y_train in train_loader:

                        # clear the gradients of all optimized variables
                        optimizer.zero_grad()

                        X_train=X_train.float()                
                        y_train=y_train.float()

                        if torch.cuda.is_available():
                            X_train,y_train=X_train.cuda(),y_train.cuda()

                        y_pred=model.forward(X_train)

                        if torch.cuda.is_available():
                            y_pred=y_pred.cuda()

                        loss=criterion(y_pred, y_train)

                        loss.backward()
                        optimizer.step()
                        if epoch==N_EPOCHS-1:
                            train_loss += loss.item()*y_train.size(0)

                    ind_train_loss=train_loss/len(train_loader)
                    if epoch==0:
                        trainResults[itr][trainCntr]={"epoch losses":[ind_train_loss]}
                    else:
                        trainResults[itr][trainCntr]["epoch losses"]=trainResults[itr][trainCntr]["epoch losses"]+[ind_train_loss]

            if train_loss<1:
                train_loss=train_loss/(len(train_loader)*N_EPOCHS)
            else:
                train_loss=np.log(train_loss)/(len(train_loader))
            final_y_preds=np.mod(torch.argmax(y_pred,dim=1)+1,2)
            train_acc=np.count_nonzero(final_y_preds.detach().numpy()==trainIndc)/y_pred.shape[0]
            trainResults[itr][trainCntr]={'train loss':train_loss, 'train_acc':train_acc}
            trainCntr+=1
            print('Test Speaker ID#(<51 => pd, >50 => hc): {} \t Train Speaker ID#: {} \nTraining Loss: {:.6f} \nTraining Pred (>0.5 => pd, <0.5 => hc): {:.2f} \tTime: {:.6f}\n'.format(
            itr+1,
            trainItr+1,
            train_loss,
            train_acc,
            time.time()-start
            ))         
#             #VALIDATE on random speaker (not test)
#             model.eval()
#             yVal_pred_tag=[]
#             with torch.no_grad():
#                 for X_val in val_loader:

#                     yVal=np.vstack((np.ones((X_val.shape[0]))*valIndc,np.ones((X_val.shape[0]))*valOpp)).T
#                     if torch.cuda.is_available():
#                         X_val=X_val.cuda()

#                     y_val_pred = model.forward(X_val)
#                     yVal_pred_tag.append(np.mod(torch.argmax(y_val_pred).numpy()+1,2))
#                     loss = criterion(y_val_pred, torch.from_numpy(yVal).float())
#                     val_loss+=loss.item()

#             val_loss = val_loss/len(val_loader.dataset)
#             final_val_pred=np.count_nonzero(yVal_pred_tag==valIndc)/len(yVal_pred_tag)
#             trainResults[itr][trainCntr]={'train loss':train_loss, 'train_acc':train_acc, 'val loss':val_loss, 
#                              'val acc':final_val_pred}
#             trainCntr+=1
#             print('Test Speaker #: {} \t Train Speaker #: {} \nTraining Loss: {:.6f} \Val Loss: {:.6f} \nTraining Pred (>0.5 => pd, <0.5 => hc): {:.2f} \Val Pred: {:.2f} \tTime: {:.6f}\n'.format(
#             itr,
#             trainItr,
#             train_loss,
#             train_acc,
#             val_loss,
#             final_val_pred,
#             time.time()-start
#             ))

        #IMPLEMENT: FP/TP/FN/TN       

        #AFTER MODEL TRAINED FOR ALL SPEAKERS TEST MODEL ON LEFT OUT RANDOM OTHER SPEAKER SNIPPETS
        y_pred_tag=[]
        test_loss=0.0
        model.eval()
        with torch.no_grad():
            for X_test in test_loader:
                yTest=np.vstack((np.ones((X_test.shape[0]))*indc,np.ones((X_test.shape[0]))*opp)).T
                if torch.cuda.is_available():
                    X_test=X_test.cuda()

                y_test_pred = model.forward(X_test)
                y_pred_tag.append(np.mod(torch.argmax(y_test_pred).numpy()+1,2))
                loss = criterion(y_test_pred, torch.from_numpy(yTest).float())
                test_loss+=loss.item()*X_test.size(0)
        
        if test_loss<1:
            test_loss=test_loss/len(test_loader.dataset)
        else:
            test_loss=np.log(test_loss)/len(test_loader.dataset)
        test_acc=np.count_nonzero(y_pred_tag==indc)/len(y_pred_tag)

        testResults[itr]={'train loss':train_loss, 'train_acc':train_acc, 'test loss':test_loss, 'test acc':test_acc}

        print('Test Speaker ID# (<51 => pd, >50 => hc): {} \tTest Loss: {:.6f} \tTest Pred (>0.5 => pd, <0.5 => hc): {:.2f} \tTime: {:.6f}\n'.format(
            itr,
            test_loss,
            final_test_pred,
            time.time()-start
            ))
    trainResults.to_pickle(PATH+"/pdSpanish/feats/dnnTrainResults.pkl")
    testResults.to_pickle(PATH+"/pdSpanish/feats/dnnTestResults.pkl")
    
    
    