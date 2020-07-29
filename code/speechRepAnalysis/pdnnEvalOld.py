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

def saveFeats(model,units,rep,wav_path):
    global UNITS
    global PATH
    
    # load the pretrained model with 256 units and get temp./freq. rep (spec or wvlt)
    aespeech=AEspeech(model=model,units=UNITS,rep=rep) 
    
    #compute the bottleneck and error-based features from a directory with wav files inside 
    #(dynamic: one feture vector for each 500 ms frame)
    #(global i.e. static: one feture vector per utterance)
    feat_vecs=aespeech.compute_dynamic_features(wav_path)
    #     df1, df2=aespeech.compute_global_features(wav_path)
    
#     feat_vecs=np.zeros((df['wav_file'].shape[0],df['error'].shape[1]+df['bottleneck'].shape[1]))
#     for itr,wav in enumerate(df['wav_file']):    
#         feat_vecs[itr,:]=np.concatenate((df['error'][itr], df['bottleneck'][itr]),axis=0)
    
    with open(PATH+'/pdFeats.pickle', 'wb') as handle:
        pickle.dump(feat_vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return feat_vecs
            

def getFeats(model,units,rep,wav_path):
    global PATH
    
    if os.path.isfile(PATH+'/pdFeats.pickle'):
        with open('pdFeats.pickle', 'rb') as handle:
            feat_vecs = pickle.load(handle)
    else:
        feat_vecs=saveFeats(model,units,rep,wav_path)
    
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
        
    
    
    TRAIN_BATCH_SIZE=250
    TEST_BATCH_SIZE=50
    NUM_W=0
    N_EPOCHS=50
    LR=0.1

    pd_path=PATH+sys.argv[3]+"/pd/"
    hc_path=PATH+sys.argv[3]+"/hc/"
    
    pds=[name for name in os.listdir(pd_path) if '.wav' in name]
    hcs=[name for name in os.listdir(hc_path) if '.wav' in name]
    pds.sort()
    hcs.sort()
    pats=pds+hcs
    num_pd=len(pds)
    num_hc=len(hcs)
    total_pats=num_pd+num_hc
    results=pd.DataFrame(index=np.arange(total_pats),columns=np.arange(N_EPOCHS))
    
    pdFeats=getFeats(model,UNITS,rep,pd_path)
    hcFeats=getFeats(model,UNITS,rep,hc_path)
    
    #iterate through all pd and hc speakers for a given utterance (see UTTERS for options) and using leave one out, train a DNN
    #(see pdnn.py) and classify one by one if PD or HC.
    for itr in range(total_pats):
        if itr<num_pd:
            indc=1
            xTrain=np.concatenate(pdFeats)
            
        
        
        
        if itr<num_pd:
            if not os.path.isdir(pd_path+'/test/'):
                os.mkdir(pd_path+'/test/')
            else:
                for file in os.listdir(pd_path+'/test/'):
                    shutil.move(pd_path+'/test/'+file, pd_path+'/'+file)
            shutil.move(pd_path+'/'+pats[itr],pd_path+'/test/'+pats[itr])
            test_path=pd_path+'/test/'
            indc=1 #indicator of pd or hc (1 => pd, 0 => hc)
        
        else:
            if not os.path.isdir(hc_path+'/test/'):
                os.mkdir(hc_path+'/test/')
            else:
                for file in os.listdir(hc_path+'/test/'):
                    shutil.move(hc_path+'/test/'+file, hc_path+'/'+file)
            shutil.move(hc_path+'/'+pats[itr],hc_path+'/test/'+pats[itr])
            test_path=hc_path+'/test/'
            indc=0
        
        #get features for training and test (dimensions depend on number of bottleneck features chosen for 
        #CAE/RAE model plus number of frequency bands used for the spectral or wavelet representations).
#         pdTrain=getFeats(model,UNITS,rep,pd_path)
#         hcTrain=getFeats(model,UNITS,rep,hc_path)
        xTrain=np.concatenate((pdTrain,hcTrain),axis=0)
        xTest=getFeats(model,UNITS,rep,test_path)
        yTrain=np.concatenate((np.ones((pdTrain.shape[0])),np.zeros((hcTrain.shape[0]))),axis=0)
        yTest=np.ones((xTest.shape[0]))*indc
                
        train_data=trainData(torch.FloatTensor(xTrain), torch.FloatTensor(yTrain))
        test_data=testData(torch.FloatTensor(xTest))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_W, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, num_workers=NUM_W, shuffle=True)

        if rep=='spec':
            model=pdn(UNITS+128)
        elif rep=='wvlt':
            model=pdn(UNITS+64)
        criterion=nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr = LR)

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            model.cuda()
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


#         avg_train_losses=[]
#         avg_train_accs=[]

        for epoch in range(N_EPOCHS):
            start=time.time()
            # monitor training loss
            train_loss=0.0
            train_acc=0.0
            test_loss=0.0
            test_acc=0.0
            model.train() # prep model for training
            
            #TRAINING ON SNIPPETS OF SPEAKER UTTERANCES
            for X_train, y_train in train_loader:
                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                X_train=X_train.float()                
                y_train=y_train.long()
                
                if torch.cuda.is_available():
                    X_train,y_train=X_train.cuda(),y_train.cuda()

                y_pred=model.forward(X_train)

                if torch.cuda.is_available():
                    y_pred=y_pred.cuda()
                    
                loss = criterion(y_pred, y_train.long())                      
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss = train_loss/len(train_loader.dataset)
            final_y_pred=torch.argmax(F.softmax(y_pred),dim=1)
            train_acc=np.count_nonzero(final_y_pred.detach().numpy()==indc)/len(train_loader.dataset)
            
            #IMPLEMENT: FP/TP/FN/TN       
        
            #TEST MODEL ON LEFT OUT SPEAKER SNIPPETS
            model.eval()
            with torch.no_grad():
                for X_test in test_loader:
                    
                    if torch.cuda.is_available():
                        X_test=X_test.cuda()
                        
                    y_test_pred = model.forward(X_test)
                    y_pred_tag = torch.argmax(F.softmax(y_test_pred)).numpy()
                    loss = criterion(y_test_pred, torch.from_numpy(yTest).long())
                    test_loss+=loss.item()
           
            test_loss = test_loss/len(test_loader.dataset)
            final_test_pred=np.count_nonzero(y_pred_tag==indc)/len(test_loader.dataset)
            
            if epoch==N_EPOCHS-1:
                results[itr][epoch]={'train loss':train_loss, 'train_acc':train_acc, 'test loss':test_loss, 
                                 'test acc':final_test_pred, 'pred list':y_pred_tag}
            else:
                results[itr][epoch]={'train loss':train_loss, 'train_acc':train_acc, 'test loss':test_loss, 
                                 'test acc':final_test_pred}
                
            print('Spkr: {} \tEpoch: {} \nTraining Loss: {:.6f} \tTraining Acc: {:.6f} \tTest Loss: {:.6f} \nTest Pred (>0.5 => pd, <0.5 => hc): {:.6f} \tTime: {:.6f}\n'.format(
                itr,
                epoch,
                train_loss,
                train_acc,
                test_loss,
                final_test_pred,
                time.time()-start
                ))

        results.to_pickle(PATH+"/dnnResults.pkl")
         
    
    
    