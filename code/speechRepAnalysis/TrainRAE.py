from SpecDatset import SpecDataset
import time
import torch
import numpy as np
import pandas as pd
import os
import sys
from RAE import RAEn
from wvRAE import wvRAEn
import toolbox.traintestsplit as tts


def standard(tensor, minval, maxval):
    temp=tensor-minval
    return temp/(maxval-minval)

def destandard(tensor, minval, maxval):
    temp=tensor*(maxval-minval)
    return temp+minval

if __name__=="__main__":
    
    PATH=os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv)!=3:
        print("python TrainRAE.py <bottleneck_sizes> <rep path>")
        sys.exit()
    #repPath: .../reps/wvlt or spec/train/ 
    
    if sys.argv[2][0] !='/':
        sys.argv[2] = '/'+sys.argv[2]
    if sys.argv[2][-1] !='/':
        sys.argv[2] = sys.argv[2]+'/'

    path_rep=PATH+sys.argv[2]
    if 'wvlt' in sys.argv[2]:
        rep_typ='wvlt'
    elif 'spec' in sys.argv[2]:
        rep_typ='spec'
    else:
        print("Please correct directory path input...")
        sys.exit()
    
    if not os.path.exists(path_rep+'/train/') or not os.path.exists(path_rep+'/test/'):
        split=tts.trainTestSplit(path_rep,file_type='.npy', tst_perc=0.1)
        split.fileTrTstSplit()

    elif len(os.listdir(path_rep+'/train/')) == 0:  
        split=tts.trainTestSplit(path_rep,file_type='.npy', tst_perc=0.1)
        split.fileTrTstSplit()
   
    PATH_TRAIN=path_rep+"/train/"
    PATH_TEST=path_rep+"/test/"

    BATCH_SIZE=16
    NUM_W=0
    BOTTLE_SIZE=int(sys.argv[1])
    LR=0.001
    N_EPOCHS = 50
    SCALERS = pd.read_csv("scales.csv")
    MIN_SCALER= float(SCALERS['Min Scale']) #MIN value of total energy.
    MAX_SCALER= float(SCALERS['Max Scale'])  #MAX value of total energy.
    NTRAIN=6000
    NVAL=500

    train=SpecDataset(PATH_TRAIN)
    test=SpecDataset(PATH_TEST)

    save_path = PATH+"/pts/"+rep_typ+'/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
   

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, drop_last=True, num_workers=NUM_W)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, drop_last=True, num_workers=NUM_W)

    if rep_typ=='spec':
        model=RAEn(BOTTLE_SIZE)
    elif rep_typ=='wvlt':
        model=wvRAEn(BOTTLE_SIZE)
    criterion = torch.nn.MSELoss()    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        model.cuda()
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    valid_loss_min = np.Inf # set initial "min" to infinity
    avg_train_losses=[]
    avg_valid_losses=[]

    for epoch in range(N_EPOCHS):
        start=time.time()
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        cdata=0
        model.train() # prep model for training
        for data in train_loader:

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if rep_typ == 'spec':
                data=standard(data, MIN_SCALER, MAX_SCALER)
                
            data=data.float()

            if torch.cuda.is_available():
                data=data.cuda()

            data_out, bottle=model.forward(data)

            if torch.cuda.is_available():
                data_out=data_out.cuda()


            loss = criterion(data_out, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

            cdata+=1

            #if cdata==NTRAIN:
            #    break
        cdata=0
        ######################    
        # validate the model #
        ######################

        model.eval() # prep model for evaluation
        for data_val in test_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            if rep_typ == 'spec':
                data_val=standard(data_val, MIN_SCALER, MAX_SCALER)
                
            data_val=data_val.float()
            if torch.cuda.is_available():
                data_val=data_val.cuda()

            data_val_out, bottle_val=model.forward(data_val)

            if torch.cuda.is_available():
                data_val_out=data_val_out.cuda()

            # calculate the loss
            loss = criterion(data_val_out, data_val)
            # update running validation loss 
            valid_loss += loss.item()*data.size(0)
            cdata+=1
            #if cdata==NVAL:
            #    break
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(test_loader.dataset)


        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTime: {:.6f}'.format(
            epoch+1, 
            train_loss,
            valid_loss,
            time.time()-start
            ))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path+'/'+str(BOTTLE_SIZE)+'_RAE.pt')
            valid_loss_min = valid_loss
        f=open(save_path+'/loss_'+str(BOTTLE_SIZE)+'_RAE.csv', "a")
        f.write(str(train_loss)+", "+str(valid_loss)+"\n")
        f.close()





