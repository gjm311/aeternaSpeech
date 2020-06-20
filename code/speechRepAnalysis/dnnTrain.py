import torch
from torch import nn, optim
import torch.nn.functional as F


class dnnTrain(nn.Module):
    def __init__(self,M):
        super().__init__()
        self.fc1=nn.Linear(M,M)
        self.fc2=nn.Linear(M,M)
        self.drop=nn.Dropout(p=.5)
        self.bn=nn.BatchNorm2d(M)

    def forward(self, x):
        M = len(x)
        x=self.bn(self.fc1(x))
        x=self.drop(x)
        x =self.bn(self.fc2(x))
        x=F.Softmax(x)
        
        return x