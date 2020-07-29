import torch
from torch import nn, optim
import torch.nn.functional as F


class pdn(nn.Module):
    def __init__(self,M):
        super().__init__()
        self.fc1=nn.Linear(M,M//2)
        self.fc2=nn.Linear(M//2,M//2)
        self.fc3=nn.Linear(M//2,2)
        self.drop=nn.Dropout(p=.5)
                
    def forward(self, x):
        M = len(x)
        x=self.fc1(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=F.softmax(x)
        return x