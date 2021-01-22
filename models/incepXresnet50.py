import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .inceptionv3 import InceptionV3Model
from .resnet50 import Resnet50Model
from torchvision import models, datasets

from tqdm import tqdm

import os

nclasses = 20 

class Net(nn.Module):
    def __init__(self, n_output=28):
        super(Net, self).__init__()
        
        #create inception v3 network
        self.inception = InceptionV3Model()
        self.inception.aux_logits = False
        
        #only perform training on the last two layers
        """for child in list(self.inception.children())[:-2]:
            for param in child.parameters():
                param.requires_grad = False"""
        self.inception.fc = nn.Sequential()
        
        #create resnet50
        self.resnet50 = Resnet50Model()
        """#only perform training on the last layer
        for child in list(self.resnet50.children())[:-3]:
            for param in child.parameters():
                param.requires_grad = False"""
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        self.relu = nn.ReLU()
        #add averagepool
        self.pool = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(4096, n_output)
        
    def forward(self, x):
        x = self.resnet50(x)
        print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        #a = self.pool(self.relu(self.resnet50(x)))
        a = x.view(-1, 2048)
        
        b = self.inception(x)
        b = b.view(-1,2048)
        
        y = torch.cat([a,b],axis=1)
        
        return self.fc(y)