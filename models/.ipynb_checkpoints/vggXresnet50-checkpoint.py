import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .inceptionv3 import InceptionV3Model
from .resnet50 import Resnet50Model
from .VGG16 import VGG16Model
from torchvision import models, datasets

from tqdm import tqdm

import os

nclasses = 20 

class Net(nn.Module):
    def __init__(self, n_output=28):
        super(Net, self).__init__()
        
        #create inception v3 network
        self.vgg16 = VGG16Model(mixte=True)

        self.vgg16.vgg16.layer8 = nn.Linear(4096,2048)
        
        #create resnet50
        self.resnet50 = Resnet50Model()

        #self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.resnet.fc = nn.Sequential()
        self.relu = nn.ReLU()
        #add averagepool
        self.pool = nn.AvgPool2d(4)

        self.fc = nn.Linear(4096, n_output)
        
    def forward(self, x):
        y = self.resnet50(x)
        y = self.relu(y)

        a = y.view(-1, 2048)

        b = self.vgg16(x)
        b = b.view(-1,2048)

        z = torch.cat([a,b],axis=1)
        
        return self.fc(z)