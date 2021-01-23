import torch
import torch.nn as nn
import torchvision.models as models
from .base_models.vgg16 import *


class VGG16Model(nn.Module):
    def __init__(self, num_classes=28):
        super(VGG16Model, self).__init__()
        self.vgg16 = VGG16()

    def forward(self, x):
        return self.vgg16(x)
    

