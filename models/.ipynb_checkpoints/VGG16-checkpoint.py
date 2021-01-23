import torch
import torch.nn as nn
import torchvision.models as models
from .base_models.vgg16 import *


class VGG16Model(nn.Module):
    def __init__(self, num_classes=28):
        super(VGG16Model, self).__init__()
        self.vgg16 = VGG16()

        #self.vgg16 = models.vgg16(pretrained=False)
        """self.vgg16.layer1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )"""


        """w = self.vgg16.features[0].weight
        self.vgg16.features[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.vgg16.features[0].weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        #self.vgg16.add_module(module=nn.Linear(list(self.resnet.children())[-1][-1].in_features, 75), name='fc')
        #self.resnet.classifier[6] = nn.Linear(4096,num_classes)
        self.vgg16.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000,num_classes)
            )"""
        """nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )"""

    def forward(self, x):
        return self.vgg16(x)
    

