import torch
import torch.nn as nn
#from base import BaseModel
import torchvision.models as models


class Resnet50Model(nn.Module):
    def __init__(self, num_classes=28):
        super(Resnet50Model, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        w = self.resnet.conv1.weight
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(512 * 4),
            nn.Dropout(0.5),
            nn.Linear(512 * 4, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
    
