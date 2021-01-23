import torch
import torch.nn as nn
import torchvision.models as models


class InceptionV3Model(nn.Module):
    def __init__(self, num_classes=28):
        super(InceptionV3Model, self).__init__()
        self.inceptionv3 = models.inception_v3(pretrained=False)

        w = self.inceptionv3.Conv2d_1a_3x3.conv.weight
        self.inceptionv3.Conv2d_1a_3x3.conv = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.inceptionv3.Conv2d_1a_3x3.conv.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        #self.inceptionv3.add_module(module=nn.Linear(list(self.resnet.children())[-1][-1].in_features, 75), name='fc')

        self.inceptionv3.fc = nn.Linear(512 * 4, num_classes)

        
    def forward(self, x):
        return self.inceptionv3(x)