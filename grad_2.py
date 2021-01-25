import torch.nn.functional as F
from torch.autograd import grad
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image


class Grad_CAM():
    def __init__(self, model, reduc_size):
        self.model = model
        self.activations = None
        self.gradient = None
        
        # register hooks to capture the feature_map gradients
        def forward_hook(model, input, output):
            self.activations = output[0]#.cuda()
        def backward_hook(model, grad_input, grad_output):
            self.gradient = grad_output[0][0]#.cuda()
            
        feat_map = model.layer4#[29]
        feat_map.register_forward_hook(forward_hook)
        feat_map.register_backward_hook(backward_hook)
        
        self.reduc_size = reduc_size
    def get_grad_cam(self, img):
        self.model.eval()
        print(img.shape)
        out = self.model(img)
        num_features = self.activations.size()[0]
        topk = 3
        values, indices = torch.topk(out, topk)
        # Compute 14x14 heatmaps
        heatmaps = torch.zeros(topk,self.reduc_size,self.reduc_size)#.cuda()
        self.model.layer4.zero_grad()
        self.model.fc.zero_grad()
        for i,c in enumerate(indices[0]):
            self.model.zero_grad()
            out[0,c].backward(retain_graph=True)
            # feature importance
            feature_importance = self.gradient.mean(dim=[1,2])
            # pixel importance
            for f in range(num_features):
                heatmaps[i] += feature_importance[f] * self.activations[f]
            heatmaps[i] = F.relu(heatmaps[i])
        
        # Upsample to 224x224
        large_heatmaps = F.interpolate(heatmaps.expand((1,topk,self.reduc_size,self.reduc_size)), (299,299), mode='bilinear')
        return large_heatmaps[0].data.cpu().numpy(), values.data.cpu().numpy()[0], indices.data.cpu().numpy()[0]
    
    


class_names = [
"Nucleoplasm", 
"Nuclear membrane",   
"Nucleoli",   
"Nucleoli fibrillar center" ,  
"Nuclear speckles",
"Nuclear bodies",
"Endoplasmic reticulum",   
"Golgi apparatus",
"Peroxisomes",
"Endosomes",
"Lysosomes",
"Intermediate filaments",   
"Actin filaments",
"Focal adhesion sites",   
"Microtubules",
"Microtubule ends",   
"Cytokinetic bridge",   
"Mitotic spindle",
"Microtubule organizing center",  
"Centrosome",
"Lipid droplets",   
"Plasma membrane",   
"Cell junctions", 
"Mitochondria",
"Aggresome",
"Cytosol",
"Cytoplasmic bodies",   
"Rods & rings" 
]


def set_titles(values, indices):
    predictions = np.zeros((28,))
    predictions[indices] = values
    decoded = np.argwhere(predictions.argsort()[-3:][::-1])
    class_number = predictions.argsort()[-3:][::-1]
    return [class_names[class_number[0]],class_names[class_number[1]],class_names[class_number[2]]]

def plot_heatmaps(img, heatmaps, values = None, indices = None):
    fig, axs = plt.subplots(figsize=(18, 5), ncols=3)
    for k in range(3):
        ax = axs[k]
        ax.imshow(img)
        h = ax.imshow(heatmaps[k], cmap='jet', alpha=0.4)        
        title = 'Top '+str(k+1)
        if values is not None and indices is not None:
            title += '\nclass '+str(indices[k])
        ax.set_title(title, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(h, ax=ax, )
        cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    
    titles = set_titles(values, indices)
    suptitle = 'Top 1: '+titles[0]+'\nTop 2: '+titles[1]+'\nTop 3: '+titles[2]
    plt.suptitle(suptitle)
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.99, bottom=0.0, top=0.5)
    plt.savefig('./images/cam.png', pad_inches=1)
    return fig, suptitle


