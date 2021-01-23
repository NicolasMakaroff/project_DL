import torch.nn.functional as F
from torch.autograd import grad
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from dataloader2 import ProteinDataLoader

class Grad_CAM():
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradient = None
        
        # register hooks to capture the feature_map gradients
        def forward_hook(model, input, output):
            self.activations = output[0]
        def backward_hook(model, grad_input, grad_output):
            self.gradient = grad_output[0][0]
            
        feat_map = model.Mixed_7c#[29]
        feat_map.register_forward_hook(forward_hook)
        feat_map.register_backward_hook(backward_hook)
        
    def get_grad_cam(self, img):
        print(img.shape)
        out = self.model(img)
        num_features = self.activations.size()[0]
        topk = 3
        values, indices = torch.topk(out, topk)
        # Compute 14x14 heatmaps
        heatmaps = torch.zeros(topk,36,36)
        self.model.Mixed_7c.zero_grad()
        self.model.linear.zero_grad()
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
        large_heatmaps = F.interpolate(heatmaps.expand((1,topk,36,36)), (299,299), mode='bilinear')
        return large_heatmaps[0].data.numpy(), values.data.numpy()[0], indices.data.numpy()[0]
    
    
import matplotlib.pyplot as plt
from tensorflow.keras.applications.imagenet_utils import decode_predictions
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


def get_titles(values, indices):
    predictions = np.zeros((28,))
    predictions[indices] = values
    decoded = np.argwhere(predictions.argsort()[-3:][::-1])
    class_number = predictions.argsort()[-3:][::-1]
    print(predictions)
    print(decoded)
    #decoded = decode_predictions(predictions, top=3)
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
            title += '    score '+str(np.round(values[k],2))
        ax.set_title(title, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(h, ax=ax, )
        cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    
    titles = get_titles(values, indices)
    suptitle = 'Top 1: '+titles[0]+'\nTop 2: '+titles[1]+'\nTop 3: '+titles[2]
    fig.suptitle(suptitle, x=0, y=1.2, fontsize=20, ha='left', va='top')
    return fig, suptitle


from models.VGG16 import VGG16Model
from models.inceptionv3 import InceptionV3Model

vgg_model = InceptionV3Model()
cam = Grad_CAM(vgg_model.inceptionv3)

dictio = {'data_dir': "./human-protein-atlas-image-classification/train",
            'csv_path': "./human-protein-atlas-image-classification/train.csv",
            'img_size': 299,
            'batch_size': 1,
            'shuffle': True,
            'validation_split': 0.15,
            'num_workers': 0,
            'num_classes': 28}
data_loader = ProteinDataLoader(**dictio)

inputs, classes = next(iter(data_loader))
inputs = inputs.squeeze(0)
R = np.array(inputs[0].cpu())
G = np.array(inputs[1].cpu())
B = np.array(inputs[2].cpu())
Y = np.array(inputs[3].cpu())

image = np.stack((
    R,
    G, 
    (B+Y)/2),-1)

import cv2
image = cv2.resize(image, (299, 299))
image = np.divide(image, 255)
image = np.float32(image)
image = image / np.max(image)
image = np.uint8(255 * image)
print(inputs.unsqueeze(0).shape)
heatmaps, values, indices = cam.get_grad_cam(inputs.unsqueeze(0))
print(image.shape)
rgb_img = np.transpose(image, [1,0,2])
fig, suptitle = plot_heatmaps(rgb_img, heatmaps, values, indices)
fig.show()
plt.title(suptitle)
plt.show()

