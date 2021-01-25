from models.resnet50 import Resnet50Model
from models.VGG16 import VGG16Model
from models.inceptionv3 import InceptionV3Model
from models.incepXresnet50 import Net
from dataloader2 import ProteinDataLoader
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from collections import namedtuple
import numpy as np
from F1 import f1_loss
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Learning Project 2020')
    parser.add_argument('--saved_model', type=str, default=None, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
    parser.add_argument('--model', type=str, default='resnet', metavar='D',
                        help="model to choose to perform training (resnet, vgg16, inceptionv3, mixte)")
    parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    dictio = {'data_dir': "./human-protein-atlas-image-classification/train",
                'csv_path': "./human-protein-atlas-image-classification/train.csv",
                'img_size': 299,
                'batch_size': 32,
                'shuffle': True,
                'validation_split': 0.9,
                'num_workers': 0,
                'num_classes': 28}

    data_loader = ProteinDataLoader(**dictio)
    
    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Data initialization and loading

    val_loader = data_loader.split_validation()
    
    if args.model == 'resnet':
        model = Resnet50Model()
    elif args.model == 'vgg16':
        model = VGG16Model()
    elif args.model == 'inceptionv3':
        model = InceptionV3Model()
    elif args.model == 'mixte':
        model = Net()
    else :
        model == Resnet50Model()

    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    import torch.nn.functional as F


    def focal_loss(logit, target):
        gamma = 1
        target = target.float()
        max_val = (-1*logit).clamp(min=0)
        loss = logit - logit * target + max_val + ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            if use_cuda:
                data, target = data.cuda(), torch.tensor(target, dtype=torch.long, device='cuda')

            optimizer.zero_grad()
            output = model(data)
            #print(target)
            #print(output)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, torch.argmax(target, 1))
            loss_focal = focal_loss(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{:.0f} ({:.0f}%)]\tLoss Cross Entropy: {:.6f}\tLoss Focal: {:.6f} '.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset)*0.9,
                    100. * batch_idx / np.floor(len(data_loader)*0.9), loss.data.item(),loss_focal.data.item()))

    def validation():
        model.eval()
        validation_loss = 0
        loss_focal = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if use_cuda:
                data, target = data.cuda(), torch.tensor(target, dtype=torch.long, device='cuda')
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, torch.argmax(target, 1)).data.item()
            loss_focal += focal_loss(output,target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            #_, pred = torch.max(output, dim=1)
            #print(output.data.max(1, keepdim=True))
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            target = target.data.max(1,keepdim=True)[1]
            correct += pred.eq(target).cpu().sum()


        validation_loss /= (len(val_loader.dataset)*0.1)
        loss_focal /= (len(val_loader.dataset)*0.1)
        print('\nValidation set: Average loss Cross Entropy: {:.4f} Average Loss Focal: {:.4f}, Accuracy: {}/{:.0f} ({:.0f}%)'.format(
            validation_loss, loss_focal, correct, np.floor(len(val_loader.dataset)*0.1),
            100. * correct / np.floor(len(val_loader.dataset)*0.1)))
        return validation_loss

    val_error = []
    if args.saved_model is None:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            loss = validation()
            val_error.append(loss)
            model_file = args.experiment + '/model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_file)
            print('Saved model to ' + model_file)
        plt.plot(range(1, args.epochs+1), val_error)
        plt.xlabel("num_epochs")
        plt.ylabel("Train error")
        plt.title("Visualization of convergence")
        plt.savefig('./images/inception.png')
    
    else:
        model.load_state_dict(torch.load(args.saved_model, map_location=torch.device('cuda')))

 
    from grad_2 import *
    
    if args.model == 'resnet':
        model = model.resnet
        cam = Grad_CAM(model.cuda(),reduc_size=10)
    elif args.model == 'vgg16':
        model = model.vgg16
        cam = Grad_CAM(model.cuda(),reduc_size=9)
    elif args.model == 'inceptionv3':
        model = model.inceptionv3
    else :
        model = model.resnet
        
    if use_cuda:
        model.cuda()
    
    dictio = {'data_dir': "./train",
                'csv_path': "./train-2.csv",
                'img_size': 299,
                'batch_size': 1,
                'shuffle': True,
                'validation_split': 0.15,
                'num_workers': 0,
                'num_classes': 28}
    data_loader = ProteinDataLoader(**dictio)

    inputs, classes = next(iter(data_loader))
    print(classes)
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

    
    if use_cuda:
        heatmaps, values, indices = cam.get_grad_cam(inputs.unsqueeze(0).cuda())
    else:
        heatmaps, values, indices = cam.get_grad_cam(inputs.unsqueeze(0))

    rgb_img = np.transpose(image, [1,0,2])
    fig, suptitle = plot_heatmaps(rgb_img, heatmaps, values, indices)


    

    