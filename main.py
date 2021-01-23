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
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
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
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    dictio = {'data_dir': "./train",
                'csv_path': "./train-2.csv",
                'img_size': 299,
                'batch_size': 32,
                'shuffle': True,
                'validation_split': 0.1,
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
        gamma = 2
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
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
 
            loss = focal_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{:.0f} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset)*0.9,
                    100. * batch_idx / np.floor(len(data_loader)*0.9), loss.data.item()))

    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss

            validation_loss += focal_loss(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            #_, pred = torch.max(output, dim=1)
            #print(output.data.max(1, keepdim=True))
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            target = target.data.max(1,keepdim=True)[1]
            correct += pred.eq(target).cpu().sum()


        validation_loss /= len(val_loader.dataset)*0.1
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{:.0f} ({:.0f}%)'.format(
            validation_loss, correct, np.floor(len(val_loader.dataset)*0.1),
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

    """from gradcam import *
    
    if args.model == 'resnet':
        grad_cam = GradCam(model=model.resnet, feature_module=model.resnet.layer4, \
                       target_layer_names=["2"], use_cuda=use_cuda)
    elif args.model == 'vgg16':
        grad_cam = GradCam(model=model.vgg16, feature_module=model.vgg16.layer5, \
                       target_layer_names=["2"], use_cuda=use_cuda)
    elif args.model == 'inceptionv3':
        grad_cam = GradCam(model=model.inceptionv3, feature_module=model.inceptionv3.Mixed_7c, \
                       target_layer_names=["2"], use_cuda=use_cuda)
    elif args.model == 'mixte':
        grad_cam = GradCam(model=model.resnet, feature_module=model.resnet.layer4, \
                       target_layer_names=["2"], use_cuda=use_cuda)
    else :
        grad_cam = GradCam(model=model.resnet, feature_module=model.resnet.layer4, \
                       target_layer_names=["2"], use_cuda=use_cuda)


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

    if use_cuda:
        inputs = inputs.cuda()

    target_category = None

    grayscale_cam = grad_cam(inputs, target_category)
    inputs = inputs.squeeze(0)

    R = np.array(inputs[0].cpu())
    G = np.array(inputs[1].cpu())
    B = np.array(inputs[2].cpu())
    Y = np.array(inputs[3].cpu())

    image = np.stack((
        R,
        G, 
        (B+Y)/2),-1)

    image = cv2.resize(image, (299, 299))
    image = np.divide(image, 255)
    
    grayscale_cam = cv2.resize(grayscale_cam, (299, 299))
    cam = show_cam_on_image(image, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model.resnet, use_cuda=use_cuda)
    gb = gb_model(inputs.unsqueeze(0), target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite("./images/cam_resnet.jpg", cam)
    cv2.imwrite('./images/gb_resnet.jpg', gb)
    cv2.imwrite('./images/cam_gb_resnet.jpg', cam_gb)
    image = np.float32(image)
    image = image / np.max(image)
    image = np.uint8(255 * image)
    cv2.imwrite('./images/image.jpg',image)"""
    
    from grad import *
    
    if args.model == 'resnet':
        model = model.resnet
    elif args.model == 'vgg16':
        model = model.vgg16
    elif args.model == 'inceptionv3':
        model = model.inceptionv3
    elif args.model == 'mixte':
        model = model.resnet
    else :
        model = model.resnet
        
    cam = Grad_CAM(model)

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
    plt.savefig('./images/cam.png')

    