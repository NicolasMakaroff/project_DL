from models.resnet50 import Resnet50Model
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
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
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    dictio = {'data_dir': "./human-protein-atlas-image-classification/train",
                'csv_path': "./human-protein-atlas-image-classification/train.csv",
                'img_size': 225,
                'batch_size': 64,
                'shuffle': True,
                'validation_split': 0.1,
                'num_workers': 0,
                'num_classes': 28}

    data_loader = ProteinDataLoader(**dictio)
    print(len(data_loader.dataset))
    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Data initialization and loading
    

    val_loader = data_loader.split_validation()
    print(len(data_loader.dataset))
    print(len(val_loader.dataset))


    #pretrained_model = models.resnet50(pretrained = True)
    #IN_FEATURES = pretrained_model.fc.in_features 
    #OUTPUT_DIM = 200

    #fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    #pretrained_model.fc = fc
    #model = ResNet50Model(resnet50_config, OUTPUT_DIM)
    #model.load_state_dict(pretrained_model.state_dict())
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model = Resnet50Model()
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    import torch.nn.functional as F


    def focal_loss(logit, target):
        gamma = 2
        target = target.float()
        max_val = (-logit).clamp(min=0)
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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.data.item()))

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
            print(batch_idx)

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))



    for epoch in range(1, args.epochs + 1):
        #train(epoch)
        validation()
        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')


    