# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 02:27:29 2018

@author: james
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:00:40 2018

@author: JamesChiou
"""
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import math
import matplotlib.pyplot as plt

# Define dataset class
class ImageDataset(Dataset):
    def __init__(self, file_path, transform = None):
        df = pd.read_csv(file_path)
        if 'label' in df.columns:
            self.is_train = True
        else:
            self.is_train = False
        
        if self.is_train:
            # training data
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = torch.from_numpy(df.iloc[:,0].values)
        else:
            # test data
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = None
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])

# Define WideResNet network
##############################
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model

# Model save dir
if not os.path.isdir('models'):
    os.mkdir('models')
modeldir = 'models'

# Use cuda or cpu
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Fix random seed for reproducibility
randomSeed = 2018
random.seed(randomSeed)
torch.manual_seed(randomSeed)
np.random.seed(randomSeed)

# Best valid accuracy
best_acc = 0
best_epoch = 0

# Training parameters
n_epoches = 300

# Record
losses = np.zeros((n_epoches))
valid_losses = np.zeros((int(n_epoches/2)))
valid_accuracy = np.zeros((int(n_epoches/2),2))
y_pred = []

def main():
    global best_acc,best_epoch,losses,valid_losses,valid_accuracy
    global n_epoches
    global y_pred
    
    # Dataset transforms
    transform1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=1.),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    transform2 = transforms.Compose([
                 transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])
    
    # Load dataset
    print('Start loading data')
    test_dataset1 = ImageDataset('data/test.csv', transform1)
    test_dataset2 = ImageDataset('data/test.csv', transform2)
    
    # Load dataloader
    testloader1 = DataLoader(test_dataset1,batch_size=100,
                            num_workers=2,shuffle=False,pin_memory=True)
    testloader2 = DataLoader(test_dataset2,batch_size=100,
                            num_workers=2,shuffle=False,pin_memory=True)
    
    y_pred_probs = []
    print('Start predict')
    #1
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.,).to(device)
    best_para = torch.load('models/best/CC_epoch_258_nodropout.pth')
    y_pred_prob1 = test(testloader1,model,None,device,best_para,best_epoch)
    y_pred_prob2 = test(testloader2,model,None,device,best_para,best_epoch)
    y_pred_prob = (y_pred_prob1+y_pred_prob2)/2
    y_pred_probs.append(y_pred_prob)
    print('predict complete: 1')
    #2
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_184_1211_9602.pth')
    y_pred_prob1 = test(testloader1,model,None,device,best_para,best_epoch)
    y_pred_prob2 = test(testloader2,model,None,device,best_para,best_epoch)
    y_pred_prob = (y_pred_prob1+y_pred_prob2)/2
    y_pred_probs.append(y_pred_prob)
    print('predict complete: 2')
    #3
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_192_1212_9580.pth')
    y_pred_prob1 = test(testloader1,model,None,device,best_para,best_epoch)
    y_pred_prob2 = test(testloader2,model,None,device,best_para,best_epoch)
    y_pred_prob = (y_pred_prob1+y_pred_prob2)/2
    y_pred_probs.append(y_pred_prob)
    print('predict complete: 3')
    #4
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_240_1217_9595_pseudolabel.pth')
    y_pred_prob1 = test(testloader1,model,None,device,best_para,best_epoch)
    y_pred_prob2 = test(testloader2,model,None,device,best_para,best_epoch)
    y_pred_prob = (y_pred_prob1+y_pred_prob2)/2
    y_pred_probs.append(y_pred_prob)
    print('predict complete: 4')
    
    y_pred_probs_mean = (y_pred_probs[0]+y_pred_probs[1]+y_pred_probs[2]+y_pred_probs[3])/4
    y_pred_probs_mean = y_pred_probs_mean.cpu().data.numpy()
    y_pred = np.argmax(y_pred_probs_mean, axis=1)
    sub = pd.DataFrame(y_pred, columns=['label'])
    sub.index.name='id'
    sub.to_csv('ensemble_4_hflip.csv', index=True)

def test(testloader,model,criterion,device,best_param,best_epoch):
    # custom select model
    model.load_state_dict(best_param)
    model.eval()
    y_pred_prob = []
    for i_batch, data in enumerate(testloader):
        images = data.to(device)
        outputs = model(images).detach()
        prob = torch.nn.Softmax(dim=1)(outputs)
        y_pred_prob.append(prob)
    
    y_pred_prob = torch.cat(y_pred_prob)
    #print(y_pred_prob)
    return y_pred_prob
    '''
    y_pred = y_pred.astype(int)
    sub = pd.DataFrame(y_pred, columns=['label'])
    sub.index.name='id'
    sub.to_csv('answer_wrn_28_10_%d.csv'%best_epoch, index=True)
    '''

if __name__ == '__main__':
    main()