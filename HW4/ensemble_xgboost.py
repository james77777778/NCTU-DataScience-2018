# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:22:30 2018

@author: james
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import math

from xgboost import XGBClassifier

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

class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.from_numpy(np.arange(len(self.indices))))

    def __len__(self):
        return len(self.indices)

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

# Record
y_pred = []
xgb_pred2 = []
def main():
    global y_pred,xgb_pred2
    
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
    train_dataset = ImageDataset('data/train.csv', transform2)
    valid_dataset = ImageDataset('data/train.csv', transform2)
    test_dataset = ImageDataset('data/test.csv', transform2)
    
    # Split train and validation
    valid_ratio = 0.1
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio*num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetSequentialSampler(train_idx)
    valid_sampler = SubsetSequentialSampler(valid_idx)
    
    # Load dataloader
    trainloader = DataLoader(train_dataset,batch_size=100,num_workers=2,
                              sampler=train_sampler,pin_memory=True)
    validloader = DataLoader(valid_dataset,batch_size=100,num_workers=2,
                              sampler=valid_sampler,pin_memory=True)
    testloader = DataLoader(test_dataset,batch_size=100,num_workers=2,
                             shuffle=False,pin_memory=True)
    
    ###########################################################################
    y_pred_probs = []
    print('Start predict for xgboost')
    #1
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.,).to(device)
    best_para = torch.load('models/best/CC_epoch_258_nodropout.pth')
    y_pred_prob1 = predict_train(trainloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 1')
    #2
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_184_1211_9602.pth')
    y_pred_prob1 = predict_train(trainloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 2')
    #3
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_192_1212_9580.pth')
    y_pred_prob1 = predict_train(trainloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 3')
    #4
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_240_1217_9595_pseudolabel.pth')
    y_pred_prob1 = predict_train(trainloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 4')
    #5
    model = wrn(num_classes=10,
                depth=40,
                widen_factor=4,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_142_1217_wrn404.pth')
    y_pred_prob1 = predict_train(trainloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 5')
    #6
    model = wrn(num_classes=10,
                depth=40,
                widen_factor=4,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_286_1217_wrn404_pseudolabel.pth')
    y_pred_prob1 = predict_train(trainloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 6')
    
    xgb_train_probs = torch.cat(y_pred_probs,dim=1).cpu().data.numpy()
    train_data = pd.read_csv('data/train.csv')
    train_data = train_data.iloc[train_idx]
    xgb_train_label = train_data.iloc[:,0].values.ravel()
    
    xgbc1 = XGBClassifier(learning_rate=0.1,n_estimators=150)
    xgbc1.fit(xgb_train_probs, xgb_train_label)
    xgbc2 = XGBClassifier(learning_rate=0.1,n_estimators=300)
    xgbc2.fit(xgb_train_probs, xgb_train_label)
    xgbc3 = XGBClassifier(learning_rate=0.1,n_estimators=400)
    xgbc3.fit(xgb_train_probs, xgb_train_label)
    xgbc4 = XGBClassifier(learning_rate=0.1,n_estimators=500)
    xgbc4.fit(xgb_train_probs, xgb_train_label)
    xgbc5 = XGBClassifier(learning_rate=0.1,n_estimators=600)
    xgbc5.fit(xgb_train_probs, xgb_train_label)
    
    ##########################################################################
    y_pred_probs = []
    print('Start predict for validation')
    #1
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.,).to(device)
    best_para = torch.load('models/best/CC_epoch_258_nodropout.pth')
    y_pred_prob1 = predict_train(validloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 1')
    #2
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_184_1211_9602.pth')
    y_pred_prob1 = predict_train(validloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 2')
    #3
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_192_1212_9580.pth')
    y_pred_prob1 = predict_train(validloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 3')
    #4
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_240_1217_9595_pseudolabel.pth')
    y_pred_prob1 = predict_train(validloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 4')
    #5
    model = wrn(num_classes=10,
                depth=40,
                widen_factor=4,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_142_1217_wrn404.pth')
    y_pred_prob1 = predict_train(validloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 5')
    #6
    model = wrn(num_classes=10,
                depth=40,
                widen_factor=4,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_286_1217_wrn404_pseudolabel.pth')
    y_pred_prob1 = predict_train(validloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 6')
    
    xgb_valid_probs = torch.cat(y_pred_probs,dim=1).cpu().data.numpy()
    valid_data = pd.read_csv('data/train.csv')
    valid_data = valid_data.iloc[valid_idx]
    xgb_valid_label = valid_data.iloc[:,0].values.ravel()
    
    #xgb_pred = xgbc.predict(xgb_valid_probs)
    print('XGB1 ensemble accuracy:', xgbc1.score(xgb_valid_probs, xgb_valid_label))
    print('XGB2 ensemble accuracy:', xgbc2.score(xgb_valid_probs, xgb_valid_label))
    print('XGB3 ensemble accuracy:', xgbc3.score(xgb_valid_probs, xgb_valid_label))
    print('XGB4 ensemble accuracy:', xgbc4.score(xgb_valid_probs, xgb_valid_label))
    print('XGB5 ensemble accuracy:', xgbc5.score(xgb_valid_probs, xgb_valid_label))
    
    ###########################################################################
    y_pred_probs = []
    print('Start predict for submission')
    #1
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.,).to(device)
    best_para = torch.load('models/best/CC_epoch_258_nodropout.pth')
    y_pred_prob1 = predict_test(testloader,model,None,device,best_para )
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 1')
    #2
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_184_1211_9602.pth')
    y_pred_prob1 = predict_test(testloader,model,None,device,best_para )
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 2')
    #3
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_192_1212_9580.pth')
    y_pred_prob1 = predict_test(testloader,model,None,device,best_para )
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 3')
    #4
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_240_1217_9595_pseudolabel.pth')
    y_pred_prob1 = predict_test(testloader,model,None,device,best_para )
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 4')
    #5
    model = wrn(num_classes=10,
                depth=40,
                widen_factor=4,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_142_1217_wrn404.pth')
    y_pred_prob1 = predict_test(testloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 5')
    #6
    model = wrn(num_classes=10,
                depth=40,
                widen_factor=4,
                dropRate=0.3,).to(device)
    best_para = torch.load('models/best/CC_epoch_286_1217_wrn404_pseudolabel.pth')
    y_pred_prob1 = predict_test(testloader,model,None,device,best_para)
    y_pred_probs.append(y_pred_prob1)
    print('predict completed: 6')
    
    xgb_test_probs = torch.cat(y_pred_probs,dim=1).cpu().data.numpy()
    xgb_pred = xgbc1.predict(xgb_test_probs)
    xgb_pred2 = xgb_pred
    y_pred = xgb_pred
    '''
    y_pred_probs_mean = (y_pred_probs[0]+y_pred_probs[1]+y_pred_probs[2]+y_pred_probs[3])/4
    y_pred_probs_mean = y_pred_probs_mean.cpu().data.numpy()
    y_pred = np.argmax(y_pred_probs_mean, axis=1)
    '''
    sub = pd.DataFrame(y_pred, columns=['label'])
    sub.index.name='id'
    sub.to_csv('ensemble_6_xgboost.csv', index=True)

def predict_train(trainloader,model,criterion,device,best_param):
    # custom select model
    model.load_state_dict(best_param)
    model.eval()
    y_pred_prob = []
    for i_batch, (data, target) in enumerate(trainloader):
        images = data.to(device)
        outputs = model(images).detach()
        prob = torch.nn.Softmax(dim=1)(outputs)
        y_pred_prob.append(prob)
    
    y_pred_prob = torch.cat(y_pred_prob)
    #print(y_pred_prob)
    return y_pred_prob

def predict_test(trainloader,model,criterion,device,best_param):
    # custom select model
    model.load_state_dict(best_param)
    model.eval()
    y_pred_prob = []
    for i_batch, data in enumerate(trainloader):
        images = data.to(device)
        outputs = model(images).detach()
        prob = torch.nn.Softmax(dim=1)(outputs)
        y_pred_prob.append(prob)
    
    y_pred_prob = torch.cat(y_pred_prob)
    #print(y_pred_prob)
    return y_pred_prob

if __name__ == '__main__':
    main()