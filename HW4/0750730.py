# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:23:26 2018

@author: james
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import math


# Random erasing
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by
    Zhong et al.
    ---------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    ---------------------------------------------------------------------------
    '''
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


# Define dataset class
class ImageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        df = pd.read_csv(file_path)
        if 'label' in df.columns:
            self.is_train = True
        else:
            self.is_train = False

        if self.is_train:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)) \
                       .astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)
        else:
            # test data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)) \
                       .astype(np.uint8)[:, :, :, None]
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
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
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
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
                 dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate))
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
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
                                   dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                   dropRate)
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


def adjust_learning_rate(optimizer, epoch):
    schedule = [45, 90, 135, 180]
    gamma = 0.2
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
            print('lr decayed by gamma = %.2f\tlr = %f' % (gamma,
                                                           param_group['lr']))


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
n_epoches = 200

# Record
losses = np.zeros((n_epoches))
valid_losses = np.zeros((int(n_epoches/2)))
valid_accuracy = np.zeros((int(n_epoches/2), 2))


def main():
    global best_acc, best_epoch, losses, valid_losses, valid_accuracy
    global n_epoches

    # Dataset transforms
    randomErasing = RandomErasing()
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                randomErasing])
    transform2 = transforms.Compose([
                 transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])

    # Load dataset
    print('Start loading data')
    train_dataset = ImageDataset('data/train.csv', transform)
    validation_dataset = ImageDataset('data/train.csv', transform2)
    test_dataset = ImageDataset('data/test.csv', transform2)

    # Split train and validation
    valid_ratio = 0.1
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio*num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Load dataloader
    trainloader = DataLoader(train_dataset, batch_size=200,
                             sampler=train_sampler, num_workers=2,
                             pin_memory=True)
    validationloader = DataLoader(validation_dataset, batch_size=100,
                                  sampler=valid_sampler, num_workers=2,
                                  pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=100,
                            num_workers=2, shuffle=False, pin_memory=True)

    # Build model
    model = wrn(num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,).to(device)

    # Setup optimizer and criterion
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss().to(device)
    # no use
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20,
    #                                        gamma=0.1)

    # Resume training (from epoch=49)
    # optimizer.load_state_dict(torch.load('models/optimizer_epoch_249.pth'))
    # model.load_state_dict(torch.load('models/CC_epoch_249.pth'))

    print('Start training')
    start_time = time.time()
    for epoch in range(0, n_epoches):
        # Train phase
        train_loss = train(trainloader,
                           model,
                           criterion,
                           optimizer,
                           epoch,
                           device)

        losses[epoch] = train_loss

        # Valid phase
        if(epoch % 2 == 0 and epoch != 0) or epoch == (n_epoches-1):
            v_loss, v_acc = valid(validationloader,
                                  model,
                                  criterion,
                                  epoch,
                                  device,
                                  valid_ratio)

            valid_losses[int(epoch/2)] = v_loss
            valid_accuracy[int(epoch/2), 1] = epoch
            valid_accuracy[int(epoch/2), 0] = v_acc

            if valid_accuracy[int(epoch/2), 0] > best_acc:
                best_acc = valid_accuracy[int(epoch/2), 0]
                best_epoch = epoch
                best_param = model.state_dict()

        end_epoch_time = time.time()
        print('Epoch %d : %d secs \tTrain loss : %.4f'
              % (epoch, (end_epoch_time-start_time), np.mean(losses[epoch])))

    print('Best valid acc at epoch %d: %.2f' % (best_epoch, best_acc))

    print('Start testing')
    # Testing phase
    test(testloader, model, criterion, device, best_param, best_epoch)

    # Save the record
    df = pd.DataFrame(losses)
    df.to_csv("train_losses.csv", header=None)
    df = pd.DataFrame(valid_losses)
    df.to_csv("v_loss.csv", header=None)
    df = pd.DataFrame(valid_accuracy)
    df.to_csv("v_acc.csv", header=None)


def train(trainloader, model, criterion, optimizer, epoch, device):
    start_time = time.time()
    model.train()
    # LR decay
    adjust_learning_rate(optimizer, epoch)
    losses = np.zeros((len(trainloader)))
    for i_batch, (data, target) in enumerate(trainloader):
        # Load data
        images = data.to(device)
        labels = target.to(device)
        # Output
        ouputs = model(images)
        loss = criterion(ouputs, labels)
        # Update param
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record loss
        losses[i_batch] = loss.item()
        end_batch_time = time.time()
        if i_batch % 200 == 0 and i_batch != 0:
            print('Epoch %d : %d/%d %d secs\tLoss:%.6f'
                  % (epoch,
                     i_batch+1,
                     len(trainloader),
                     (end_batch_time-start_time),
                     np.mean(losses)))

    end_epoch_time = time.time()
    print('Epoch %d : %d/%d %d secs\tLoss:%.6f'
          % (epoch,
             i_batch+1,
             len(trainloader),
             (end_epoch_time-start_time),
             np.mean(losses)))

    torch.save(model.state_dict(),
               '%s/CC_epoch_%d.pth' % (modeldir, epoch))
    # torch.save(optimizer.state_dict(),
    #           '%s/optimizer_epoch_%d.pth'%(modeldir,epoch))
    # Return loss
    return np.mean(losses)


def valid(validationloader, model, criterion, epoch, device, valid_ratio):
    # Evaluation parameters
    num_correct = 0
    model.eval()
    losses = np.zeros((len(validationloader)))
    # Evaluation phase
    for i_batch, (data, target) in enumerate(validationloader):
        images = torch.tensor(data, requires_grad=False).to(device)
        labels = torch.tensor(target, requires_grad=False).to(device)

        ouputs = model(images)
        y_pred = ouputs.data.max(1)[1]
        loss = criterion(ouputs, labels)
        losses[i_batch] = loss.item()
        num_correct += y_pred.eq(labels).long().sum().item()

    accuracy = num_correct / (len(validationloader.dataset)*valid_ratio) * 100

    print('Epoch %d: Val Avg loss: %.6f, Acc: %d/%d \t(%.2f%%)'
          % (epoch,
             np.mean(losses),
             num_correct,
             int(len(validationloader.dataset)*valid_ratio),
             accuracy))

    return np.mean(losses), accuracy


def test(testloader, model, criterion, device, best_param, best_epoch):
    # custom select model
    model.load_state_dict(best_param)
    model.eval()
    y_pred = np.empty(len(testloader.dataset))
    bs = testloader.batch_size
    for i_batch, data in enumerate(testloader):
        images = data.to(device)
        ouputs = model(images)
        pred = ouputs.data.max(1)[1]
        y_pred[i_batch*bs:i_batch*bs+bs] = pred.cpu().numpy()

    y_pred = y_pred.astype(int)
    sub = pd.DataFrame(y_pred, columns=['label'])
    sub.index.name = 'id'
    sub.to_csv('answer_wrn_28_10_%d.csv' % best_epoch, index=True)


if __name__ == '__main__':
    main()
