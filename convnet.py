#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

import os
import sys
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from beautifultable import BeautifulTable as BT

parser = argparse.ArgumentParser(description='Recursive Networks with Ensemble Learning')
parser.add_argument('--filters', '-M', default=32, type=int, help='# of filters')
parser.add_argument('--layers', '-L', default=16, type=int, help='# of layers')
parser.add_argument('--batch', '-bs', default=128, type=int, help='batch size')
parser.add_argument('--ensemble', '-es', default=5, type=int, help='ensemble size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--comments', '-c', default=True, type=bool, help='print all the statements')

args = parser.parse_args()


''' OPTIMIZER PARAMETERS - Analysis on those '''

best_acc = 0  
start_epoch = 0  
num_epochs = 4  ## TODO: set to 200
batch_size = 128  ## TODO: set to 128
milestones = [100, 150]
 
n_workers =torch.multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpus = True if torch.cuda.device_count() > 1 else False
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPUs'
    
table = BT()
table.append_row(['Python Version', sys.version[:5]])
table.append_row(['PyTorch Version', torch.__version__])
table.append_row(['Device', str(device_name)])
table.append_row(['Cores', str(n_workers)])
table.append_row(['GPUs', str(torch.cuda.device_count())])
table.append_row(['CUDNN Enabled', str(torch.backends.cudnn.enabled)])
table.append_row(['Architecture', 'DenseNet x7'])
table.append_row(['Dataset', 'CIFAR10'])
table.append_row(['Epochs', str(num_epochs)])
table.append_row(['Batch Size', str(batch_size)])

print(table)


# Data
# ----

dataset = 'MNIST'
dataset = 'CIFAR'

if dataset == 'CIFAR':
    
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Models 
# ------
    
# For now, NO SHARING of any layers withing the ensemble
    
comments = False
from utils import count_parameters
from models import Conv_Net, Conv_Recusive_Net

E = 5
L = 16
M = 32

print('Recursive ConvNet')
print(r_convnet)
print('\n\n\t\tParameters: {}M'.format(count_parameters(r_convnet)/1e6))

from collections import OrderedDict

ensemble = OrderedDict()
for n in range(1,1+E):
    ensemble['net_{}'.format(n)] = Conv_Recusive_Net('net_{}'.format(n), layers=L, filters=M)

if args.resume:
    
    print('==> Resuming from checkpoint..')
    print("[IMPORTANT] Don't forget to rename the results object to not overwrite!! ")
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    
    for n,net in enumerate(ensemble):
        net.load_state_dict(checkpoint['net_{}'.format(n)])
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

optimizers = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(convnet.parameters(), lr=args.lr, momentum=0.9)

for n in range(1,1+E):
    optimizers.append(optim.SGD(ensemble['net_{}'.format(n)].parameters(), lr=args.lr, momentum=0.9))


# Training
# --------
    
from utils import timeit
from results import TrainResults as Results
## Note: the paper doesn't mention about trainining iterations


def train(epoch):
    
    print('\nEpoch: %d' % epoch)
    convnet.train()

    total = 0
    correct = 0
    train_loss = 0
    global results
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = convnet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.*correct/total        
    print('Train :: Epoch {} - Loss: {} | Accy: {}'.format(epoch, train_loss, accuracy))

        
def test(epoch):
    
    convnet.eval()

    total = 0
    correct = 0
    test_loss = 0
    global results
    global best_acc

    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = convnet(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.*correct/total
        print('Valid :: Epoch {} - Loss: {} | Accy: {}'.format(epoch, test_loss, accuracy))
            
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': convnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


def lr_schedule(epoch):

    global optimizer
    global milestones
    if epoch == milestones[0] or epoch == milestones[1]:
        for p in optimizer.param_groups:  p['lr'] = p['lr'] / 10
        print('\n** Changing LR to {} \n'.format(p['lr']))    
    return
    
def results_backup(epoch):
    # Save every X epochs in case training breaks we don't loose results    
    global results
    if epoch % 20 == 0:
        with open('Results_Singe.pkl', 'wb') as object_result:
                pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL)     


@timeit
def run_epoch(epoch):
    lr_schedule(epoch)
    train(epoch)
    test(epoch)
    results_backup(epoch)
    
    
results = Results([convnet])
convnet.to(device)
if device == 'cuda':
    convnet = torch.nn.DataParallel(convnet)
    cudnn.benchmark = True

print('[OK]: Starting Training of Single Model')
for epoch in range(start_epoch, num_epochs):
    run_epoch(epoch)

    
results.show()
exit()


## TODO: train_ensemble, test_ensemble