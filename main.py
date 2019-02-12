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
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--comments', '-c', default=True, type=bool, help='print all the statements')
args = parser.parse_args()


''' OPTIMIZER PARAMETERS - Analysis on those '''

best_acc = 0  
start_epoch = 0  
num_epochs = 350  ## TODO: set to 350 
batch_size = 128  ## TODO: set to 128
ensemble_size = 7 ## TODO: set to 7 

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

L = 16
M = 32

convnet = Conv_Net('ConvNet', layers=L, filters=M)
r_convnet = Conv_Recusive_Net('RecursiveConvNet', layers=L, filters=M)
print('Regular ConvNet')
print('Parameters: {}M'.format(count_parameters(convnet)/1e6))
if comments: print(convnet)

print('Recursive ConvNet')
print('Parameters: {}M'.format(count_parameters(r_convnet)/1e6))
if comments: print(r_convnet)

#P1 = (8*8*3*M + 3*3*M**2*L + M*(L+1) + 64*M*10+10) * 1e-6
#P2 = 16 * ((8*8*3*M + 3*3*M**2 + M*2 + 64*M*10+10) * 1e-6)

from collections import OrderedDict

ensemble = OrderedDict()
E = round((count_parameters(convnet)/count_parameters(r_convnet)))
for n in range(1,1+E):
    ensemble['net_{}'.format(n)] = Conv_Recusive_Net('net_{}'.format(n), layers=L, filters=M)

#if args.resume:
#    
#    print('==> Resuming from checkpoint..')
#    print("[IMPORTANT] Don't forget to rename the results object to not overwrite!! ")
#    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#    checkpoint = torch.load('./checkpoint/ckpt.t7')
#    
#    for n,net in enumerate(ensemble):
#        net.load_state_dict(checkpoint['net_{}'.format(n)])
#    
#    best_acc = checkpoint['acc']
#    start_epoch = checkpoint['epoch']

optimizers = []
criterion = nn.CrossEntropyLoss()
for n in range(1,1+E):
    optimizers.append(optim.SGD(ensemble['net_{}'.format(n)].parameters(), lr=args.lr, momentum=0.9))


# Training



