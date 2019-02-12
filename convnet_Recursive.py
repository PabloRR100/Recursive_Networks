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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from beautifultable import BeautifulTable as BT


parser = argparse.ArgumentParser(description='Recursive Networks with Ensemble Learning')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--layers', '-L', default=16, type=int, help='# of layers')
parser.add_argument('--batch', '-bs', default=128, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='num epochs')
parser.add_argument('--filters', '-M', default=32, type=int, help='# of filters')
parser.add_argument('--ensemble', '-es', default=5, type=int, help='ensemble size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--comments', '-c', default=True, type=bool, help='print all the statements')
args = parser.parse_args()


''' OPTIMIZER PARAMETERS - Analysis on those '''

best_acc = 0  
start_epoch = 0  
num_epochs = 4  ## TODO: set to args.epochs
batch_size = 128  ## TODO: set to args.barch
milestones = [100, 150]

L = args.layers
M = args.filters
E = args.ensemble
 
n_workers = torch.multiprocessing.cpu_count()
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
from data import dataloaders
trainloader, testloader, classes = dataloaders(dataset, batch_size)


# Models 
# ------
    
# For now, NO SHARING of any layers withing the ensemble
    
comments = False
from models import Conv_Recusive_Net

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


