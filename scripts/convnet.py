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

from utils import avoidWarnings
from beautifultable import BeautifulTable as BT

avoidWarnings()
parser = argparse.ArgumentParser(description='Recursive Networks with Ensemble Learning')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
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
num_epochs = 10  ## TODO: set to args.epochs
batch_size = 128  ## TODO: set to args.barch
milestones = [100, 150]

L = args.layers
M = args.filters
E = args.ensemble
 
comments = args.comments
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

avoidWarnings()
dataset = 'MNIST'
dataset = 'CIFAR'
from data import dataloaders
trainloader, testloader, classes = dataloaders(dataset, batch_size)


# Models 
# ------
    
# For now, NO SHARING of any layers withing the ensemble

avoidWarnings()
comments = True
from models import Conv_Net
from utils import count_parameters
net = Conv_Net('net', layers=L, filters=M)

print('Regular net')
if comments: print(net)
print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))

if args.resume:
    
    print('==> Resuming from checkpoint..')
    print("[IMPORTANT] Don't forget to rename the results object to not overwrite!! ")
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    
    net.load_state_dict(checkpoint['net'])
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)


# Training
# --------
    
#from utils import timeit
from results import TrainResults as Results
## Note: the paper doesn't mention about trainining iterations

def train(epoch):
    
    net.train()
    print('\nEpoch: %d' % epoch)

    total = 0
    correct = 0
    global results
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
    
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
#        if batch_idx == 20:
#            break
    
    accuracy = 100.*correct/total    
    results.append_loss(loss.item(), 'train')
    results.append_accy(accuracy, 'train')    
    print('Train :: Loss: {} | Accy: {}'.format(round(loss.item(),2), round(accuracy,2)))

        
def test(epoch):
    
    net.eval()

    total = 0
    correct = 0
    global results
    global best_acc

    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
#            if batch_idx == 20:
#                break
            
    # Save checkpoint.
    acc = 100.*correct/total
    results.append_loss(loss.item(), 'valid')
    results.append_accy(acc, 'valid')
    print('Valid :: Loss: {} | Accy: {}'.format(round(loss.item(),2), round(acc,2)))
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


#def lr_schedule(epoch):
#
#    global milestones
#    if epoch == milestones[0] or epoch == milestones[1]:
#        for p in optimizer.param_groups:  p['lr'] = p['lr'] / 10
#        print('\n** Changing LR to {} \n'.format(p['lr']))    
#    return
#    
#def results_backup(epoch):
#    # Save every X epochs in case training breaks we don't loose results    
#    global results
#    if epoch % 20 == 0:
#        with open('Results_Singe.pkl', 'wb') as object_result:
#                pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL)     
#
#
#@timeit
#def run_epoch(epoch):
#    
#    lr_schedule(epoch)
#    train(epoch)
#    test(epoch)
#    results_backup(epoch)
    
    
    
import time
start = time.time()

results = Results([net])
net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('[OK]: Starting Training of Single Model')
for epoch in range(start_epoch, num_epochs):
    
    # LR Scheduler
    if epoch == milestones[0] or epoch == milestones[1]:
        for p in optimizer.param_groups:  p['lr'] = p['lr'] / 10
        print('\n** Changing LR to {} \n'.format(p['lr'])) 
        
    # Run epoch
    train(epoch)
    test(epoch)
    
    # Timer
    print('Time: {}'.format(round((time.time() - start),2)))
    start = time.time()
    
    # Results backup
    if epoch % 20 == 0:
        with open('Results_Singe.pkl', 'wb') as object_result:
                pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL) 

    
results.show()
exit()


import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(num_epochs), results.train_loss, label='Train')
plt.plot(range(num_epochs), results.valid_loss, label='Valid')
plt.title('Loss')
plt.show()

plt.figure()
plt.plot(range(num_epochs), results.train_accy, label='Train')
plt.plot(range(num_epochs), results.valid_accy, label='Valid')

plt.title('Accuracy')
plt.show()


with open('Results_Singe.pkl', 'rb') as input:
    res = pickle.load(input)
