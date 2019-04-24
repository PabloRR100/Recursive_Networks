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

from utils import timeit, avoidWarnings
from beautifultable import BeautifulTable as BT

avoidWarnings()
## Note: the paper doesn't mention about trainining epochs/iterations
parser = argparse.ArgumentParser(description='Recursive Networks with Ensemble Learning')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate') #changed the learning rate to 0.001 as the paper uses. 
parser.add_argument('--layers', '-L', default=16, type=int, help='# of layers')
parser.add_argument('--batch', '-bs', default=128, type=int, help='batch size')
parser.add_argument('--batchnorm', '-bn', default=False, type=bool, help='batch norm')
parser.add_argument('--epochs', '-e', default=200, type=int, help='num epochs')
parser.add_argument('--filters', '-M', default=32, type=int, help='# of filters')
parser.add_argument('--ensemble', '-es', default=5, type=int, help='ensemble size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--comments', '-c', default=True, type=bool, help='print all the statements')
parser.add_argument('--testing', '-t', default=False, type=bool, help='set True if running without GPU for debugging purposes')
args = parser.parse_args()


L = args.layers
M = args.filters
E = args.ensemble
BN = args.batchnorm

# Paths to Results
check_path = './checkpoint/Single_Non_Recursive_L_{}_M_{}_BN_{}.t7'.format(L,M,BN)
path = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_{}_M_{}_BN_{}.pkl'.format(L,M,BN)



''' OPTIMIZER PARAMETERS - Analysis on those '''

best_acc = 0  
start_epoch = 0  
num_epochs = 700  ## TODO: set to args.epochs
batch_size = 128  ## TODO: set to args.batch
milestones = [550]

testing = args.testing
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
table.append_row(['Architecture', 'Recursive NN'])
table.append_row(['Dataset', 'CIFAR10'])
table.append_row(['Testing', str(testing)])
table.append_row(['Epochs', str(num_epochs)])
table.append_row(['Batch Size', str(batch_size)])
table.append_row(['Learning Rate', str(args.lr)])
table.append_row(['LR Milestones', str(milestones)])
table.append_row(['Layers', str(L)])
table.append_row(['Filters', str(M)])
table.append_row(['BatchNorm', str(BN)])
print(table)



# Data
# ----

avoidWarnings()
dataset = 'CIFAR'
from data import dataloaders
trainloader, testloader, classes = dataloaders(dataset, batch_size)



# Models 
# ------
    
avoidWarnings()
comments = True
from models import Conv_Net
from utils import count_parameters
net = Conv_Net('net', L, M, normalize=BN)


print('Regular net')
if comments: print(net)
print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

if args.resume:
    
    print('==> Resuming from checkpoint..')
    print("[IMPORTANT] Don't forget to rename the results object to not overwrite!! ")
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(check_path)
    
    net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['opt'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Training
# --------
    
# Helpers
from results import TrainResults as Results


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
    
    accuracy = 100.*correct/total    
    results.append_loss(round(loss.item(),2), 'train')
    results.append_accy(round(accuracy,2), 'train')    
    print('Train :: Loss: {} | Accu: {}'.format(round(loss.item(),2), round(accuracy,2)))

        
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
            
    # Save checkpoint.
    acc = 100.*correct/total
    results.append_loss(round(loss.item(),2), 'valid')
    results.append_accy(round(acc,2), 'valid')
    print('Valid :: Loss: {} | Accy: {}'.format(round(loss.item(),2), round(acc,2)))
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'opt': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, check_path)
        best_acc = acc


def lr_schedule(epoch):

    global milestones
    if epoch in milestones:
        for p in optimizer.param_groups:  p['lr'] = p['lr'] / 10
        print('\n** Changing LR to {} \n'.format(p['lr']))    
    return
    

def results_backup():
    global results
    with open(path, 'wb') as object_result:
        pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL)     


@timeit
def run_epoch(epoch):
    
    lr_schedule(epoch)
    train(epoch)
    test(epoch)
    results_backup()
    

# Send model to GPU(s)
results = Results([net])
results.name = net.name
net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



# Start training
import click
print('Current set up')
print('Testing ', testing)
print('[ALERT]: Path to results (this may overwrite', path)
print('[ALERT]: Path to checkpoint (this may overwrite', check_path)
if click.confirm('Do you want to continue?', default=True):

    print('[OK]: Starting Training of Single Model')
    for epoch in range(start_epoch, num_epochs):
        run_epoch(epoch)

else:
    print('Exiting...')
    exit()
    
results.show()




exit()





## TODO: Simplify using analysis module

L = [16,32,16,32,16,32]
M = [16,16,32,32,64,64]

BN = [False] * len(L)
num_epochs = 700
recursive = False
ensemble = False


## Calculate Results
# --------------------
# --------------------

from analysis import accuracy_metrics, time_metrics
acc = accuracy_metrics(L,M,BN,recursive, ensemble)
times = time_metrics(L,M,BN,recursive,ensemble)



## Plot Results
# ---------------
# ---------------

from analysis import plot_loss, plot_accuracy, \
    plot_classwise_accuracy, plot_inference_time, plot_top1_top5_accuracy

plot_loss(L,M,BN,recursive,ensemble)
plot_accuracy(L,M,BN,recursive,ensemble)
plot_inference_time(L,M,BN,recursive,ensemble,results=times) 
plot_classwise_accuracy(L,M,BN,recursive,ensemble,results=acc)
plot_top1_top5_accuracy(L,M,BN,recursive,ensemble,results=acc)










