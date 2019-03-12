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
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--layers', '-L', default=16, type=int, help='# of layers')
parser.add_argument('--batch', '-bs', default=128, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='num epochs')
parser.add_argument('--filters', '-M', default=32, type=int, help='# of filters')
parser.add_argument('--ensemble', '-es', default=5, type=int, help='ensemble size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--comments', '-c', default=True, type=bool, help='print all the statements')
parser.add_argument('--testing', '-t', default=False, type=bool, help='set True if running without GPU for debugging purposes')
args = parser.parse_args()


# Paths to Results
check_path = './checkpoint/ckpt_rec_ens.t7'
path = '../results/ensemble_recursives/Results_Ensemble_Recursive.pkl'



''' OPTIMIZER PARAMETERS - Analysis on those '''

best_acc = 0  
start_epoch = 0  
num_epochs = 500  ## TODO: set to args.epochs
batch_size = 128  ## TODO: set to args.barch
milestones = [150, 300, 400]

L = args.layers
M = args.filters
E = args.ensemble

num_epochs = 3
testing = True
#testing = args.testing 
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
table.append_row(['Architecture', 'DenseNet x{}'.format(E)])
table.append_row(['Dataset', 'CIFAR10'])
table.append_row(['Epochs', str(num_epochs)])
table.append_row(['Batch Size', str(batch_size)])
table.append_row(['Testing', str(testing)])

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
    
avoidWarnings()
comments = True
from models import Conv_Recusive_Net
from utils import count_parameters
net = Conv_Recusive_Net('recursive_net', L, M)


print('Recursive ConvNet')
if comments: print(net)
print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))


from collections import OrderedDict

ensemble = OrderedDict()
for n in range(1,1+E):
    ensemble['net_{}'.format(n)] = Conv_Recusive_Net('net_{}'.format(n), L, M)

if args.resume:
    
    print('==> Resuming from checkpoint..')
    print("[IMPORTANT] Don't forget to rename the results object to not overwrite!! ")
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_Ens_Rec.t7')
    
    for n,net in enumerate(ensemble):
        net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['net_{}'.format(n)])
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

optimizers = []
criterion = nn.CrossEntropyLoss()

for n in range(1,1+E):
    optimizers.append(
        optim.SGD(ensemble['net_{}'.format(n)].parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    )


# Training
# --------
    
# Helpers
from results import TrainResults as Results


def train(epoch):
    
    global device
    global results
    global optimizers    
    print('\nEpoch: %d' % epoch)

    total = 0
    correct = 0
    len_ = len(trainloader)
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        individual_outputs = list()
        
        for n, net in enumerate(ensemble.values()):
            
            if device == 'cuda':
                net.to(device)
                net = torch.nn.DataParallel(net)
        
            net.train()
            optimizers[n].zero_grad()
            
            ## Individuals forward pass
            
            n_total = 0
            n_correct = 0
    
            inputs, targets = inputs.to(device), targets.to(device)
            output = net(inputs)
            loss = criterion(output, targets)
            
            loss.backward()
            optimizers[n].step()
        
            # Individual network performance            
            _, predicted = output.max(1)
            n_total += targets.size(0)
            n_correct += predicted.eq(targets).sum().item()
            n_accuracy = 100. * n_correct / n_total
            
            # Store iteration results for this individual
            results.append_iter_loss(round(loss.item(), 3), 'train', n+1)
            results.append_iter_accy(round(n_accuracy, 2), 'train', n+1)
            
#            if batch_idx == len_-1:
            if batch_idx == 0:
                # Store epoch results for this individual (as last iter) (== 0 as first iter of the epoch)
                results.append_loss(round(loss.item(), 3), 'train', n+1)
                results.append_accy(round(n_accuracy, 2), 'train', n+1)
            
            individual_outputs.append(output)
        
        ## TODO: Just set testing = True when debuggin on local
        if testing and batch_idx == 5:
            break
    
     ## Ensemble forward pass
        
    output = torch.mean(torch.stack(individual_outputs), dim=0)
    loss = criterion(output, targets) 
    
    _, predicted = output.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    accuracy = 100 * correct / total
    
    # Store iteration results for Ensemble
    results.append_loss(round(loss.item(), 3), 'train', None)
    results.append_accy(round(accuracy, 2), 'train', None)

    print('Train :: Loss: {} | Accy: {}'.format(round(loss.item(),2), round(accuracy,2)))

        
def test(epoch):

    total = 0
    correct = 0
    global results
    global best_acc

    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outs = []
            for n, net in enumerate(ensemble.values()):
                
                net.eval()  
                net.to(device)

                ## Individual forward pass
                
                n_total = 0
                n_correct = 0
                
                output = net(inputs)
                outs.append(output)
                loss = criterion(output, targets)
    
                # Store epoch (as first iteration of the epoch) results for each net
                if batch_idx == 0:
    
                    _, predicted = output.max(1)
                    n_total += targets.size(0)
                    n_correct += predicted.eq(targets).sum().item()
                    n_accuracy = n_correct / n_total
                    
                    results.append_loss(round(loss.item(), 3), 'valid', n+1)
                    results.append_accy(round(n_accuracy * 100, 2), 'valid', n+1)
    
            
            # TODO: UNCOMMENT WHEN RUNNING ON SERVER -> wraped in test parameter
            if testing and batch_idx == 5:
                break
            
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Store epoch results for ensemble
        acc = 100.*correct/total
        results.append_loss(round(loss.item(), 3), 'valid', None)
        results.append_accy(round(acc,2), 'valid', None)
        print('Valid :: Loss: {} | Accy: {}'.format(round(loss.item(),2), round(acc,2)))
    
            
    # Save checkpoint.
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ens_rec_ckpt.t7')
        best_acc = acc
    return


def lr_schedule(epoch):

    global E
    global milestones
    if epoch in milestones:
        for n in range(E):
            for p in optimizers[n].param_groups:  p['lr'] = p['lr'] / 10
        print('\n** Changing LR to {} \n'.format(p['lr']))    
    return

def results_backup():
    global results
    with open(path, 'wb') as object_result:
        pickle.dump(results, object_result, pickle.HIGHEST_PROTOCOL)   
    return


@timeit
def run_epoch(epoch):
    
    lr_schedule(epoch)
    train(epoch)
    test(epoch)
    results_backup()
        
    
results = Results(list(ensemble))
results.append_time(0)


#testing = True
names = [n.name for n in ensemble.values()]
results.name = names[0][:-2] + '(x' + str(len(names)) + ')'

# Start Training
import click
print('Current set up')
print('Testing ', testing)
print('[ALERT]: Path to results (this may overwrite', path)
if click.confirm('Do you want to continue?', default=True):

    print('[OK]: Starting Training of Recursive Ensemble Model')
    for epoch in range(start_epoch, num_epochs):
        run_epoch(epoch)

else:
    print('Exiting...')
    exit()
    
results.show()


exit()


# Analysizing Results
# --------------------

import matplotlib.pyplot as plt
plot_single_model = psm = True
## Ensemble vs the Individuals is the default option
## set plot_single_model to True to include the results from train_non_Recursive.py


#path = '../results/ensemble_recursives/Results_Ensemble_Recursive.pkl'
path = '../results/ensemble_recursives/definitives/Results_Ensemble_Recursive.pkl'
path_ = '../results/single_recursive/definitives/Results_Single_Recursive.pkl'
with open(path, 'rb') as input: results = pickle.load(input)
with open(path_, 'rb') as input: results_ = pickle.load(input)


c = [0, 'red', 'blue', 'green', 'yellow', 'purple']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for m in range(1,1+E):
    ax1.plot(range(num_epochs), results.train_loss['m{}'.format(m)], label='Train m{}'.format(m), color=c[m], alpha=0.4)
ax1.plot(range(num_epochs), results.train_loss['ensemble'], label='Train Ensemble', color='black', alpha=1)
if psm: ax1.plot(range(num_epochs), results_.train_loss, label='Single Model', color='red', alpha=1, linewidth=0.5)
ax1.set_title('Trianing Loss')
ax1.grid(True)

for m in range(1,1+E):
    ax2.plot(range(num_epochs), results.valid_loss['m{}'.format(m)], label='Valid ,{}'.format(m), color=c[m], alpha=0.4)
ax2.plot(range(num_epochs), results.valid_loss['ensemble'], label='Valid Ensemble', color='black', alpha=1)
ax2.set_title('Validation Loss')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
if psm: ax2.plot(range(num_epochs), results_.valid_loss, label='Single Model', color='red', alpha=1, linewidth=0.5)
ax2.grid(True)

for m in range(1,1+E):
    ax3.plot(range(num_epochs), results.train_accy['m{}'.format(m)], label='Train m{}'.format(m), color=c[m], alpha=0.4)
ax3.plot(range(num_epochs), results.train_accy['ensemble'], label='Train Ensemble', color='black', alpha=1)
if psm: ax3.plot(range(num_epochs), results_.train_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
ax3.set_title('Training Accuracy')
ax3.grid(True)

for m in range(1,1+E):
    ax4.plot(range(num_epochs), results.valid_accy['m{}'.format(m)], label='Valid m{}'.format(m), color=c[m], alpha=0.4)
ax4.plot(range(num_epochs), results.valid_accy['ensemble'], label='Valid Ensemble', color='black', alpha=1)
if psm: ax4.plot(range(num_epochs), results_.valid_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
ax4.set_title('Validation Accuracy')
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax4.grid(True)
plt.show()


## Training and validation together
#fig, (ax1, ax2) = plt.subplots(2, 1)
#for m in range(1,1+E):
#    ax1.plot(range(num_epochs), results.train_loss['m{}'.format(m)], label='Train m{}'.format(m), color=c[m], alpha=0.4)
#    ax1.plot(range(num_epochs), results.valid_loss['m{}'.format(m)], label='Valid ,{}'.format(m), color=c[m], alpha=0.4)
#ax1.plot(range(num_epochs), results.train_loss['ensemble'], label='Train Ensemble', color='black', alpha=1)
#ax1.plot(range(num_epochs), results.valid_loss['ensemble'], label='Valid Ensemble', color='black', alpha=1)
#ax1.set_title('Loss')
#ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax1.grid(True)
#
#for m in range(1,1+E):
#    ax2.plot(range(num_epochs), results.train_accy['m{}'.format(m)], label='Train m{}'.format(m), color=c[m], alpha=0.4)
#    ax2.plot(range(num_epochs), results.valid_accy['m{}'.format(m)], label='Valid m{}'.format(m), color=c[m], alpha=0.4)
#ax2.plot(range(num_epochs), results.train_accy['ensemble'], label='Train Ensemble', color='black', alpha=1)
#ax2.plot(range(num_epochs), results.valid_accy['ensemble'], label='Valid Ensemble', color='black', alpha=1)
#ax2.set_title('Accuracy')
#ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax2.grid(True)
#
#fig.tight_layout()
#plt.show()
#

