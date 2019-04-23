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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import timeit, avoidWarnings
from beautifultable import BeautifulTable as BT

avoidWarnings()
## Note: the paper doesn't mention about trainining epochs/iterations
parser = argparse.ArgumentParser(description='Recursive Networks with Ensemble Learning')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--layers', '-L', default=16, type=int, help='# of layers')
parser.add_argument('--batch', '-bs', default=128, type=int, help='batch size')
parser.add_argument('--batchnorm', '-bn', default=False, type=bool, help='batch norm')
parser.add_argument('--epochs', '-E', default=700, type=int, help='num epochs')
parser.add_argument('--filters', '-M', default=32, type=int, help='# of filters')
parser.add_argument('--ensemble', '-K', default=5, type=int, help='ensemble size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--comments', '-c', default=True, type=bool, help='print all the statements')
parser.add_argument('--testing', '-t', default=False, type=bool, help='set True if running without GPU for debugging purposes')
args = parser.parse_args()


L = args.layers
M = args.filters
K = args.ensemble
BN = args.batchnorm



# Paths to Results
check_path = './checkpoint/Ensemble_Non_Recursive_L_{}_M_{}_BN_{}_K_{}.t7'.format(L,M,BN,K)
path = '../results/dicts/ensemble_non_recursives/Ensemble_Non_Recursive_L_{}_M_{}_BN_{}_K_{}.pkl'.format(L,M,BN,K)


''' OPTIMIZER PARAMETERS - Analysis on those '''

best_acc = 0  
start_epoch = 0  
num_epochs = args.epochs 
batch_size = args.batch
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
table.append_row(['Architecture', 'Recursive NN (x{})'.format(K)])
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


print('Non Recursive ConvNet')
if comments: print(net)
print('\n\n\t\tParameters: {}M'.format(count_parameters(net)/1e6))


from collections import OrderedDict

ensemble = OrderedDict()
for n in range(1,1+K):
    ensemble['net_{}'.format(n)] = Conv_Net('net_{}'.format(n), L, M)

optimizers = []
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

for n in range(1,1+K):
    optimizers.append(
        optim.SGD(ensemble['net_{}'.format(n)].parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    )

def load_model(net, n, check_path, device):
    # Function to load saved models
    def load_weights(check_path):
        assert os.path.exists(check_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(check_path, map_location=device)
        new_state_dict = OrderedDict()
        
        for k,v in checkpoint['net_{}'.format(n)].state_dict().items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        return new_state_dict 
    
    net.load_state_dict(load_weights(check_path)) # remove word `module`
    
    net.to(device)
    if device == 'cuda': 
        net = torch.nn.DataParallel(net)
    return net


if args.resume:
    
    print('==> Resuming from checkpoint..')
    print("[IMPORTANT] Don't forget to rename the results object to not overwrite!! ")
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(check_path, map_location=device)
    
    for n,net in enumerate(ensemble.values()):
        net = load_model(net, n+1, check_path, device)
        optimizers[n].load_state_dict(checkpoint['opt'][n])  
        
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



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
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        individual_outputs = list()
        
        for n, net in enumerate(ensemble.values()):
            
            if device == 'cuda':
                net.to(device)
        
            net.train()
            net.to(device)
            optimizers[n].zero_grad()
            
            ## Individuals forward pass
            
            n_total = 0
            n_correct = 0
    
            output = net(inputs)
            loss = criterion(output, targets)
            
            loss.backward()
            optimizers[n].step()
        
            # Individual network performance            
            _, predicted = output.max(1)
            n_total += targets.size(0)
            n_correct += predicted.eq(targets).sum().item()
            n_accuracy = 100. * n_correct / n_total
            
            # Store epoch results for this individual (as first iter of the epoch)
            if batch_idx == 0:
            
                results.append_loss(round(loss.item(), 3), 'train', n+1)
                results.append_accy(round(n_accuracy, 2), 'train', n+1)
            
            individual_outputs.append(output)
    
         ## Ensemble forward pass
            
        output = torch.mean(torch.stack(individual_outputs), dim=0)
        loss = criterion(output, targets) 
        
        # Ensemble perofmance
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100 * correct / total
    
    # Store iteration results for Ensemble
    results.append_loss(round(loss.item(), 2), 'train', None)
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
            
            individual_outputs = list()
            for n, net in enumerate(ensemble.values()):
                
                net.eval()  
                net.to(device)

                ## Individual forward pass
                
                n_total = 0
                n_correct = 0
                
                output = net(inputs)
                individual_outputs.append(output)
                loss = criterion(output, targets)
                
                _, predicted = output.max(1)
                n_total += targets.size(0)
                n_correct += predicted.eq(targets).sum().item()
                n_accuracy = n_correct / n_total
                
                # Store epoch (as first iteration of the epoch) results for each net
                if batch_idx == 0:     
                    results.append_loss(round(loss.item(), 2), 'valid', n+1)
                    results.append_accy(round(n_accuracy * 100, 2), 'valid', n+1)
    
                ## Ensemble forward pass
            
            output = torch.mean(torch.stack(individual_outputs), dim=0)
            loss = criterion(output, targets) 
            
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Store epoch results for ensemble
        acc = 100.*correct/total
        results.append_loss(round(loss.item(), 2), 'valid', None)
        results.append_accy(round(acc,2), 'valid', None)
        print('Valid :: Loss: {} | Accy: {}'.format(round(loss.item(),2), round(acc,2)))
    
            
    # Save checkpoint.
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'acc': acc,
            'epoch': epoch,
            'opt': [optimizers[k].state_dict() for k in range(K)]
        }
            
        for k in range(1,K+1):
            netname = 'net_{}'.format(k)
            state[netname] = ensemble[netname]
        if not os.path.isdir('checkpoint/'):
            os.mkdir('checkpoint')
        torch.save(state, check_path)
        best_acc = acc
    return


def lr_schedule(epoch):

    global K
    global milestones
    if epoch in milestones:
        for k in range(K):
            for p in optimizers[k].param_groups:  p['lr'] = p['lr'] / 10
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


names = [n.name for n in ensemble.values()]
results.name = names[0][:-2] + '(x' + str(len(names)) + ')'

if device == 'cuda':
    for net in ensemble.values():
        net = torch.nn.DataParallel(net)

# Start Training
import click
print('Current set up')
print('Testing ', testing)
print('[ALERT]: Path to results (this may overwrite', path)
print('[ALERT]: Path to checkpoint (this may overwrite', check_path)
if click.confirm('Do you want to continue?', default=True):

    print('[OK]: Starting Training of Recursive Ensemble Model')
    for epoch in range(start_epoch, num_epochs):
        run_epoch(epoch)

else:
    print('Exiting...')
    exit()
    
results.show()


exit()



import pickle

# Analysizing Results
# --------------------

# Round 2: L = 32, M = 64 --> Using scripts to build table and then chose examples
# -----------------------
single_prmts = {'L': 32, 'M': 64, 'BN': False} 
#ensemble_prmts = {'L': 16, 'M': 31, 'BN': False, 'K': 4}  #  - done
#ensemble_prmts = {'L': 4,  'M': 36, 'BN': False, 'K': 16}  # - done
#ensemble_prmts = {'L': 4, 'M': 54, 'BN': False, 'K': 8}   #  - done 
#ensemble_prmts = {'L': 8, 'M': 40, 'BN': False, 'K': 8}   #  - done


# Round 2: L = 32, M = 64
# -----------------------

# Horizontal Division
ensemble_prmts = {'L': 32,  'M': 31, 'BN': False, 'K': 4}   # - done
ensemble_prmts = {'L': 32,  'M': 21, 'BN': False, 'K': 8}   # - done
ensemble_prmts = {'L': 32,  'M': 14, 'BN': False, 'K': 16}  # - done

# Vertical Division
ensemble_prmts = {'L': 2,  'M': 64, 'BN': False, 'K': 4}    # - done
ensemble_prmts = {'L': 6,  'M': 64, 'BN': False, 'K': 8}    # - done

# Round 3.1: L = 32, M = 64 -> Soft Constraint  (Me = 48)
# --------------------------------------------------------
# Vertical Division
ensemble_prmts = {'L': 12,  'M': 48, 'BN': False, 'K': 4}       # - TODO
ensemble_prmts = {'L': 5,   'M': 48, 'BN': False, 'K': 8}       # - TODO
ensemble_prmts = {'L': 3,   'M': 48, 'BN': False, 'K': 12}      # - TODO
ensemble_prmts = {'L': 1,   'M': 48, 'BN': False, 'K': 16}      # - TODO & RECURSIVE


# Round 3.1: L = 32, M = 64 -> Soft Constraint  (Me = 32)
# --------------------------------------------------------
# Vertical Division
ensemble_prmts = {'L': 30,  'M': 32, 'BN': False, 'K': 4}       # - TODO
ensemble_prmts = {'L': 13,  'M': 32, 'BN': False, 'K': 8}       # - TODO
ensemble_prmts = {'L': 8,   'M': 32, 'BN': False, 'K': 12}      # - TODO
ensemble_prmts = {'L': 5,   'M': 32, 'BN': False, 'K': 16}      # - TODO


path = '../results/dicts/ensemble_non_recursives/Ensemble_Non_Recursive_L_{L}_M_{M}_BN_{BN}_K_{K}.pkl'.format(**ensemble_prmts)
path_ = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_{L}_M_{M}_BN_{BN}.pkl'.format(**single_prmts)
with open(path, 'rb') as input: results = pickle.load(input)
with open(path_, 'rb') as input: results_ = pickle.load(input)

num_epochs = 700
colors = c = ['pink', 'blue', 'green', 'yellow', 'purple']
L,M,BN,K = list(ensemble_prmts.values())








## Calculate Results
# --------------------
# --------------------

from analysis_ensembles import accuracy_metrics, time_metrics
times = time_metrics(L,M,BN,K,is_recursive=False)
acc = accuracy_metrics(L,M,BN,K,is_recursive=False)


## Plot Results
# ---------------
# ---------------

from analysis_ensembles import plot_loss_ensembles_vs_single  ## print_inidividuals, print_single
from analysis_ensembles import plot_accuracy_ensembles_vs_single ## print_inidividuals, print_single
from analysis_ensembles import plot_classwise_accuracy

plot_loss_ensembles_vs_single(L,M,BN,K, results, print_individuals=True)
plot_accuracy_ensembles_vs_single(L,M,BN,K, results, print_individuals=True)
plot_accuracy_ensembles_vs_single(L,M,BN,K, results, print_individuals=False, results_=results_)
plot_classwise_accuracy(L,M,BN,K,recursive=False, results=acc)



recursive = False
L = [16, 4, 4, 8]
L = [32, 4, 4, 8]
M = [31, 36, 54, 40]
K = [4, 16, 8, 8]
BN = [False] * len(L)
results_.name = 'L = {L} M = {M}'.format(**single_prmts)

from analysis_ensembles import plot_compare_ensembles_accuracy
plot_compare_ensembles_accuracy(L,M,BN,K, results=None, results_=results_)

from analysis_ensembles import plot_compare_ensembles_loss
plot_compare_ensembles_loss(L,M,BN,K, results=None, results_=results_)


