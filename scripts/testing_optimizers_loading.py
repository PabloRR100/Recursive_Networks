#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:23:09 2019

@author: pabloruizruiz
"""

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

L = 3
M = 5
K = 3
BN = False

# Paths to Results
check_path = 'test_L_{}_M_{}_BN_{}_K_{}.t7'.format(L,M,BN,K)
path = 'test_L_{}_M_{}_BN_{}_K_{}.pkl'.format(L,M,BN,K)

''' OPTIMIZER PARAMETERS - Analysis on those '''

best_acc = 0  
start_epoch = 0  
num_epochs = 5  ## TODO: set to args.epochs
batch_size = 128  ## TODO: set to args.barch
milestones = [550]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = 'CIFAR'
from data import dataloaders
trainloader, testloader, classes = dataloaders(dataset, batch_size)

    
avoidWarnings()
comments = True
from models import Conv_Net
net = Conv_Net('net', L, M, normalize=BN)
print(net)

from collections import OrderedDict
ensemble = OrderedDict()
for n in range(1,1+K):
    ensemble['net_{}'.format(n)] = Conv_Net('net_{}'.format(n), L, M)

optimizers = []
for n in range(1,1+K):
    optimizers.append(
        optim.SGD(ensemble['net_{}'.format(n)].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    )

criterion = nn.CrossEntropyLoss()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
collector = list()
for var_name in optimizers[0].state_dict():
    print(var_name, "\t", optimizers[0].state_dict()[var_name])
    collector.append((var_name, "\t", optimizers[0].state_dict()[var_name]))
    
'''
Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.01, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 1e-05, 'nesterov': False, 
'params': [5090678032, 5090678104, 5090678320, 5090678464, 5090678680, 5090678824, 5090678896, 5090678968,
 5090679040, 5090679112, 5090679184, 5090679256, 5090679328, 5090679400, 5090679472, 5090679544, 5090678752, 
 5090679760, 5090679688, 5090679976, 5090679904, 5090680192, 5090680120, 5090680408, 5090680336, 5090680624, 
 5090680552, 5090779208, 5090779280, 5090779424, 5090779352, 5090779640, 5090779568, 5090779856, 5090779784, 5090780072]}]
'''

state = {
    'opt': [optimizers[k].state_dict() for k in range(K)]
}



for epoch in range(num_epochs):
    
    print('Epoch ', epoch)
    
    total = 0
    correct = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        if batch_idx >= 10: 
            break
        
        individual_outputs = list()
        
        for n, net in enumerate(ensemble.values()):
            
            if device == 'cuda':
                net.to(device)
        
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
            individual_outputs.append(output)
    
    
         ## Ensemble forward pass
            
        output = torch.mean(torch.stack(individual_outputs), dim=0)
        loss = criterion(output, targets) 
        
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100 * correct / total
    
    # Store iteration results for Ensemble
    print('Train :: Loss: {} | Accy: {}'.format(round(loss.item(),2), round(accuracy,2)))


# Print optimizer's state_dict
print("Optimizer's state_dict:")
collector2 = list()
for var_name in optimizers[0].state_dict():
    print(var_name, "\t", optimizers[0].state_dict()[var_name])
    collector2.append((var_name, "\t", optimizers[0].state_dict()[var_name]))
    
state2 = {
    'opt': [optimizers[k].state_dict() for k in range(K)]
}




print('==> Resuming from checkpoint..')
print("[IMPORTANT] Don't forget to rename the results object to not overwrite!! ")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

for o in range(K):
    optimizers[o].load_state_dict(state['opt'][o])

# Print optimizer's state_dict
print("Optimizer's state_dict:")
collector3 = list()
for var_name in optimizers[0].state_dict():
    print(var_name, "\t", optimizers[0].state_dict()[var_name])
    collector.append((var_name, "\t", optimizers[0].state_dict()[var_name]))
    
state3 = {
    'opt': [optimizers[k].state_dict() for k in range(K)]
}
