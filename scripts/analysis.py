#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:43:42 2019
@author: pabloruizruiz
"""

import os
import time
import torch
import pickle
import numpy as np
from models import Conv_Net
from data import dataloaders
from results import accuracies
from utils import count_parameters
from collections import OrderedDict



# Data, Device
# -------------

_, testloader_1, _ = dataloaders('CIFAR', 1)
_, testloader, classes = dataloaders('CIFAR', 128)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(net, check_path, device):
    # Function to load saved models
    def load_weights(check_path):
        assert os.path.exists(check_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(check_path, map_location=device)
        new_state_dict = OrderedDict()
        for k,v in checkpoint['net'].items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    
    if device == 'cpu': net.load_state_dict(load_weights(check_path)) # remove word `module`
    else: net.load_state_dict(torch.load(check_path)['net'])
    
    net.to(device)
    if device == 'cuda': net = torch.nn.DataParallel(net)
    return net



# INFERENCE TIME FOR IMAGE / BATCH
# ----------------------------------

def inference_time(net):
    # Function to calculate inference time
    image, _ = next(iter(testloader_1))
    images, _ = next(iter(testloader))
    
    def inference(net, input):
        start = time.time()
        net(input)
        return (time.time() - start) * 1000
    
    single_image = inference(net, image)
    batch_images = inference(net, images)
    print('Inference time 1 image: {}ms'.format(round(single_image,3)))
    print('Inference time {} image: {}ms'.format(testloader.batch_size, round(batch_images, 3)))
    return single_image, batch_images



## TEST TOP-K ACCURACY AND PER CLASS METRICS
# -------------------------------------------

def test_accuracies(net):
    # Function to calculate top1, top5 and classwise accuracies
    prec1, prec5 = list(), list()
    class_total = list(0. for i in range(10))
    class_correct = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in testloader:
    
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
    
            # General Results Top1, Top5
            p1, p5 = accuracies(outputs.data, labels.data, topk=(1, 5))
            prec1.append(p1.item())
            prec5.append(p5.item())
            
            # Class-Wise Results
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    class_wise_accy = {classes[i]: round(100 * class_correct[i] / class_total[i], 3) \
                       for i in range(10)}
    
    
    print('Top-1 Accuracy = ', np.mean(prec1))
    print('Top-5 Accuracy = ', np.mean(prec5))
    
    for k,v in class_wise_accy.items(): print('Accuracy of %5s : %2d %%' % (k,v))
    return round(np.mean(prec1),3), round(np.mean(prec5),3), class_wise_accy



def load_models(L,M,BN,is_recursive=False, is_ensemble=False):
    # Function to create models given the hyperparameters    
    err = 'Input should be a list'
    err2 = 'rec should be a boolean indicanting if recursive architecture'
    assert isinstance(L, list), err
    assert isinstance(M, list), err
    assert isinstance(BN, list), err
    assert isinstance(is_recursive, bool), err2
    
    root = './checkpoint/'
    prefix = 'Non_' if is_recursive == False else ''
    type_ = 'Single' if is_ensemble == False else 'Ensemble'
    paths = [root + '{}_{}Recursive_L_{}_M_{}_BN_{}.t7'.\
             format(type_,prefix,l,m,b) for l,m,b in zip(L,M,BN)]
    
    P = [count_parameters(Conv_Net('',l,m,bn)) for l,m,bn in zip(L,M,BN)]   ## TODO: adjust for Rec_Conv_Net
    names = ['L={}  M={} P={}'.format(l,m,p) for l,m,p in zip(L,M,P)]
    models = [Conv_Net(n,l,m) for n,l,m in zip(names,L,M)]

    nets = []
    for check_path,model in zip(paths,models):
        nets.append(load_model(model, check_path, device))
    return models


def accuracy_metrics(L,M,BN,is_recursive, is_ensemble):
    # Function to compute and store the accuracy results
    results = dict()
    nets = load_models(L,M,BN,is_recursive,is_ensemble)
    for net in nets:
        print('\n\nNetwork = ', net.name)
        print('===================================')
        top1, top5, classwise = test_accuracies(net)
        results[net.name] = {'top1':top1, 'top5':top5, 'classwise':classwise}
    return results    


def time_metrics(L,M,BN,is_recursive,is_ensemble):
    # Function to compute and store the inference time results
    results = dict()
    nets = load_models(L,M,BN,is_recursive,is_ensemble)
    for net in nets:
        print('\n\nNetwork = ', net.name)
        print('------------------------')
        img_inf_time, batch_inf_time = inference_time(net)
        results[net.name] = {'img_inf_time':img_inf_time, 'batch_inf_time':batch_inf_time}
    return results    





r = accuracy_metrics(L,M,BN,recursive,ensemble)




colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink']



def load_training_results(L,M,BN,is_recursive=False, is_ensemble=False):
    # Function to create models given the hyperparameters    
    err = 'Input should be a list'
    err2 = 'rec should be a boolean indicanting if recursive architecture'
    assert isinstance(L, list), err
    assert isinstance(M, list), err
    assert isinstance(BN, list), err
    assert isinstance(is_recursive, bool), err2

    preroot = 'non_' if is_recursive == False else ''
    root = '../results/dicts/single_{}recursive/'.format(preroot)
    prefix = 'Non_' if is_recursive == False else ''
    type_ = 'Single' if is_ensemble == False else 'Ensemble'
    paths = [root + '{}_{}Recursive_L_{}_M_{}_BN_{}.pkl'.\
             format(type_,prefix,l,m,b) for l,m,b in zip(L,M,BN)]
        
    def load_dict(path):
        with open(path, 'rb') as input:
            return pickle.load(input)

    return [load_dict(path) for path in paths]


r2 = load_training_results(L,M,BN,recursive,ensemble)

## TEST LOSS AND ACCY EVOLUTION
# ------------------------------

import matplotlib.pyplot as plt

# OF A SINGLE MODEL
# ------------------

L = [16]
M = [64]
BN = [False] * len(L)
recursive = False
ensemble = False
num_epochs = 700

        ## Single Non Recursive Results
path = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_16_M_16_BN_False.pkl'
#path = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_16_M_32_BN_False.pkl'
#path = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_16_M_64_BN_False.pkl'
#path = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_32_M_64_BN_False.pkl'
#path = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_32_M_32_BN_False.pkl'

with open(path, 'rb') as input:
    results = pickle.load(input)

plt.figure()
plt.title('Loss :: L={} M={} BN={}'.format(L,M,BN))
plt.plot(range(num_epochs), results.train_loss, label='Train')
plt.plot(range(num_epochs), results.valid_loss, label='Valid')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title('Accuracy :: L={} M={} BN={}'.format(L,M,BN))
plt.plot(range(num_epochs), results.train_accy, label='Train')
plt.plot(range(num_epochs), results.valid_accy, label='Valid')
plt.legend()
plt.grid()
plt.show()



# OF MULTIPLE SINGLES
# --------------------

num_epochs = 700
L = [16,32,16,32,16,32]
M = [16,16,32,32,64,64]
BN = [False] * len(L)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink']
P = [count_parameters(Conv_Net('',l,m,bn)) for l,m,bn in zip(L,M,BN)]


root = '../results/dicts/single_non_recursive/'
paths = [root + 'Single_Non_Recursive_L_{}_M_{}_BN_{}.pkl'.format(l,m,b) for l,m,b in zip(L,M,BN)]
names = ['L={}  M={} P={}'.format(l,m,p) for l,m,p in zip(L,M,P)]

results = []
for path in paths:
    with open(path, 'rb') as input:
        results.append(pickle.load(input))
        
plt.figure()
plt.title('Loss :: L = Layers, M = Filters, P = Parameters')
for c,name,result in zip(colors,names,results):
    plt.plot(range(num_epochs), result.train_loss, label='Train ' + name, color=c, linewidth=0.5)
    plt.plot(range(num_epochs), result.valid_loss, label='Valid ' + name, color=c, alpha=0.5, linestyle='--')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title('Accuracy :: L = Layers, M = Filters, P = Parameters')
for c,name,result in zip(colors,names,results):
    plt.plot(range(num_epochs), result.train_accy, label='Train ' + name, color=c)
    plt.plot(range(num_epochs), result.valid_accy, label='Valid ' + name, color=c, alpha=0.5, linestyle='--')
plt.legend()
plt.grid()
plt.show()





concat = False
if concat:
    ## CONCAT 2 RESULTS
    # -----------------
    
    path1 = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_16_M_32_I.pkl'
    path2 = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_16_M_32_II.pkl'
    path_concat = '../results/dicts/single_non_recursive/Single_Non_Recursive_L_16_M_32.pkl'
    
    def concat_resumed_training(path1, path2, resume_at):
        with open(path1, 'rb') as input: results1 = pickle.load(input)
        with open(path2, 'rb') as input: results2 = pickle.load(input)
        results1.train_loss = results1.train_loss[:resume_at] + results2.train_loss
        results1.train_accy = results1.train_accy[:resume_at] + results2.train_accy
        results1.valid_loss = results1.valid_loss[:resume_at] + results2.valid_loss
        results1.valid_accy = results1.valid_accy[:resume_at] + results2.valid_accy
        return results1
    
    res = concat_resumed_training(path1, path2, 481)
    with open(path_concat, 'wb') as object_result:
            pickle.dump(res, object_result, pickle.HIGHEST_PROTOCOL)   
    