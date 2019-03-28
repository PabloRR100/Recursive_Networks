
import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
from models import Conv_Net
from data import dataloaders
from results import accuracies
import matplotlib.pyplot as plt
from utils import count_parameters
from collections import OrderedDict


# Data, Device
# -------------

_, testloader_1, _ = dataloaders('CIFAR', batch_size=1)
_, testloader, classes = dataloaders('CIFAR', batch_size=128)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_models(L,M,BN,K):
    P = count_parameters(Conv_Net('',L,M,BN)) * K
    name = 'L={} M={} K={} P={}'.format(L,M,K,P)
    ensemble = OrderedDict()
    for n in range(1,1+K):
        ensemble['net_{}'.format(n)] = Conv_Net('net_{}'.format(n), L, M)
    return name, ensemble


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



# INFERENCE TIME FOR IMAGE / BATCH
# ----------------------------------

def inference_time(ensemble):
    # Function to calculate inference time
    image, _ = next(iter(testloader_1))
    images, _ = next(iter(testloader))
    
    def inference(ensemble, input):
        start = time.time()
        for net in ensemble.values():
            net(input)
        return (time.time() - start) * 1000
    
    single_image = inference(ensemble, image)
    batch_images = inference(ensemble, images)
    print('Inference time 1 image: {}ms'.format(round(single_image,3)))
    print('Inference time {} image: {}ms'.format(testloader.batch_size, round(batch_images, 3)))
    return single_image, batch_images


def time_metrics(L,M,BN,K,is_recursive):
    # Function to compute and store the inference time results
    _, ensemble = create_models(L,M,BN,K)
    img_inf_time, batch_inf_time = inference_time(ensemble)
    results = {'img_inf_time':img_inf_time, 'batch_inf_time':batch_inf_time}
    return results    



## TEST TOP-K ACCURACY AND PER CLASS METRICS
# -------------------------------------------

def test_accuracies(ensemble):
    # Function to calculate top1, top5 and classwise accuracies
    prec1, prec5 = list(), list()
    class_total = list(0. for i in range(10))
    class_correct = list(0. for i in range(10))
    
    for net in ensemble.values():
        net.eval()
        
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)
        
        # Ensemble forward pass
        individual_outputs = list()
        for net in ensemble.values():
            outputs = net(images)
            individual_outputs.append(outputs)
            
        outputs = torch.mean(torch.stack(individual_outputs), dim=0)
        _, predicted = torch.max(outputs.data, 1)

        # General Results Top1, Top5
        p1, p5 = accuracies(outputs.data, labels.data, topk=(1, 5))
        prec1.append(p1.item())
        prec5.append(p5.item())
    
        # Class-Wise Results
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    
    class_wise_accy = {classes[i]: round(100 * class_correct[i] / class_total[i], 3) \
                       for i in range(10)}
    
    print('Top-1 Accuracy = ', np.mean(prec1))
    print('Top-5 Accuracy = ', np.mean(prec5))
    
    for k,v in class_wise_accy.items(): print('Accuracy of %5s : %2d %%' % (k,v))
    return round(np.mean(prec1),3), round(np.mean(prec5),3), class_wise_accy


#from analysis import test_accuracies as indiv_test_accy
def accuracy_metrics(L,M,BN,K,is_recursive):
    
    root = './checkpoint/'
    prefix = 'Non_' if is_recursive == False else ''
    type_ = 'Ensemble'
    check_path = root + '{}_{}Recursive_L_{}_M_{}_BN_{}_K_{}.t7'.format(type_,prefix,L,M,BN,K)
    
    # Function to compute and store the accuracy results
    results = dict()
    _, ensemble = create_models(L,M,BN,K)    
    for n,net in enumerate(ensemble.values()):
        net = load_model(net, n+1, check_path, device)
        
#    for net in ensemble.values():    
    top1, top5, classwise = test_accuracies(ensemble)
    results['ensemble'] = {'top1':top1, 'top5':top5, 'classwise':classwise}
    return results  


## TEST LOSS AND ACCY EVOLUTION
# ------------------------------

colors = ['pink', 'blue', 'green', 'yellow', 'purple']

def model_definition(L,M,BN,K):
    name, _ = create_models(L,M,BN,K)
    return name

def plot_loss_ensembles_vs_single(L,M,BN,K, results, print_individuals=False, results_=None):
    
    colors = plt.cm.jet(np.linspace(0,1,K))
    num_epochs = len(results.train_loss['ensemble'])
    
    name = model_definition(L,M,BN,K)
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(range(num_epochs), results.train_loss['ensemble'], label='Ensemble', color='black', alpha=1)

    if print_individuals:
        for m,c in zip(range(1,K+1),colors):
            ax1.plot(range(num_epochs), results.train_loss['m{}'.format(m)], label='net_{}'.format(m), color=c, alpha=0.4)
            
    if results_ is not None:
        ax1.plot(range(num_epochs), results_.train_loss, label='Single Model', color='red', alpha=1, linewidth=0.5)
        
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(range(num_epochs), results.valid_loss['ensemble'], label='Ensemble', color='black', alpha=1)

    if print_individuals:
        for m,c in zip(range(1,K+1),colors):
            ax2.plot(range(num_epochs), results.valid_loss['m{}'.format(m)], label='net_{}'.format(m), color=c, alpha=0.4)
    
    if results_ is not None:
        ax2.plot(range(num_epochs), results_.valid_loss, label='Single Model', color='red', alpha=1, linewidth=0.5)
    
    ax2.set_title('Validation Loss')
    ax2.grid(True)
    ax2.legend()
    plt.suptitle(name)
    plt.show()


def plot_accuracy_ensembles_vs_single(L,M,BN,K, results, print_individuals=False, results_=None):
    
    colors = plt.cm.jet(np.linspace(0,1,K))
    num_epochs = len(results.train_loss['ensemble'])

    name = model_definition(L,M,BN,K)
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(range(num_epochs), results.train_accy['ensemble'], label='Ensemble', color='black', alpha=1)

    if print_individuals:
        for m,c in zip(range(1,K+1),colors):
            ax1.plot(range(num_epochs), results.train_accy['m{}'.format(m)], label='net_{}'.format(m), color=c, alpha=0.4)
            
    if results_ is not None:
        ax1.plot(range(num_epochs), results_.train_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
        
    ax1.set_title('Training Accuracy')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(range(num_epochs), results.valid_accy['ensemble'], label='Ensemble', color='black', alpha=1)

    if print_individuals:
        for m,c in zip(range(1,K+1),colors):
            ax2.plot(range(num_epochs), results.valid_accy['m{}'.format(m)], label='net_{}'.format(m), color=c, alpha=0.4)
    
    if results_ is not None:
        ax2.plot(range(num_epochs), results_.valid_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
    
    ax2.set_title('Validation Accuracy')
    ax2.grid(True)
    ax2.legend()
    plt.suptitle(name)
    plt.show()


def plot_classwise_accuracy(L,M,BN,K,recursive,results=None):
    
    if results is None:
        res = pd.DataFrame(accuracy_metrics(L,M,BN,recursive))           
    else:
        res = pd.DataFrame(results)
        
    c = []
    clas = res.loc[['classwise']].T.reset_index()
    for i in range(clas.shape[0]):
        c.append(clas.iloc[i,1])
    
    clas = pd.DataFrame(c, index=res.keys())
        
    X = np.arange(clas.shape[1])
    # Class-Wise
    fig, axs = plt.subplots(nrows=len(res.keys()))
    if len(res.keys()) == 1: axs = [axs]
    for i,r in enumerate(res.keys()):
        axs[i].set_title(str(clas.index[i]))
        axs[i].bar(X, clas.loc[r,:], width=0.8)
        axs[i].set_xticks(X)
        axs[i].set_xticklabels(clas.columns)
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.boxplot(clas, labels=clas.index)
    plt.grid()
    plt.title('Mean - Variance Comparison')
    plt.show()
    



def load_training_results(L,M,BN,K,is_recursive=False):
    # Function to create models given the hyperparameters    
    err = 'Input should be a list'
    err2 = 'rec should be a boolean indicanting if recursive architecture'
    assert isinstance(L, list), err
    assert isinstance(M, list), err
    assert isinstance(BN, list), err
    assert isinstance(is_recursive, bool), err2

    preroot = 'non_' if is_recursive == False else ''
    root = '../results/dicts/ensemble_{}recursives/'.format(preroot)
    prefix = 'Non_' if is_recursive == False else ''
    paths = [root + 'Ensemble_{}Recursive_L_{}_M_{}_BN_{}_K_{}.pkl'.\
             format(prefix,l,m,b,k) for l,m,b,k in zip(L,M,BN,K)]
        
    def load_dict(path):
        with open(path, 'rb') as input:
            return pickle.load(input)

    return [load_dict(path) for path in paths]


def plot_compare_ensembles_accuracy(L,M,BN,K,recursive=False,results=None, results_=None):
    
    colors = plt.cm.jet(np.linspace(0,1,len(K)))
    if results is None:
        results = load_training_results(L,M,BN,K,recursive)        
        
    num_epochs = len(results[0].train_loss['ensemble'])
    names = [create_models(l,m,bn,k)[0] for l,m,bn,k in zip(L,M,BN,K)]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    for c,name,result in zip(colors,names,results):
        ax1.plot(range(num_epochs), result.train_accy['ensemble'], label=name, color=c, alpha=1)

    if results_ is not None:
        name = 'Single: ' + results_.name
        ax1.plot(range(num_epochs), results_.train_accy, label=name, color='red', alpha=1, linewidth=0.5)
        
    ax1.set_title('Training Accuracy')
    ax1.grid(True)
    ax1.legend()
    
    for c,name,result in zip(colors,names,results):
        ax2.plot(range(num_epochs), result.valid_accy['ensemble'], label=name, color=c, alpha=1)    
    
    if results_ is not None:
        name = 'Single: ' + results_.name
        ax2.plot(range(num_epochs), results_.valid_accy, label=name, color='red', alpha=1, linewidth=0.5)
    
    ax2.set_title('Validation Accuracy')
    ax2.grid(True)
    ax2.legend()
    plt.suptitle('Comparing different ensembles')
    plt.show()


def plot_compare_ensembles_loss(L,M,BN,K,recursive=False,results=None, results_=None):
    
    colors = plt.cm.jet(np.linspace(0,1,len(K)))
    if results is None:
        results = load_training_results(L,M,BN,K,recursive)        
        
    num_epochs = len(results[0].train_loss['ensemble'])
    names = [create_models(l,m,bn,k)[0] for l,m,bn,k in zip(L,M,BN,K)]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    for c,name,result in zip(colors,names,results):
        ax1.plot(range(num_epochs), result.train_loss['ensemble'], label=name, color=c, alpha=1)

    if results_ is not None:
        name = 'Single: ' + results_.name
        ax1.plot(range(num_epochs), results_.train_loss, label=name, color='red', alpha=1, linewidth=0.5)
        
    ax1.set_title('Training Accuracy')
    ax1.grid(True)
    ax1.legend()
    
    for c,name,result in zip(colors,names,results):
        ax2.plot(range(num_epochs), result.valid_loss['ensemble'], label=name, color=c, alpha=1)    
    
    if results_ is not None:
        name = 'Single: ' + results_.name
        ax2.plot(range(num_epochs), results_.valid_loss, label=name, color='red', alpha=1, linewidth=0.5)
    
    ax2.set_title('Validation Accuracy')
    ax2.grid(True)
    ax2.legend()
    plt.suptitle('Comparing different ensembles')
    plt.show()













#
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#for m in range(1,1+E):
#    ax1.plot(range(num_epochs), results.train_loss['m{}'.format(m)], label='m{}'.format(m), color=c[m], alpha=0.4)
#ax1.plot(range(num_epochs), results.train_loss['ensemble'], label='Ensemble', color='black', alpha=1)
#if psm: ax1.plot(range(num_epochs), results_.train_loss, label='Single Model', color='red', alpha=1, linewidth=0.5)
#ax1.set_title('Trianing Loss')
#ax1.grid(True)
#
#for m in range(1,1+E):
#    ax2.plot(range(num_epochs), results.valid_loss['m{}'.format(m)], label='m{}'.format(m), color=c[m], alpha=0.4)
#ax2.plot(range(num_epochs), results.valid_loss['ensemble'], label='Ensemble', color='black', alpha=1)
#if psm: ax2.plot(range(num_epochs), results_.valid_loss, label='Single Model', color='red', alpha=1, linewidth=0.5)
#ax2.set_title('Validation Loss')
#ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax2.grid(True)
#
#for m in range(1,1+E):
#    ax3.plot(range(num_epochs), results.train_accy['m{}'.format(m)], label='m{}'.format(m), color=c[m], alpha=0.4)
#ax3.plot(range(num_epochs), results.train_accy['ensemble'], label='Ensemble', color='black', alpha=1)
#if psm: ax3.plot(range(num_epochs), results_.train_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
#ax3.set_title('Training Accuracy')
#ax3.grid(True)
#
#for m in range(1,1+E):
#    ax4.plot(range(num_epochs), results.valid_accy['m{}'.format(m)], label='m{}'.format(m), color=c[m], alpha=0.4)
#ax4.plot(range(num_epochs), results.valid_accy['ensemble'], label='Ensemble', color='black', alpha=1)
#if psm: ax4.plot(range(num_epochs), results_.valid_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
#ax4.set_title('Validation Accuracy')
#ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax4.grid(True)
#plt.title()
#plt.show()
#
