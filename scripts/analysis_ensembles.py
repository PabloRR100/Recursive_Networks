
import os
import time
import torch

from models import Conv_Net
from data import dataloaders
import matplotlib.pyplot as plt
from utils import count_parameters
from collections import OrderedDict


# Data, Device
# -------------

_, testloader_1, _ = dataloaders('CIFAR', 1)
_, testloader, classes = dataloaders('CIFAR', 128)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_models(L,M,BN,K):
    P = count_parameters(Conv_Net('',L,M,BN)) * K
    name = 'L={} M={} K={} P={}'.format(L,M,K,P)
    ensemble = [Conv_Net('',L,M,BN) for k in range(K)]
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


def time_metrics(L,M,BN,K,is_recursive,is_ensemble):
    # Function to compute and store the inference time results
    ensemble = create_models(L,M,BN,K)
    
    results = dict()
    for n,net in enumerate(ensemble.values()):
        net = load_model(net, n+1, check_path, device)
    
    for net in nets:
        print('\n\nNetwork = ', net.name)
        print('------------------------')
        img_inf_time, batch_inf_time = inference_time(net)
        results[net.name] = {'img_inf_time':img_inf_time, 'batch_inf_time':batch_inf_time}
    return results    




## TEST LOSS AND ACCY EVOLUTION
# ------------------------------

colors = ['pink', 'blue', 'green', 'yellow', 'purple']

def model_definition(L,M,BN,K):
    P = count_parameters(Conv_Net('',L,M,BN)) * K
    name = 'L={} M={} K={} P={}'.format(L,M,K,P)
    return name

def plot_loss_ensembles_vs_single(L,M,BN,K, results, print_individuals=False, results_=None):
    
    global colors
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
    
    global colors
    num_epochs = len(results.train_loss['ensemble'])

    name = model_definition(L,M,BN,K)
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(range(num_epochs), results.train_accy['ensemble'], label='Ensemble', color='black', alpha=1)

    if print_individuals:
        for m,c in zip(range(1,K+1),colors):
            ax1.plot(range(num_epochs), results.train_accy['m{}'.format(m)], label='net_{}'.format(m), color=c, alpha=0.4)
            
    if results_ is not None:
        ax1.plot(range(num_epochs), results_.train_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
        
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(range(num_epochs), results.valid_accy['ensemble'], label='Ensemble', color='black', alpha=1)

    if print_individuals:
        for m,c in zip(range(1,K+1),colors):
            ax2.plot(range(num_epochs), results.valid_accy['m{}'.format(m)], label='net_{}'.format(m), color=c, alpha=0.4)
    
    if results_ is not None:
        ax2.plot(range(num_epochs), results_.valid_accy, label='Single Model', color='red', alpha=1, linewidth=0.5)
    
    ax2.set_title('Validation Loss')
    ax2.grid(True)
    ax2.legend()
    plt.suptitle(name)
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
