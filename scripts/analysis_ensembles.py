
from models import Conv_Net
import matplotlib.pyplot as plt
from utils import count_parameters

colors = ['pink', 'blue', 'green', 'yellow', 'purple']


def model_definition(L,M,BN,K):
    P = count_parameters(Conv_Net('',L,M,BN)) * K
    name = 'L={}  M={} P={} K={}'.format(L,M,P,K)
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
    plt.title(name)
    plt.show()


def plot_accuracy_ensembles_vs_single(L,M,BN,K, results, print_individuals=False, results_=None):
    
    global colors
    num_epochs = len(results.train_loss['ensemble'])

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
