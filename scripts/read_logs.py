
## Introduce Epoch 0 as initialization of the model? -> just include first line of the logs
## densenet.txt reaches 141 epochs but was saved at 96

import os
import pickle


## Introduce the correct path ##
path = os.path.abspath('../results/ensemble_recursive_model/definitives/ensemble.txt')  

f = open(path, 'r')
x = f.readlines()
x = [t for t in x if (t != '\n')]
f.close()

MODELS = 1

def get_loss_accy(tr):
    loss = list(map(float, [t.split('Loss: ')[1].split(' |')[0] for t in tr if 'Loss' in t]))
    accy = list(map(float, [t.split('Accy: ')[1].split('%')[0] for t in tr if 'Accy' in t]))
    return loss, accy

# Training
# --------

# Single Model

tr_x = [t for t in x if 'Train' in t]
tr_loss, tr_accy = get_loss_accy(tr_x)


# Validation
# ----------

va_x = [t for t in x if 'Valid' in t]
va_loss, va_accy = get_loss_accy(va_x)


# Timer
# -----

## TODO: get mean run_epoch: time and multiply by num_epochs

# Save to Result Class
# --------------------

from results import TrainResults

def remove_empty_keys(d):
    d = {k: v for k, v in d.items() if v != []}
    return d

MODELS = 5
models = ['m'+str(i) for i in range(1, MODELS+1)]
eres = TrainResults(models)
eres.name = 'Recursive (x5)'

eres.train_loss['ensemble'] = tr_loss
eres.train_accy['ensemble'] = tr_accy
eres.valid_loss['ensemble'] = va_loss
eres.valid_accy['ensemble'] = va_accy
eres.train_loss = remove_empty_keys(eres.train_loss)
eres.train_accy = remove_empty_keys(eres.train_accy)
eres.valid_loss = remove_empty_keys(eres.valid_loss)
eres.valid_accy = remove_empty_keys(eres.valid_accy)


with open('Results_Ensemble_Recursive.pkl', 'wb') as object_result:
    pickle.dump(eres, object_result, pickle.HIGHEST_PROTOCOL)
