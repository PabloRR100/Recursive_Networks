
## Introduce Epoch 0 as initialization of the model? -> just include first line of the logs
## densenet.txt reaches 141 epochs but was saved at 96

import os
import pickle


## TODO: Introduce the correct path ##
path = os.path.abspath('../results/ensemble_non_recursives/ensemble_non_recursive.txt')  
path = os.path.abspath('../results/ensemble_recursives/definitives/ensemble_recursive.txt')  

f = open(path, 'r')
x = f.readlines()
x = [t for t in x if (t != '\n')]
f.close()

MODELS = 5

def get_loss_accy(tr):
    loss = list(map(float, [t.split('Loss: ')[1].split(' |')[0] for t in tr if 'Loss' in t]))
    accy = list(map(float, [t.split('Accy: ')[1].split('%')[0] for t in tr if 'Accy' in t]))
    return loss, accy

# Training
# --------

# Single Model

tr_x = [t for t in x if 'Train' in t]
tr_loss, tr_accy = get_loss_accy(tr_x)
tr_accy_max = max(tr_accy)

# Validation
# ----------

va_x = [t for t in x if 'Valid' in t]
va_loss, va_accy = get_loss_accy(va_x)
va_accy_max = max(va_accy)


# Timer
# -----
time = list(map(float, [t.split()[1] for t in x if 'run_epoch' in t]))
time_per_epoch = sum(time)/len(time)



## TODO: get mean run_epoch: time and multiply by num_epochs

# Save to Result Class
# --------------------

# Create new results to populate
from results import TrainResults

def remove_empty_keys(d):
    d = {k: v for k, v in d.items() if v != []}
    return d

MODELS = 5
models = ['m'+str(i) for i in range(1, MODELS+1)]
res = TrainResults(models)
res.name = 'Recursive (x5)'

# Use a created results
path = os.path.abspath('../results/ensemble_non_recursives/Results_Ensemble_Recursive.pkl')  
with open(path, 'rb') as input: res = pickle.load(input)

res.train_loss['ensemble'] = tr_loss
res.train_accy['ensemble'] = tr_accy
res.valid_loss['ensemble'] = va_loss
res.valid_accy['ensemble'] = va_accy
res.train_loss = remove_empty_keys(res.train_loss)
res.train_accy = remove_empty_keys(res.train_accy)
res.valid_loss = remove_empty_keys(res.valid_loss)
res.valid_accy = remove_empty_keys(res.valid_accy)


# Save results object from the log
## TODO: set the path and name of the results - avoid overwrite!
name = 'Results_Ensemble_Recursive.pkl'
with open(name, 'wb') as object_result:
    pickle.dump(res, object_result, pickle.HIGHEST_PROTOCOL)







