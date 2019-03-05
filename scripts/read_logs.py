
## Introduce Epoch 0 as initialization of the model? -> just include first line of the logs
## densenet.txt reaches 141 epochs but was saved at 96

import os
import pickle


## Introduce the correct path ##
path = os.path.abspath('results/logs/ensemble/densenet.txt')  
path2 = os.path.abspath('results/logs/ensemble/densenet2.txt')  

f = open(path, 'r')
x = f.readlines()
x = [t for t in x if (t != '\n')]
f.close()

MODELS = 1

def get_loss_accy(tr):
    loss = list(map(float, [t.split('Loss: ')[1].split(' |')[0] for t in tr]))
    accy = list(map(float, [t.split('Acc: ')[1].split('%')[0] for t in tr]))
    return loss, accy

# Training
# --------

# Single Model

tr_x = [t for t in x if '391/391' in t]
tr_loss, tr_accy = get_loss_accy(tr_x)


# Validation
# ----------

va_x = [t for t in x if '100/100' in t]
va_loss2, va_accy2 = get_loss_accy(va_x)

# Each learner
## NO INFORMATION RECORDED IN THE LOG FOR THIS
#va_singles = []
#for i in range(1,MODELS+1):
#    va_singles.append([t for t in va if 'Model {}'.format(i) in t])    
#va_singles_loss = [get_loss_accy(v)[0] for v in va_singles]
#va_singles_accy = [get_loss_accy(v)[1] for v in va_singles]
#va_ensemble_loss, va_ensemble_accy = get_loss_accy(va_ensemble)

#single_validation_results = dict(
#       tr_loss = va_single_loss,
#       tr_accy = va_single_accy)
#
#with open('Results_Single_Valiation.pkl', 'wb') as object_result:
#        pickle.dump(single_validation_results, object_result, pickle.HIGHEST_PROTOCOL)


# Testing
# -------

ts = x[-4:]
ts_single = ts[0].split('%')[0][-5:]
ts_ensemble = ts[2].split('%')[0][-5:]



# Timer
# -----

h0, m0 =  x[0].split('Time ')[1].split(':')
h1, m1 = x[640].split('Time ')[1].split(':')
h0, m0, h1, m1 = int(h0), int(m0), int(h1), int(m1)
dh_single = (h1 + 24 - h0 if h1 < h0 else h1 - h0) * 60 + (m1+60-m0 if m1 > m0 else m1)


h0, m0 = x[540].split('Time ')[1].split(':')
h1, m1 = x[1641].split(':')
h0, m0, h1, m1 = int(h0), int(m0), int(h1), int(m1) + 35
dh_essemble = (h1 + 24 - h0 if h1 < h0 else h1 - h0) * 60 + (m1+60-m0 if m1 > m0 else m1)



# Save to Result Class
# --------------------

from results import TrainResults, TestResults

def remove_empty_keys(d):
    d = {k: v for k, v in d.items() if v != []}
    return d


# Single Model

res = TrainResults([0])
res.name = 'DenseNet121'
res.train_loss = res_['train_loss']
res.train_accy = res_['train_accy']
res.valid_loss = res_['valid_loss']
res.valid_accy = res_['valid_accy']
res.timer = dh_single


# Ensemble Model
MODELS =7
models = ['m'+str(i) for i in range(1, MODELS+1)]
eres = TrainResults(models)
eres.name = 'DenseNet-CIFAR(x7)'

eres.train_loss['ensemble'] = eres_['train_loss']
eres.train_accy['ensemble'] = eres_['train_accy']
eres.valid_loss['ensemble'] = eres_['valid_loss']
eres.valid_accy['ensemble'] = eres_['valid_accy']
eres.train_loss = remove_empty_keys(eres.train_loss)
eres.train_accy = remove_empty_keys(eres.train_accy)
eres.valid_loss = remove_empty_keys(eres.train_loss)
eres.valid_accy = remove_empty_keys(eres.train_accy)

#for i, m in enumerate(models): 
#    eres.valid_loss[m] = va_singles_loss[i]
#for i, m in enumerate(models): 
#    eres.valid_accy[m] = va_singles_accy[i]


# Testing 

tres = TestResults()
tres.single_accy = ts_single
tres.ensemble_accy = ts_ensemble


with open('Results_Single_Models.pkl', 'wb') as object_result:
    pickle.dump(res, object_result, pickle.HIGHEST_PROTOCOL)

with open('Results_Ensemble_Models.pkl', 'wb') as object_result:
    pickle.dump(eres, object_result, pickle.HIGHEST_PROTOCOL)

with open('Results_Testing.pkl', 'wb') as object_result:
    pickle.dump(tres, object_result, pickle.HIGHEST_PROTOCOL)

