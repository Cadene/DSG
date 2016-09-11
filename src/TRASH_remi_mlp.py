import numpy as np
import pandas as pd
import pickle as pkl
from optparse import OptionParser
import sys
import os
import io
import scipy as sp
import time
t_start = time.time()

from sklearn.neural_network import MLPRegressor

parser = OptionParser()
parser.add_option("--xtrain", default="data/processed/X_TrainSplit_4.csv")
parser.add_option("--ytrain", default="data/processed/Y_TrainSplit_4.csv")
parser.add_option("--xval",   default="data/processed/X_ValSplit_4.csv")
parser.add_option("--yval",   default="data/processed/Y_ValSplit_4.csv")
parser.add_option("--xtest",  default="data/processed/X_test_4.csv")

parser.add_option("--dirpred",  default="data/prediction/mlp")
parser.add_option("--dirmodel", default="models/mlp")
parser.add_option("--dirlog",   default="logs/mlp")

parser.add_option('--hidden_layer_sizes',  default=(100,))
parser.add_option('--activation',  default='relu')
parser.add_option('--algorithm',  default='adam')
parser.add_option('--alpha',  default=0.0001) # L2?
parser.add_option('--batch_size',  default=256)
parser.add_option('--learning_rate',  default='invscaling') # adaptative?
parser.add_option('--max_iter',  default=200)
parser.add_option('--random_state',  default=1)
parser.add_option('--shuffle',  default=True)
parser.add_option('--tol',  default=1e-4)
parser.add_option('--learning_rate_init',  default=0.001)
parser.add_option('--power_t',  default=0.5) # only learning_rate invscaling
# parser.add_option('--momentum',  default=0.9) # only sgd
# parser.add_option('--nesterovs_momentum',  default=True) # only sgd
parser.add_option('--early_stopping',  default=False)

options, args = parser.parse_args()
list_opt = vars(options)

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

print 'Loading data...'
print('Before read_csv', int((time.time() - t_start) * 1000))
Xtrain = pd.read_csv(options.xtrain)
Xval   = pd.read_csv(options.xval)
Xtest  = pd.read_csv(options.xtest)
Ytrain = pd.read_csv(options.ytrain)['Converted']
Yval   = pd.read_csv(options.yval)['Converted']
print('After read_csv', int((time.time() - t_start) * 1000))

print 'Fitting...'
clf = MLPRegressor(
    hidden_layer_sizes=options.hidden_layer_sizes,
    activation=options.activation,
    algorithm=options.algorithm,
    alpha=options.alpha,
    batch_size=options.batch_size,
    learning_rate=options.learning_rate,
    learning_rate_init=options.learning_rate_init,
    power_t=options.power_t,
    max_iter=options.max_iter,
    shuffle=options.shuffle,
    random_state=options.random_state,
    tol=options.tol,
    verbose=False,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=options.early_stopping,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08)
print('Before clf.fit', int((time.time() - t_start) * 1000))
clf.fit(Xtrain, Ytrain)
print('After clf.fit', int((time.time() - t_start) * 1000))

int_key    = np.random.randint(1,10000)
date_key   = os.times()[4] * 10
path_log   = os.path.join(options.dirlog,   'log_'  +str(date_key)+'_'+str(int_key)+'.csv')
path_model = os.path.join(options.dirmodel, 'model_'+str(date_key)+'_'+str(int_key)+'.pkl')
path_pred  = os.path.join(options.dirpred,  'pred_' +str(date_key)+'_'+str(int_key)+'.csv')

log_info = list_opt
log_info['path_log']    = path_log
log_info['path_model']  = path_model
log_info['path_pred']   = path_pred
print('Before clf.predict train val', int((time.time() - t_start) * 1000))
log_info['train_score'] = [logloss(Ytrain, clf.predict(Xtrain))]
log_info['val_score']   = [logloss(Yval,   clf.predict(Xval))]
print('After clf.predict train val', int((time.time() - t_start) * 1000))

print("Saving the log to " + path_log)
df_log_info = pd.DataFrame(log_info)
df_log_info.to_csv(path_log)

print("Saving the model to " + path_model)
fm = io.open(path_model, 'wb')
pkl.dump(clf, fm)
fm.close()

print("Saving the pred to " + path_pred)
print('Before clf.predict test', int((time.time() - t_start) * 1000))
df_test_pred = pd.DataFrame(clf.predict(Xtest))
df_test_pred.to_csv(path_pred, index=False, header=None) 
print('End', int((time.time() - t_start) * 1000))

