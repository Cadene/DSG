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

from sklearn.ensemble import RandomForestRegressor

parser = OptionParser()
parser.add_option("--xtrain", default="data/processed/X_TrainSplit_4.csv")
parser.add_option("--ytrain", default="data/processed/Y_TrainSplit_4.csv")
parser.add_option("--xval",   default="data/processed/X_ValSplit_4.csv")
parser.add_option("--yval",   default="data/processed/Y_ValSplit_4.csv")
parser.add_option("--xtest",  default="data/processed/X_test_4.csv")

parser.add_option("--dirpred",  default="data/prediction/randomforest")
parser.add_option("--dirmodel", default="models/randomforest")
parser.add_option("--dirlog",   default="logs/randomforest")

parser.add_option('--n_estimators',  default=10)
parser.add_option('--criterion',     default='mse')
parser.add_option('--max_features',  default='auto')
parser.add_option('--max_depth',     default=None)
parser.add_option('--min_samples_split',     default=2)
parser.add_option('--min_samples_leaf',     default=1)
parser.add_option('--min_weight_fraction_leaf',     default=0.)
parser.add_option('--max_leaf_nodes',     default=2)
parser.add_option('--bootstrap',     default=True)
parser.add_option('--oob_score',     default=False)
parser.add_option('--n_jobs',     default=1)
parser.add_option('--random_state',     default=1)

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
clf = RandomForestRegressor(
    n_estimators=options.n_estimators,
    criterion=options.criterion,
    max_depth=options.max_depth,
    min_samples_split=options.min_samples_split,
    min_samples_leaf=options.min_samples_leaf,
    min_weight_fraction_leaf=options.min_weight_fraction_leaf,
    max_features=options.max_features,
    max_leaf_nodes=options.max_leaf_nodes,
    bootstrap=options.bootstrap,
    oob_score=options.oob_score,
    n_jobs=options.n_jobs,
    random_state=options.random_state,
    verbose=1,
    warm_start=False)
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

