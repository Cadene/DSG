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

from sklearn.linear_model import LogisticRegression

parser = OptionParser()
parser.add_option("--xtrain", default="data/processed/X_TrainSplit_4.csv")
parser.add_option("--ytrain", default="data/processed/Y_TrainSplit_4.csv")
parser.add_option("--xval",   default="data/processed/X_ValSplit_4.csv")
parser.add_option("--yval",   default="data/processed/Y_ValSplit_4.csv")
parser.add_option("--xtest",  default="data/processed/X_test_4.csv")

parser.add_option("--dirpred",  default="data/prediction/logistic")
parser.add_option("--dirmodel", default="models/logistic")
parser.add_option("--dirlog",   default="logs/logistic")

parser.add_option('--random_state',  default=1, type='float') # seed

#parser.add_option("--penalty",      default="l2")
#parser.add_option("--dual",         default=False)
#parser.add_option("--tol",          default=0.0001)
parser.add_option("--C",            default=1.0, type='float')
#parser.add_option("--fit_intercept", default=True)
parser.add_option("--class_weight",  default='balanced') # auto calculated
#parser.add_option("--max_iter",      default=100)

parser.add_option("--n_jobs",        default=1)

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

if option.class_weight == "None":
    option.class_weight = None

print 'Fitting...'
clf = LogisticRegression(
    penalty='l2',
    dual=False,
    C=options.C,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=options.class_weight,
    random_state=options.random_state,
    solver='liblinear',
    verbose=0,
    warm_start=False,
    n_jobs=options.n_jobs)
print('Before clf.fit', int((time.time() - t_start) * 1000))
clf.fit(Xtrain, Ytrain)
print('After clf.fit', int((time.time() - t_start) * 1000))

print(clf.classes_)

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
log_info['train_score'] = [logloss(Ytrain, clf.predict_proba(Xtrain)[:,1])]
log_info['val_score']   = [logloss(Yval,   clf.predict_proba(Xval)[:,1])]
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
df_test_pred = pd.DataFrame(clf.predict_proba(Xtest)[:,1])
df_test_pred.to_csv(path_pred, index=False, header=None) 
print('End', int((time.time() - t_start) * 1000))