import numpy as np
import pandas as pd
import pickle as pkl
from optparse import OptionParser
import sys
import os
import io

from sklearn.svm import SVR

parser = OptionParser()
parser.add_option("--xtrain", default="data/interim/X_train_features_1.csv")
parser.add_option("--ytrain", default="data/raw/Y_train.csv")
parser.add_option("--xval",   default="data/interim/X_train_features_1.csv")
parser.add_option("--yval",   default="data/raw/Y_train.csv")
parser.add_option("--xtest",  default="data/interim/X_test_features_1.csv")

parser.add_option("--dirpred",  default="data/prediction/logistic")
parser.add_option("--dirmodel", default="models/logistic")
parser.add_option("--dirlog",   default="logs/logistic")

parser.add_option("--penalty",  default="l2")
parser.add_option("--dual",  default=False)
parser.add_option("--tol",  default=0.0001)
parser.add_option("--C",  default=1.0)
#(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)[source]¶

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
Xtrain = pd.read_csv(options.xtrain, index_col=0)
Xval   = pd.read_csv(options.xval,   index_col=0)
Xtest  = pd.read_csv(options.xtest,  index_col=0)
Ytrain = pd.read_csv(options.ytrain)['Converted']
Yval   = pd.read_csv(options.yval)['Converted']

print 'Fitting...'
clf = LogisticRegression(class_weight='balanced')
clf.fit(Xtrain, Ytrain)

log_info = list_opt
log_info['path_log']    = path_log
log_info['path_model']  = path_model
log_info['path_pred']   = path_pred
log_info['train_score'] = [logloss(Ytrain, clf.predict(Xtrain))]
log_info['val_score']   = [logloss(Yval,   clf.predict(Xval))]

print("Saving the log to " + path_log)
df_log_info = pd.DataFrame(log_info)
df_log_info.to_csv(path_log)

print("Saving the model to " + path_model)
fm = io.open(path_model, 'wb')
pkl.dump(clf, fm)
fm.close()

print("Saving the pred to " + path_pred)
df_test_pred = pd.DataFrame(clf.predict(Xtest))
df_test_pred.to_csv(path_pred, index=False, header=None) 
