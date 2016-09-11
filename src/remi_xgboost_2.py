import numpy as np
import pandas as pd
import xgboost as xgb
import scipy as sps
import pickle as pkl
import sys
import os
import io
from optparse import OptionParser
import scipy as sp

parser = OptionParser()
parser.add_option("--xtrain", default="data/processed/X_TrainSplit_4.csv")
parser.add_option("--ytrain", default="data/processed/Y_TrainSplit_4.csv")
parser.add_option("--xval",   default="data/processed/X_ValSplit_4.csv")
parser.add_option("--yval",   default="data/processed/Y_ValSplit_4.csv")
parser.add_option("--xtest",  default="data/processed/X_test_4.csv")

parser.add_option("--dirpred",  default="data/prediction")
parser.add_option("--dirmodel", default="models")
parser.add_option("--dirlog",   default="logs")

parser.add_option('--seed',             type="int",   default=0)
parser.add_option("--nepoch",           type="int",   default=10)

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
parser.add_option("--eta",              type="float", default=0.3)

# tree specific parameters
parser.add_option("--max_depth",        type="int",   default=3)
parser.add_option("--min_child_weight", type="int",   default=1)
parser.add_option("--gamma",            type="float", default=0.1)
parser.add_option("--subsample",        type="float", default=0.8)
parser.add_option("--colsample_bytree", type='float', default=0.8)

# class imbalance
parser.add_option("--scale_pos_weight", type="float", default=1.)
parser.add_option("--max_delta_step",   type="float", default=6.) #[0,+oo]

# regularization parameters
parser.add_option('--alpha',            type="float", default=0.)
parser.add_option("--lambdaa",          type="float", default=0.)

options, args = parser.parse_args()
list_opt = vars(options)

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

param = {}
param['objective']        = 'binary:logistic'
param['eval_metric']      = 'logloss'
param['silent']           = 1
param['nthread']          = 1

param['seed']             = options.seed
param['eta']              = options.eta
param['max_depth']        = options.max_depth
param['min_child_weight'] = options.min_child_weight
param['gamma']            = options.gamma
param['subsample']        = options.subsample
param['colsample_bytree'] = options.colsample_bytree

param['scale_pos_weight'] = options.scale_pos_weight
param['max_delta_step']   = options.max_delta_step

param['alpha']            = options.alpha
param['lambda']           = options.lambdaa

print 'Loading data...'
Xtrain = pd.read_csv(options.xtrain)  #, index_col=0)
Xval   = pd.read_csv(options.xval)    #,index_col=0)
Xtest  = pd.read_csv(options.xtest)   #,index_col=0)
Ytrain = pd.read_csv(options.ytrain)['Converted']
Yval   = pd.read_csv(options.yval)['Converted']

xg_train = xgb.DMatrix(np.array(Xtrain), label=np.array(Ytrain).flatten())
xg_val   = xgb.DMatrix(np.array(Xval),   label=np.array(Yval).flatten())
xg_test  = xgb.DMatrix(np.array(Xtest))

watchlist = [(xg_train,'train'), (xg_val,'val')]
clf = xgb.train(param, xg_train, options.nepoch, watchlist)

int_key    = np.random.randint(1,10000)
date_key   = os.times()[4] * 10
path_log   = os.path.join(options.dirlog,   'log_'  +str(date_key)+'_'+str(int_key)+'.csv')
path_model = os.path.join(options.dirmodel, 'model_'+str(date_key)+'_'+str(int_key)+'.pkl')
path_pred  = os.path.join(options.dirpred,  'pred_' +str(date_key)+'_'+str(int_key)+'.csv')

log_info = list_opt
log_info['path_log']    = path_log
log_info['path_model']  = path_model
log_info['path_pred']   = path_pred
log_info['train_score'] = [logloss(Ytrain, clf.predict(xg_train))]
log_info['val_score']   = [logloss(Yval,   clf.predict(xg_val))]

print("Saving the log to " + path_log)
df_log_info = pd.DataFrame(log_info)
df_log_info.to_csv(path_log)

print("Saving the model to " + path_model)
fm = io.open(path_model, 'wb')
pkl.dump(clf, fm)
fm.close()

print("Saving the pred to " + path_pred)
df_test_pred = pd.DataFrame(clf.predict(xg_test))
df_test_pred.to_csv(path_pred, index=False, header=None)   


