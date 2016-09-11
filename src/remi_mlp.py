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

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
#from keras.utils.np_utils import to_categorical


parser = OptionParser()
parser.add_option("--xtrain", default="data/processed/X_TrainSplit_4.csv")
parser.add_option("--ytrain", default="data/processed/Y_TrainSplit_4.csv")
parser.add_option("--xval",   default="data/processed/X_ValSplit_4.csv")
parser.add_option("--yval",   default="data/processed/Y_ValSplit_4.csv")
parser.add_option("--xtest",  default="data/processed/X_test_4.csv")

parser.add_option("--dirpred",  default="data/prediction/mlp")
parser.add_option("--dirmodel", default="models/mlp")
parser.add_option("--dirlog",   default="logs/mlp")

parser.add_option('--nb_epoch',  default=1)
parser.add_option('--batch_size',  default=256)

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
#Ytrain = to_categorical(Ytrain)
#Yval   = to_categorical(Yval)
print('After read_csv', int((time.time() - t_start) * 1000))

print(Xtrain.shape)

print 'Fitting...'
clf = Sequential()
clf.add(Dense(output_dim=20, input_dim=142))
clf.add(Activation("relu"))
clf.add(Dense(output_dim=1))
clf.add(Activation("softmax"))

clf.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

print('Before clf.fit', int((time.time() - t_start) * 1000))
clf.fit(Xtrain, Ytrain, nb_epoch=options.nb_epoch, batch_size=options.batch_size)
print('After clf.fit', int((time.time() - t_start) * 1000))

int_key    = np.random.randint(1,10000)
date_key   = os.times()[4] * 10
path_log   = os.path.join(options.dirlog,   'log_'  +str(date_key)+'_'+str(int_key)+'.csv')
path_model = os.path.join(options.dirmodel, 'model_'+str(date_key)+'_'+str(int_key)+'.h5')
path_pred  = os.path.join(options.dirpred,  'pred_' +str(date_key)+'_'+str(int_key)+'.csv')

log_info = list_opt
log_info['path_log']    = path_log
log_info['path_model']  = path_model
log_info['path_pred']   = path_pred
print('Before clf.predict train val', int((time.time() - t_start) * 1000))
log_info['train_score'] = [logloss(Ytrain, clf.predict_proba(Xtrain, batch_size=options.batch_size))]
log_info['val_score']   = [logloss(Yval,   clf.predict_proba(Xval, batch_size=options.batch_size))]
print('After clf.predict train val', int((time.time() - t_start) * 1000))

print("Saving the log to " + path_log)
df_log_info = pd.DataFrame(log_info)
df_log_info.to_csv(path_log)

print("Saving the model to " + path_model)
clf.save(path_model)
# fm = io.open(path_model, 'wb')
# pkl.dump(clf, fm)
# fm.close()

print("Saving the pred to " + path_pred)
print('Before clf.predict test', int((time.time() - t_start) * 1000))
df_test_pred = pd.DataFrame(clf.predict_proba(Xtest, batch_size=options.batch_size))
df_test_pred.to_csv(path_pred, index=False, header=None) 
print('End', int((time.time() - t_start) * 1000))

