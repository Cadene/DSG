import numpy as np
import pandas as pd
import xgboost as xgb
import scipy as sps
import pickle as pkl
import scipy as sp
import sys
import os
import io
from os import listdir
from os.path import isfile, join
from optparse import OptionParser
from  sklearn.linear_model import LogisticRegression
parser = OptionParser()
parser.add_option("--log_folder", default ="logs/selected_models")
parser.add_option("--xtrain", default="data/processed/X_train_4.csv")
parser.add_option("--ytrain", default="data/processed/Y_train.csv")
parser.add_option("--xtest", default="data/processed/X_test_4.csv")

parser.add_option("--k_best", default = 5)
parser.add_option("--path_pred", default='data/prediction/ensembling/mean.csv')
options, args = parser.parse_args()
csv_files = [join(options.log_folder, f) for f in listdir(options.log_folder) if isfile(join(options.log_folder, f))]

Xtest  = pd.read_csv(options.xtest)
xg_test  = xgb.DMatrix(np.array(Xtest))




def softmax(w, t = 1.0):
	e = np.exp(npa(w) / t)
	dist = e / np.sum(e)
return dist


Xtrain = pd.read_csv(options.xtrain)  #, index_col=0)
Xtest  = pd.read_csv(options.xtest)   #,index_col=0)
Ytrain = pd.read_csv(options.ytrain)['Converted']



l = []
for f in csv_files :
    l.append(pd.read_csv(f))
df = pd.concat(l, axis=0)
df.sort(['val_score'], inplace=True)
best_models = []

df[0:options.k_best]['val_score']

prediction_train  = []
prediction_test   = []
div = np.sum(range(options.k_best+1))
for i in range(options.k_best):
	mf = io.open(df.iloc[i]['path_model'], 'rb')
	model = pkl.load(mf)
	mf.close()
	if(isinstance(model, xgb.core.Booster)):
		prediction_test.append(((options.k_best - i)/float(div)) *  model.predict(xg_test))
	else :
		prediction_test.append(((options.k_best - i)/float(div)) *  model.predict(Xtest))	
	


prediction_test  = np.array(prediction_test).transpose()
print(prediction_test.shape)

df_test_pred = pd.DataFrame(prediction_test.mean(axis=1))
df_test_pred.to_csv(options.path_pred, index=False, header=None)



