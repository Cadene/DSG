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

parser.add_option("--k_best", default = 10)
parser.add_option("--path_pred", default='data/prediction/ensembling/logistic_regression.csv')
options, args = parser.parse_args()
csv_files = [join(options.log_folder, f) for f in listdir(options.log_folder) if isfile(join(options.log_folder, f))]

Xtest  = pd.read_csv(options.xtest)
xg_test  = xgb.DMatrix(np.array(Xtest))




Xtrain = pd.read_csv(options.xtrain)  #, index_col=0)
Xtest  = pd.read_csv(options.xtest)   #,index_col=0)
Ytrain = pd.read_csv(options.ytrain)['Converted']

xg_train = xgb.DMatrix(np.array(Xtrain))
xg_test  = xgb.DMatrix(np.array(Xtest))
l = []
for f in csv_files :
    l.append(pd.read_csv(f))
df = pd.concat(l, axis=0)
df.sort(['val_score'], inplace=True)
best_models = []
prediction_train  = []
prediction_test   = []

for i in range(options.k_best):
	mf = io.open(df.iloc[i]['path_model'], 'rb')
	model = pkl.load(mf)
	mf.close()
	best_models.append(df.iloc[i]['path_model'])	
	prediction_train.append(model.predict(xg_train))
	prediction_test.append(model.predict(xg_test))

prediction_train = np.array(prediction_train).transpose()
print(prediction_train.shape)
prediction_test  = np.array(prediction_test).transpose()
print(prediction_test.shape)

lr = LogisticRegression()
lr.fit(prediction_train, Ytrain)
df_test_pred = pd.DataFrame(lr.predict_proba(prediction_test)[:,1])
df_test_pred.to_csv(options.path_pred, index=False, header=None)



