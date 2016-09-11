import numpy as np
import pandas as pd
import xgboost as xgb
import scipy as sps
import pickle as pkl
import scipy as sp
import sys
import os
import io
import numpy as np
npa = np.array
from os import listdir
from os.path import isfile, join
from optparse import OptionParser
from  sklearn.linear_model import LogisticRegression
parser = OptionParser()
parser.add_option("--log_folder", default ="logs/xgboost_4")
#parser.add_option("--xtest", default="data/processed/X_test_4.csv")
parser.add_option("--xtest", default="data/processed/X_private_4.csv")
parser.add_option("--k_best", default = 25)
#parser.add_option("--path_pred", default='data/prediction/ensembling/Y_test.predict')
parser.add_option("--path_pred", default='data/prediction/ensembling/Y_private.predict')
options, args = parser.parse_args()
csv_files = [join(options.log_folder, f) for f in listdir(options.log_folder) if isfile(join(options.log_folder, f))]

Xtest  = pd.read_csv(options.xtest)
Xtest.drop(['AffinityCodeId_60.0'], axis=1, inplace=True)
xg_test  = xgb.DMatrix(np.array(Xtest))


def softmax(w, t = 0.002):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


l = []
for f in csv_files :
    l.append(pd.read_csv(f))
df = pd.concat(l, axis=0)
df.sort(['val_score'], inplace=True)
best_models = []

sf = softmax(np.array(-df[0:options.k_best]['val_score']),t=5e-4)

prediction_train  = []
prediction_test   = []
div = np.sum(range(options.k_best+1))


for i in range(options.k_best):
    mf = io.open(df.iloc[i]['path_model'], 'rb')
    model = pkl.load(mf)
    mf.close()
    prediction_test.append( sf[i] *  model.predict(xg_test))
    # if(isinstance(model, xgb.core.Booster)):
        
    # else :
    #     prediction_test.append( sf[i] *  model.predict(Xtest))  
    


prediction_test  = np.array(prediction_test).transpose()
print(prediction_test.shape)

df_test_pred = pd.DataFrame(prediction_test.sum(axis=1))
df_test_pred.to_csv(options.path_pred, index=False, header=None)



