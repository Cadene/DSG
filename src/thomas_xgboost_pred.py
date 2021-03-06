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
#parser.add_option("--xtest", default="data/processed/X_test_4.csv")
parser.add_option("--xtest", default="data/processed/X_private_4.csv")
parser.add_option("--pathModel", default="models/one_xgboost/model_291754595.8_9837.pkl")
#parser.add_option("--pathPred", default="data/prediction/one_xgboost/Y_test.predict")
parser.add_option("--pathPred", default="data/prediction/one_xgboost/Y_private.predict")

options, args = parser.parse_args()

Xtest  = pd.read_csv(options.xtest)
Xtest.drop(['AffinityCodeId_60.0'], axis=1, inplace=True)
#Xtest.drop('scid_category_N', axis=1, inplace=True)

xg_test  = xgb.DMatrix(np.array(Xtest))

#Loading the model
model_file = io.open(options.pathModel,'rb')
model = pkl.load(model_file)
model_file.close()

print("Saving the pred to " + options.pathPred)
df_test_pred = pd.DataFrame(model.predict(xg_test))
df_test_pred.to_csv(options.pathPred, index=False, header=None)

