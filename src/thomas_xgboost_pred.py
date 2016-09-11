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
parser.add_option("--xtest", default="data/processed/X_test_4.csv")
parser.add_option("--pathModel", default="models/xgboost_4/model_291621024.6_52.pkl")
parser.add_option("--pathPred", default="data/prediction/remi_Y_test.predict")

options, args = parser.parse_args()

#Loading the model
model_file = io.open(options.pathModel,'rb')
model = pkl.load(model_file)
model_file.close()

Xtest  = pd.read_csv(options.xtest)
xg_test  = xgb.DMatrix(np.array(Xtest))


print("Saving the pred to " + options.pathPred)
df_test_pred = pd.DataFrame(model.predict(xg_test))
df_test_pred.to_csv(options.pathPred, index=False, header=None)

