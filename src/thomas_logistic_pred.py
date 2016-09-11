import numpy as np
import pandas as pd
import scipy as sps
import pickle as pkl
import sys
import os
import io
from optparse import OptionParser
import scipy as sp

from sklearn.linear_model import LogisticRegression

parser = OptionParser()
parser.add_option("--xtest",     default="data/processed/X_test_4.csv")
parser.add_option("--pathModel", default="models/logistic/model.pkl")
parser.add_option("--pathPred",  default="data/prediction/Y_test.predict")

options, args = parser.parse_args()

print("Loading model from " + options.pathModel) 
model_file = io.open(options.pathModel,'rb')
model = pkl.load(model_file)
model_file.close()

print("Loading data from " + options.pathPred)
Xtest = pd.read_csv(options.xtest)

print("Saving pred to " + options.pathPred)
df_test_pred = pd.DataFrame(model.predict_proba(Xtest)[:,1])
df_test_pred.to_csv(options.pathPred, index=False, header=None)

