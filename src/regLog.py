import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

print 'Loading data...'
xTrain = pd.read_csv('data/processed/X_TrainSplit_4.csv', index_col=0)
yTrain = pd.read_csv('data/processed/Y_TrainSplit_4.csv', index_col=0)
xVal = pd.read_csv('data/processed/X_ValSplit_4.csv', index_col=0)
yVal = pd.read_csv('data/processed/Y_ValSplit_4.csv', index_col=0)
xTest = pd.read_csv('data/processed/X_test_4.csv', index_col=0)

xTrain.drop('scid_category',inplace=True,axis=1)
xVal.drop('scid_category',inplace=True,axis=1)

print 'Fitting...'
clf = LogisticRegression(class_weight='balanced')
clf.fit(xTrain, yTrain)

clf.score(xVal, yVal)
predTest = clf.predict(xTest)

np.savetxt('data/prediction/regLog_4', predTest)
