import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

print 'Loading data...'

xVal = pd.read_csv('data/processed/X_ValSplit_4.csv')
print xVal.columns
yVal = pd.read_csv('data/processed/Y_ValSplit_4.csv')
yVal = yVal['Converted']


xTrain = pd.read_csv('data/processed/X_TrainSplit_4.csv')
print xTrain.columns
yTrain = pd.read_csv('data/processed/Y_TrainSplit_4.csv')
yTrain = yTrain['Converted']

xTest = pd.read_csv('data/processed/X_test_4.csv')
print xTest.columns

print 'Fitting...'
clf = LogisticRegression(class_weight='balanced')
clf.fit(xTrain, yTrain)

clf.score(xVal, yVal)
predTest = clf.predict(xTest)

np.savetxt('data/prediction/regLog_4', predTest)
