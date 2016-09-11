import numpy as np
import pandas as pd

xTrain = pd.read_csv('data/processed/X_train_4.csv')
xTrain['CustomerMD5Key'] = pd.read_csv('data/raw/X_train.csv', index_col=0)['CustomerMD5Key']
xTrain['Converted'] = pd.read_csv('data/raw/Y_train.csv', index_col=0)['Converted']

xTrain.sort_values(by='CustomerMD5Key', inplace=True)

limit = xTrain.CustomerMD5Key.unique()[np.floor(xTrain.CustomerMD5Key.nunique() * .8)]
print 'Splitting...'
xTrainSplit = xTrain.loc[np.where(xTrain.CustomerMD5Key < limit)]
xValSplit = xTrain.loc[np.where(xTrain.CustomerMD5Key >= limit)]
print 'yTrain...'
yTrainSplit = pd.DataFrame()
yTrainSplit['Converted'] = xTrainSplit['Converted']
xTrainSplit.drop(['Converted','CustomerMD5Key'], axis=1, inplace=True)
yTrainSplit.to_csv('data/processed/Y_TrainSplit_4.csv', index=False)
xTrainSplit.to_csv('data/processed/X_TrainSplit_4.csv', index=False)
print 'yVal...'
yValSplit = pd.DataFrame()
yValSplit['Converted'] = xValSplit['Converted']
xValSplit.drop(['Converted','CustomerMD5Key'], axis=1, inplace=True)
yValSplit.to_csv('data/processed/Y_ValSplit_4.csv', index=False)
xValSplit.to_csv('data/processed/X_ValSplit_4.csv', index=False)
