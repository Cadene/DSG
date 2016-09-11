import numpy as np
import pandas as pd

xTest = pd.read_csv('data/raw/X_test.csv', index_col=0)
pred = np.loadtxt('data/predict/tweak_Y_pred.csv', header=)

df = pd.DataFrame()
df['Converted'] = pred
df['CustomerMD5Key']= xTest.CustomerMD5Key
df['index'] = pred.index

df.sort_values(by='CustomerMD5Key')

