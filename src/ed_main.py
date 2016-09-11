import numpy as np
import pandas as pd

#from data import utils

def add_dummies(frames, X, col):
    frames.append(pd.get_dummies(X[col], prefix=col))

X_train = pd.read_csv('data/raw/X_train.csv', index_col=0)
X_test  = pd.read_csv('data/raw/X_test.csv',  index_col=0)
Y_train = pd.read_csv('data/raw/Y_train.csv', index_col=0)

if True:
    X_train2 = X_train.loc[0:300]
    X_test2 = X_test.loc[0:300]
else:
    X_train2 = X_train
    X_test2  = X_test

cols_drop = [
    #time1
    #time2
    'SelectedPackage',
    'FirstDriverDrivingLicenseNumberY'
]
cols_categorical = [
    'FirstDriverMaritalStatus',
    'CarFuelId',
    'CarUsageId',
    'CarParkingTypeId',
    'FirstDriverDrivingLicenceType',
    'CoverIsNoClaimDiscountSelected', #binary
    'CarDrivingEntitlement',
    'CarTransmissionId',
    'IsPolicyholderAHomeowner',
    'NameOfPolicyProduct'
] 
cols_continuous = [
    'CarAnnualMileage',
    'FirstDriverAge',
    'CarInsuredValue',
    'CarAge',
    'VoluntaryExcess',
    'PolicyHolderNoClaimDiscountYears',
    'AllDriversNbConvictions',
    'DaysSinceCarPurchase'
]
cols_left = [
    'SCID',
    'CustomerMD5Key'
    'SocioDemographicId', # big categorical
    'PolicyHolderResidencyArea', # big categorical
    'RatedDriverNumber', # numerical + binary
    'CarMakeId', # big categorical
    'AffinityCodeId' # bin < 11 categorical
]

def extract_features(X, cols_categorical, cols_continuous):
    frames = []
    for col in cols_categorical:
        add_dummies(frames, X, col)
    for col in cols_continuous:
        frames.append(X[col])
    countNbQuotesByUser = X.groupby('CustomerMD5Key').count().SCID
    countNbQuoteBySCID = X.groupby('SCID').count().CustomerMD5Key
    countNbUserBySCID = X.groupby('SCID').CustomerMD5Key.nunique()
    

    return pd.concat(frames, axis=1)

def extract_counts(X):
    #X.group
    return
    
X_train3 = extract_features(X_train2, cols_categorical, cols_continuous)
X_test3  = extract_features(X_test2, cols_categorical, cols_continuous)


#X_train3.to_csv('data/interim/X_train_features_1.csv')
