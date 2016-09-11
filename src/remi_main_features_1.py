import numpy as np
import pandas as pd

#from data import utils

def add_dummies(frames, X, col):
    frames.append(pd.get_dummies(X[col], prefix=col))

def extract_features(X, cols_categorical, cols_continuous):
    frames = []

    for col in cols_categorical:
        add_dummies(frames, X, col)

    for col in cols_continuous:
        frames.append(X[col])

    return pd.concat(frames, axis=1)

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
    'SCID',                      # id booker
    'CustomerMD5Key'             # id customer
    'SocioDemographicId',        # big categorical
    'PolicyHolderResidencyArea', # big categorical
    'RatedDriverNumber',         # ask axa
    'CarMakeId',                 # big categorical
    'AffinityCodeId'             # ask axa
]

X_train = pd.read_csv('data/raw/X_train.csv', index_col=0)
X_test  = pd.read_csv('data/raw/X_test.csv',  index_col=0)
Y_train = pd.read_csv('data/raw/Y_train.csv', index_col=0)

if True:
    X_train2 = X_train.loc[0:1000]
    X_test2 = X_test.loc[0:1000]
else:
    X_train2 = X_train
    X_test2  = X_test

    
X_train3 = extract_features(X_train2, cols_categorical, cols_continuous)
X_test3  = extract_features(X_test2, cols_categorical, cols_continuous)

print(np.setdiff1d(X_train3.columns, X_test3.columns))

## centre reduire
def centrer_reduire(X_full, X_train, X_test, cols_continuous):
    for col in cols_continuous:
        mean = X_full[col].mean() * 1.
        std  = X_full[col].std() * 1.
        X_train[col] -= mean
        X_train[col] /= std
        X_test[col] -= mean
        X_test[col] /= std

# X_full = pd.concat([X_train3, X_test3])
# centrer_reduire(X_full, X_train3, X_test3, cols_continuous)


if False:
    X_train3.to_csv('data/interim/X_train_features_1.csv')
    X_test3.to_csv('data/interim/X_test_features_1.csv')