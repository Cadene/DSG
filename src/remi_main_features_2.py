import numpy as np
import pandas as pd
import time

#from data import utils

def add_dummies(frames, X, col):
    frames.append(pd.get_dummies(X[col], prefix=col))

def centrer_reduire(X_full, X_train, X_test, cols_continuous):
    for col in cols_continuous:
        mean = X_full[col].mean() * 1.
        std  = X_full[col].std() * 1.
        X_train[col] -= mean
        X_train[col] /= std
        X_test[col] -= mean
        X_test[col] /= std


def add_final_features(X, dfKey, cols):
    new_features = pd.DataFrame()

    mean_std = X.groupby(dfKey.CustomerMD5Key).std().mean(axis=1)
    new_features['mean_std_all_features'] = dfKey.CustomerMD5Key.apply(lambda x : mean_std[x])

    max_std = X.groupby(dfKey.CustomerMD5Key).std().max(axis=1)
    new_features['max_std_all_features'] = dfKey.CustomerMD5Key.apply(lambda x : max_std[x])

    new_features['scid_category'] = dfKey.SCID.apply(lambda x : str(x)[0])

    moyenne = X.groupby(dfKey.CustomerMD5Key).mean()
    ecart = X.groupby(dfKey.CustomerMD5Key).std()
    ecart = ecart.fillna(0)

    for col in cols:
        new_features['mean_'+col] = dfKey.CustomerMD5Key.apply(lambda x : moyenne[col][x])
        new_features['std_'+col] = dfKey.CustomerMD5Key.apply(lambda x : ecart[col][x])

    countNbQuotesByUser = dfKey.groupby('CustomerMD5Key').count().SCID
    new_features['countByUser'] = dfKey.CustomerMD5Key.apply(lambda x : countNbQuotesByUser[x])

    return new_features

def add_features_broker(X):
    new_broker_features = pd.DataFrame()
    # train
    new_broker_features['mean_converted_broker'] = X.SCID.apply(lambda x : meanConvertedByBroker[str(x)] if x == x and str(x) in meanConvertedByBroker.index else 0)
    new_broker_features['sum_converted_broker'] = X.SCID.apply(lambda x : stdConvertedByBroker[str(x)] if x == x and str(x) in stdConvertedByBroker.index else 0)
    # global
    new_broker_features['countBySCID'] = X.SCID.apply(lambda x: countNbQuotesBySCID[x])
    new_broker_features['nbUsersBySCID'] = X.SCID.apply(lambda x: countNbUsersBySCID[x])
    return new_broker_features

def add_features_global(X):
    new_features = pd.DataFrame()
    new_features['countSocioDemographicId'] = X.SocioDemographicId.apply(lambda x : countSocioDemographicId[x]) 
    return new_features

def add_new_categorical(X):
    frames = []
    newCategory_df = pd.DataFrame()
    # Bin Affinity
    newCategory_df['AffinityCodeId'] = X.AffinityCodeId.apply(lambda x: x if countAffinityCodeId[x] > 11 else -1)
    add_dummies(frames, newCategory_df, 'AffinityCodeId')
    return pd.concat(frames, axis=1)

def extract_features(X, cols_categorical, cols_continuous, cols_last_stats):
    frames = []

    for col in cols_categorical:
        add_dummies(frames, X, col)

    for col in cols_continuous:
        frames.append(X[col])

    frames.append(add_features_broker(X))
    frames.append(add_features_global(X))
    frames.append(add_new_categorical(X))

    new_dataframe = pd.concat(frames, axis=1)
    
    frames = [new_dataframe]
    frames.append(add_final_features(new_dataframe, X, cols_last_stats)) # X = keys 

    return pd.concat(frames, axis=1)

# cols = cols_drop + categorical + continuous + left
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

# cols to extract features customer (after centrer_reduire)
cols_last_stats = cols_continuous + [ 'countSocioDemographicId' ] #numerized


t_start = time.time()

X_train = pd.read_csv('data/raw/X_train.csv', index_col=0)
X_test  = pd.read_csv('data/raw/X_test.csv',  index_col=0)
Y_train = pd.read_csv('data/raw/Y_train.csv', index_col=0)

if False:
    X_train2 = X_train.loc[0:1000]
    X_test2 = X_test.loc[0:1000]
    Y_train2 = Y_train.loc[0:1000]
else:
    X_train2 = X_train
    X_test2  = X_test
    Y_train2 = Y_train

# warnin work with X_train2

X_train2 = pd.concat([X_train2, Y_train2], axis=1, join='inner')
X_train2.SCID.fillna('NULL',inplace=True)

X_full2 = pd.concat([X_train2, X_test2])
#X_full = pd.concat([X_train2, X_test2, X_private2])
centrer_reduire(X_full2, X_train2, X_test2, cols_continuous)

# globals for train features
meanConvertedByBroker = X_train2.groupby(X_train2.SCID).mean().Converted
stdConvertedByBroker  = X_train2.groupby(X_train2.SCID).sum().Converted

# globals for global features
countNbQuotesBySCID = X_full2.groupby('SCID').count().CustomerMD5Key
countNbUsersBySCID = X_full2.groupby('SCID').CustomerMD5Key.nunique()
countSocioDemographicId = X_full2.groupby('SocioDemographicId').count().CustomerMD5Key
countAffinityCodeId = X_full2.groupby('AffinityCodeId').count().CustomerMD5Key    

print('before X_train3', int((time.time() - t_start) * 1000))
X_train3 = extract_features(X_train2, cols_categorical, cols_continuous, cols_last_stats)
print('before X_test3', int((time.time() - t_start) * 1000))
X_test3  = extract_features(X_test2, cols_categorical, cols_continuous, cols_last_stats)
print('after X_test3', int((time.time() - t_start) * 1000))

print(np.setdiff1d(X_train3.columns, X_test3.columns))
print(len(X_train3.columns))
print(len(X_test3.columns))

if True:
    X_train3.to_csv('data/interim/X_train_features_2.csv')
    X_test3.to_csv('data/interim/X_test_features_2.csv')