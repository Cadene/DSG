import numpy as np
import pandas as pd
import time

#from data import utils

hasPrivate = True

def add_dummies(frames, X, col):
    frames.append(pd.get_dummies(X[col], prefix=col))

def centrer_reduire(X_full, X_train, X_test, X_private, cols_continuous):
    for col in cols_continuous:
        mean = X_full[col].mean() * 1.
        std  = X_full[col].std() * 1.
        if std == 0:
            X_train.drop(col, axis=1, inplace=True)
            X_test.drop(col, axis=1, inplace=True)
            if hasPrivate:
                X_private.drop(col, axis=1, inplace=True)
        else:
            X_train[col] -= mean
            X_train[col] /= std
            X_test[col] -= mean
            X_test[col] /= std
            if hasPrivate:
                X_private[col] -= mean
                X_private[col] /= std
        

def add_final_features(X, dfKey, cols):
    new_features = pd.DataFrame()

    X['CustomerMD5Key'] = dfKey.CustomerMD5Key
    X['SCID'] = dfKey.SCID
    
    mean_std = X.groupby('CustomerMD5Key').std().mean(axis=1)
    mean_std.fillna(0,inplace=True)
    new_features['mean_std_all_features'] = X.CustomerMD5Key.apply(lambda x : mean_std[x])

    max_std = X.groupby('CustomerMD5Key').std().max(axis=1)
    max_std.fillna(0,inplace=True)
    new_features['max_std_all_features'] = X.CustomerMD5Key.apply(lambda x : max_std[x])

    # CustomerKey intersect == null
    moyenneByCustomer = X.groupby('CustomerMD5Key').mean()
    ecartByCustomer = X.groupby('CustomerMD5Key').std()
    ecartByCustomer = ecartByCustomer.fillna(0)

    for col in cols:
        new_features['mean_customer_'+col] = X.CustomerMD5Key.apply(lambda x : moyenneByCustomer[col][x])
        new_features['std_customer_'+col] = X.CustomerMD5Key.apply(lambda x : ecartByCustomer[col][x])
        new_features['mean_broker_'+col] = X.SCID.apply(lambda x : moyenneBySCID[col][x] if x == x else 0)
        new_features['std_broker_'+col] = X.SCID.apply(lambda x : ecartBySCID[col][x] if x == x else 0)
    
    countNbQuotesByUser = X.groupby('CustomerMD5Key').count().SCID
    new_features['countByUser'] = X.CustomerMD5Key.apply(lambda x : countNbQuotesByUser[x])

    X.drop(['CustomerMD5Key', 'SCID'], axis=1, inplace=True)

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
    # Category of SCID
    newCategory_df['scid_category'] = X.SCID.apply(lambda x : str(x)[0])
    add_dummies(frames, newCategory_df, 'scid_category')
    return pd.concat(frames, axis=1)

def extract_features_1(X, cols_categorical, cols_continuous):
    frames = []

    for col in cols_categorical:
        add_dummies(frames, X, col)

    for col in cols_continuous:
        frames.append(X[col])

    frames.append(add_features_broker(X))
    frames.append(add_features_global(X))
    frames.append(add_new_categorical(X))

    return pd.concat(frames, axis=1)

def extract_features_2(X, dfKey, cols_2):
    frames = [X]
    frames.append(add_final_features(X, dfKey, cols_2))
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
cols_2 = cols_continuous + [ 'countSocioDemographicId' ] #numerized

t_start = time.time()

X_train = pd.read_csv('data/raw/X_train.csv', index_col=0)
X_test  = pd.read_csv('data/raw/X_test.csv',  index_col=0)
if hasPrivate:
    #X_private = pd.read_csv('data/raw/X_test.csv',  index_col=0)
    X_private = pd.read_csv('data/raw/X_private.csv',  index_col=0)
Y_train = pd.read_csv('data/raw/Y_train.csv', index_col=0)

if False:
    X_train2 = X_train.loc[0:1000]
    X_test2 = X_test.loc[0:1000]
    if hasPrivate:
        X_private2 = X_private.loc[0:1000]
    Y_train2 = Y_train.loc[0:1000]
else:
    X_train2 = X_train
    X_test2  = X_test
    if hasPrivate:
        X_private2 = X_private
    Y_train2 = Y_train

# warnin work with X_train2

X_train2 = pd.concat([X_train2, Y_train2], axis=1, join='inner')
X_train2.SCID.fillna('NULL',inplace=True)

if hasPrivate:
    X_full2 = pd.concat([X_train2, X_test2, X_private2])
    centrer_reduire(X_full2, X_train2, X_test2, X_private2, cols_continuous)
else:
    X_full2 = pd.concat([X_train2, X_test2])
    centrer_reduire(X_full2, X_train2, X_test2, False, cols_continuous)

# globals for train features
meanConvertedByBroker = X_train2.groupby(X_train2.SCID).mean().Converted
stdConvertedByBroker  = X_train2.groupby(X_train2.SCID).sum().Converted

# globals for global features
countNbQuotesBySCID = X_full2.groupby('SCID').count().CustomerMD5Key
countNbUsersBySCID = X_full2.groupby('SCID').CustomerMD5Key.nunique()
countSocioDemographicId = X_full2.groupby('SocioDemographicId').count().CustomerMD5Key
countAffinityCodeId = X_full2.groupby('AffinityCodeId').count().CustomerMD5Key    

print('before X_train3', int((time.time() - t_start) * 1000))
X_train3 = extract_features_1(X_train2, cols_categorical, cols_continuous)
print('before X_test3', int((time.time() - t_start) * 1000))
X_test3  = extract_features_1(X_test2, cols_categorical, cols_continuous)
if hasPrivate:
    print('before X_private3', int((time.time() - t_start) * 1000))
    X_private3  = extract_features_1(X_private2, cols_categorical, cols_continuous)
print('after X3', int((time.time() - t_start) * 1000))

if hasPrivate:
    X_full3 = pd.concat([X_train3, X_test3, X_private3])
else:
    X_full3 = pd.concat([X_train3, X_test3])

X_full3['SCID'] = X_full2.SCID
moyenneBySCID = X_full3.groupby('SCID').mean()
ecartBySCID = X_full3.groupby('SCID').std()
ecartBySCID = ecartBySCID.fillna(0)
X_full3.drop('SCID', axis=1, inplace=True)

if hasPrivate:
    print('before X_private4', int((time.time() - t_start) * 1000))
    X_private4  = extract_features_2(X_private3, X_private2, cols_2)
else:
    print('before X_train4', int((time.time() - t_start) * 1000))
    X_train4 = extract_features_2(X_train3, X_train2, cols_2)
    print('before X_test4', int((time.time() - t_start) * 1000))
    X_test4  = extract_features_2(X_test3, X_test2, cols_2)
print('after X4', int((time.time() - t_start) * 1000))

if hasPrivate:
    print(np.setdiff1d(X_train4.columns, X_private4.columns))
    print(len(X_private4.columns))
else:
    print(np.setdiff1d(X_train4.columns, X_test4.columns))
    print(len(X_train4.columns))
    print(len(X_test4.columns))

if True:
    if hasPrivate:
        X_private4.to_csv('data/processed/X_private_4.csv', index=False)
    else:
        X_train4.drop('scid_category_N', axis=1, inplace=True)
        X_train4.to_csv('data/processed/X_train_4.csv', index=False)
        X_test4.to_csv('data/processed/X_test_4.csv', index=False)

