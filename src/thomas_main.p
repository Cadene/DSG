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
parser.add_option("-x", "--xtrain", dest="xtrain",action="store",default="data/processed/X_TrainSplit_4.csv",
                  help="train file")
parser.add_option("-y", "--ytrain", dest="ytrain",action="store",default="data/processed/Y_TrainSplit_4.csv",
                  help="label file")
parser.add_option("--xv", "--xval", dest="xval",action="store",default="data/processed/X_ValSplit_4.csv",
                  help="train file")
parser.add_option("--yv", "--yval", dest="yval",action="store",default="data/processed/Y_ValSplit_4.csv",
                  help="label file")
parser.add_option("-e", "--xtest",
                  action="store", dest="xtest",default="data/processed/X_test_4.csv",
                  help="test file")
parser.add_option("-p","--save",
                  action="store", dest="save",
                  help="store prediction file", default="data/prediction/default_prediction.csv")
parser.add_option("--eta",type = "float",dest='eta',help="eq to learning rate", default=0.2)
parser.add_option("--mc",dest="mc",type="int",help="min child", default=1)
parser.add_option("-d","--max_depth",type="int", dest='max_depth', default=3)


parser.add_option("-v", type='float',dest='v', help='percent validation set',default=1.)
parser.add_option('--cv', action="store",type="int", help="perform a inform nb fold" , dest='cv', default=0)
parser.add_option('-s','--seed',type="int", dest="seed", help="seed int", default=0)
parser.add_option("-m",'--savemodel', dest="sm", help="path to save the model")
parser.add_option("-a",'--alpha', type="float", dest="alpha", help="alpha for L1 regularisation", default=0.)
parser.add_option("-l","--lambda", type="float", dest ="lamb", help="lambda fo L2 regularisation", default=0.)

#parser.add_option("--eta",type = "float",dest='eta',help="eq to learning rate", default=0.2)
parser.add_option("--log_folder",dest="log_folder",help="Allowing great minds to store there memory", default='logs/' )
parser.add_option("-n",dest="n_round",type="int",help="epoch number", default=10)

parser.add_option("--subsample",type = "float",dest='subsample',help="eq to learning rate", default=0.8)
parser.add_option("--colsample_bytree",type='float',dest="colsample_bytree",help="Allowing great minds to store there memory", default=0.8 )
parser.add_option("--scale_pos_weight",dest="scale_pos_weight",type="float",help="epoch number", default=1.)


parser.add_option("-g",dest="g",type="float",help="gamma", default=0.1)
parser.add_option("--mds",dest="mds",type="float",help="max_delta_step", default=0.)
options, args = parser.parse_args()


def option_to_dictionary(options):
    #sum_up_option = {'x':[options.xtrain],'eta':[options.eta],'d':[options.max_depth],'log_folder':[options.log_folder],'y':[options.ytrain],'e':[options.xtest],'cv':[options.cv],'s':[options.seed],'a':[options.alpha],'l':[options.lamb],'p':[options.save],'v':[options.v],'d':[options.max_depth]};
    sum_up_option = vars(options)
    return sum_up_option;

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def save_log(df):
    int_key = np.random.randint(1,10000)
    date_key = os.times()[4] * 10
    log_path = os.path.join(options.log_folder,"log_"+str(int_key)+"_"+str(date_key)+".csv")
    df.to_csv(log_path)


if(options.xtrain == None or options.ytrain == None):
    sys.stderr.write(str('ERROR : Missing parameters\n'))
    sys.exit()

print(options.xtrain)
param = {}
#XGBOOST parameterss
param['objective'] = 'binary:logistic'
param['eta'] = options.eta
param['max_delta_steps'] = options.mds
param['max_depth'] = options.max_depth
param['silent'] = 1
param['nthread'] = 4
param['eval_metric'] = 'logloss'
param['lambda'] = options.lamb
param['alpha'] = options.alpha
param['gamma'] = options.g
param['subsample'] = options.subsample
param['colsample_bytree'] = options.colsample_bytree
param['scale_pos_weight'] = options.scale_pos_weight
param['min_child_weight'] = options.mc

param['seed'] = options.seed
#param['num_class'] = 6


#Input as pandas datframe




Xtrain = pd.read_csv(options.xtrain, index_col=0)
Ytrain = pd.read_csv(options.ytrain)['Converted']
Xtrain.drop(['scid_category'],axis=1,inplace=True)
Xtrain.fillna(value=0, inplace=True)

Xval = pd.read_csv(options.xval, index_col=0)
Yval = pd.read_csv(options.yval)['Converted']
Xval.drop(['scid_category'],axis=1,inplace=True)
Xval.fillna(value=0, inplace=True)

#n = len(Xdata)
#Xdata = Xdata.set_index('Unnamed: 0')
#Ydata = Ydata.set_index('Unnamed: 0')
#Xtrain = Xdata[0:int(n*options.v)]
#Ytrain  = Ydata[0:int(n*options.v)]['Converted']
#Xval  = Xdata[int(n*options.v):n]
#Yval  = Ydata[int(n*options.v):n]['Converted']

xg_train = xgb.DMatrix(np.array(Xtrain), label=np.array(Ytrain).flatten())
xg_val = xgb.DMatrix(np.array(Xval))
xg_val= False

watchlist = [(xg_train,'train')]
watchlist.append((xg_val, 'val'))

num_round = options.n_round


if(options.cv>0 ):

    res = xgb.cv(param, xg_train, num_round, nfold=options.cv)
    log_info = option_to_dictionary(options)
    log_info['score'] = [np.mean(res)]
    log_info['val_score'] = [np.mean(res)]
    save_log(pd.DataFrame(log_info))
else:

    clf = xgb.train(param, xg_train, num_round, watchlist );

    log_info = option_to_dictionary(options)
    pred = clf.predict(xg_train);
    logloss(Ytrain, pred)
    log_info['score'] = [logloss(Ytrain, pred)]
    
    if xg_val:
	pred = clf.predict(xg_val);
    	logloss(Yval, pred)
    	log_info['val_score'] = [logloss(Yval, pred)]


    if(options.sm != None):

        if(os.path.isdir(options.sm)):
            int_key = np.random.randint(1,10000)
            date_key = os.times()[4] * 10
            options.sm = os.path.join(options.sm,"model_"+str(int_key)+"_"+str(date_key)+".model")
        log_info['model_path'] = options.sm
        print("saving the model to "+options.sm)
        fm = io.open(options.sm,'wb')
        pkl.dump(clf, fm)
        fm.close()

    if(options.xtest != None):
        Xtest = pd.read_csv(options.xtest, index_col=0);
        xg_test = xgb.DMatrix(np.array(Xtest));
        pred = clf.predict(xg_test);
        results = pd.DataFrame(pred);
        if(os.path.isdir(options.save)):
            int_key = np.random.randint(1,10000)
            date_key = os.times()[4] * 10
            options.save = os.path.join(options.save,"pred_"+str(int_key)+"_"+str(date_key)+".model")

        log_info['prediction_path'] = options.save
        results.to_csv(options.save, index=False, header = None)
    df = pd.DataFrame(log_info)
    save_log(df)
# get prediction






#sprint('logloss = '+str(logloss(Ytest,pred)))
