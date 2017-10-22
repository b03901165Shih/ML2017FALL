import sys
import numpy as np
import xgboost as xgb

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def Loss(y,y_pred):
	accurate = (np.abs(y-y_pred)<0.5).sum()
	return truncate(accurate/y.shape[0],5)

def Normalize(X):
	mu = np.sum(X ,axis = 0) ; mu = mu/X.shape[0]
	sigma = np.sqrt(np.sum(np.square(X-mu),axis = 0)/X.shape[0])+0.000001;
	X = (X-mu)/sigma;
	return X


np.set_printoptions(suppress=True)
TrainFilePath = (sys.argv)[1]
TrainLabelFilePath = (sys.argv)[2]

numOfData = 32561
numOfDim = 106

X = np.ones(shape = (numOfData,numOfDim)) # initialize to 0
y = np.zeros(shape = (numOfData,1))


print('Reading File...')
num = 0
with open(TrainFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split()[0].split(',')
		X[num][:] = np.array(lineList)
		num += 1

X = Normalize(X);


num = 0
with open(TrainLabelFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split('\n')[0]
		y[num] = np.array(lineList)
		num += 1

xg_train = xgb.DMatrix(X, label=y)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logistic'#'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 1

watchlist = [(xg_train, 'train')]
num_round = 100
#bst = xgb.train(param, xg_train, num_round, watchlist,early_stopping_rounds = 5)

# For Cross validation
res = xgb.cv(param, xg_train, num_round, nfold=3, callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(5)])
best_nrounds = res.shape[0]

bst = xgb.train(param, xg_train, best_nrounds, watchlist)#,early_stopping_rounds = 5
bst.save_model('xgb_model')