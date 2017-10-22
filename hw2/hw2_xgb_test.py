import sys
import csv
import numpy as np
import xgboost as xgb

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def Normalize(X):
	mu = np.sum(X ,axis = 0) ; mu = mu/X.shape[0]
	sigma = np.sqrt(np.sum(np.square(X-mu),axis = 0)/X.shape[0])+0.000001;
	X = (X-mu)/sigma;
	return X


np.set_printoptions(suppress=True)
TestFilePath = (sys.argv)[1]
outFilePath = (sys.argv)[2]

numOfTestData = 16281
numOfDim = 106

X_test = np.ones(shape = (numOfTestData,numOfDim)) # initialize to 0

num = 0
with open(TestFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split()[0].split(',')
		X_test[num][:] = np.array(lineList)
		num += 1

X_test = Normalize(X_test);		

xg_test = xgb.DMatrix(X_test)

bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('xgb_model')  # load data

# get prediction
y_pred = bst.predict(xg_test).reshape(numOfTestData,1)
y_pred = np.round(y_pred)

print('Writing File...')

# writing output csv 
with open(outFilePath, 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['id,label'])
	for i in range(numOfTestData):
		row_list = [','.join([str(i+1),str(int(y_pred[i][0]))])]
		spamwriter.writerow(row_list)