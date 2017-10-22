import sys
import csv
import numpy as np

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

numOfData = 16281
#numOfData = 32561
numOfDim = 106

X = np.ones(shape = (numOfData,numOfDim+1)) # initialize to 0

paramFile = open('parameters_LR','r')
params = paramFile.read().split()
w = np.array(params).reshape((numOfDim+1,1)).astype(np.float32)

print('Reading File...')
num = 0
with open(TestFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split()[0].split(',')
		X[num][1:] = np.array(lineList)
		num += 1
		
X[:,1:] = Normalize(X[:,1:]);

print('Testing and Writing to File...')

y_pred = 1/(1+np.exp(-np.dot(X,w)))

y_discrete = np.ones(shape = (numOfData,1))
y_discrete[y_pred <= 0.5] = 0

# writing output csv 
with open(outFilePath, 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['id,label'])
	for i in range(numOfData):
		row_list = [','.join([str(i+1),str(int(y_discrete[i][0]))])]
		spamwriter.writerow(row_list)