import sys
import numpy as np

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def Loss(y,y_pred):
	loss = -np.dot(y.transpose(),np.log(y_pred))-np.dot(((1-y).transpose()),np.log(1-y_pred))
	accurate = (np.abs(y-y_pred)<0.5).sum()
	return (truncate(loss[0][0]/y.shape[0],5),truncate(accurate/y.shape[0],5))

def Normalize(X):
	mu = np.sum(X ,axis = 0) ; mu = mu/X.shape[0]
	sigma = np.sqrt(np.sum(np.square(X-mu),axis = 0)/X.shape[0])+0.000001;
	X = (X-mu)/sigma;
	return X

np.set_printoptions(suppress=True)
TrainFilePath = (sys.argv)[1]
LabelFilePath = (sys.argv)[2]

#numOfData = 16281
numOfData = 32561
numOfDim = 106

X = np.ones(shape = (numOfData,numOfDim+1)) # initialize to 0
w = np.zeros(shape = (numOfDim+1,1))		# initialize to 0.5
y = np.zeros(shape = (numOfData,1))

print('Reading File...')
num = 0
with open(TrainFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split()[0].split(',')
		X[num][1:] = np.array(lineList)
		num += 1

X[:,1:] = Normalize(X[:,1:]);

#X = np.concatenate((X[:,0:4],X[:,5:]),axis = 1); numOfDim = X.shape[1]-1
#w = np.zeros(shape = (numOfDim+1,1))

num = 0
with open(LabelFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split('\n')[0]
		y[num][:] = np.array(lineList)
		num += 1

print('Training...')

# Parameters for GD
numOfIter = 2050
ita = 0.1
regulate = 10;

# Gradient Descent For Training
w_grad_norm = np.zeros(shape = w.shape)  #initialize to 0
w_grad_norm += 0.000001

for i in range(numOfIter):
	y_pred = 1/(1+np.exp(-np.dot(X,w)))
	if i % 200 == 0:
		print('Iteration #'+str(i),'Done, (loss, accuracy):',Loss(y,y_pred))
	w_grad = -np.dot(X.transpose(),y-y_pred)+np.dot(regulate*np.identity(X.shape[1]),np.sign(w))
	w_grad_norm += np.square(w_grad)
	w -= ita*w_grad/np.sqrt(w_grad_norm)

print('Writing Parameters...')

# Writing parameters 
with open('parameters_LR','w') as outFile:
	for i in range(numOfDim+1):
		outFile.write(str(w[i][0])+' ')