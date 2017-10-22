import sys
import numpy as np

#numOfData = 16281
numOfData = 32561
numOfDim = 106
# Two Gaussian for two class
P_C1 = 0 ; P_C2 = 0
mu1 = 0 ; mu2 = 0
sigma1 = np.zeros(shape = (numOfDim,numOfDim)) ; sigma2 = np.zeros(shape = (numOfDim,numOfDim)) 
invSigma = 0;

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

def findClass(x):
	x = x.reshape(numOfDim,1)
	C = 1
	if np.log(P_C1)-0.5*np.dot(np.dot((x-mu1).transpose(),invSigma),(x-mu1)) > np.log(P_C2)-0.5*np.dot(np.dot((x-mu2).transpose(),invSigma),(x-mu2)):
		C = 0
	return C

def Normalize(X):
	mu = np.sum(X ,axis = 0) ; mu = mu/X.shape[0]
	sigma = np.sqrt(np.sum(np.square(X-mu),axis = 0)/X.shape[0])+0.000001;
	X = (X-mu)/sigma;
	return X


np.set_printoptions(suppress=True)
TrainFilePath = (sys.argv)[1]
LabelFilePath = (sys.argv)[2]


X = np.ones(shape = (numOfData,numOfDim)) 	# initialize to 0
y = np.zeros(shape = (numOfData,1))
y_pred = np.zeros(shape = (numOfData,1))

print('Reading File...')
num = 0
with open(TrainFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split()[0].split(',')
		X[num][:] = np.array(lineList)
		#X[num][0] /= 100; X[num][1] /= 500000; X[num][3] /= 20000; X[num][4] /= 10000; X[num][5] /= 100
		num += 1

#X[:,1:] = Normalize(X[:,1:]);

num = 0
with open(LabelFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split('\n')[0]
		y[num][:] = np.array(lineList)
		num += 1

print('Training...')

X1 = X[(y < 0.5).reshape(numOfData,)][:] ; y1 = y[y < 0.5] ; y1 = y1.reshape(y1.shape[0],1)
X2 = X[(y > 0.5).reshape(numOfData,)][:] ; y2 = y[y > 0.5] ; y2 = y2.reshape(y2.shape[0],1)
print('X1 shape = ',X1.shape)
print('X2 shape = ',X2.shape)

P_C1 = X1.shape[0]/numOfData
P_C2 = X2.shape[0]/numOfData

mu1 = np.sum(X1 ,axis = 0) ; mu1 = mu1.reshape(mu1.shape[0],1)/X1.shape[0]
mu2 = np.sum(X2 ,axis = 0) ; mu2 = mu2.reshape(mu2.shape[0],1)/X2.shape[0]
print('Mu1 shape = ',mu1.shape)
print('Mu2 shape = ',mu2.shape)

for i in range(X1.shape[0]):
	sigma1 += np.dot((X1[i]-mu1.transpose()).transpose(),(X1[i]-mu1.transpose()))/X1.shape[0]

for i in range(X2.shape[0]):
	sigma2 += np.dot((X2[i]-mu2.transpose()).transpose(),(X2[i]-mu2.transpose()))/X2.shape[0]

'''
sigma1_inde = np.zeros(shape = (numOfDim,)); sigma2_inde = np.zeros(shape = (numOfDim,))

for i in range(numOfDim):
	sigma1_inde[i] = np.dot((X[:,i]-mu1[i,0]).transpose(),(X[:,i]-mu1[i,0]))/X1.shape[0]

for i in range(numOfDim):
	sigma2_inde[i] = np.dot((X[:,i]-mu2[i,0]).transpose(),(X[:,i]-mu2[i,0]))/X2.shape[0]

sigma1 = np.diag(sigma1_inde)
sigma2 = np.diag(sigma2_inde)
'''

sigma = (sigma1*P_C1+sigma2*P_C2)
invSigma = np.linalg.pinv(sigma)

#np.linalg.det()

for i in range(numOfData):
	if i % 10000 == 0:
		print('Evaluate #'+str(i),'Done')
	y_pred[i] = findClass(X[i])

print('Accuracy = ', Loss(y,y_pred))

# Writing parameters 
with open('parameters_GM','w') as outFile:
	outFile.write(str(P_C1)+' ')
	outFile.write(str(P_C2)+' ')
	outFile.write('\n')
	for i in range(numOfDim):
		outFile.write(str(mu1[i][0])+' ')
	outFile.write('\n')
	for i in range(numOfDim):
		outFile.write(str(mu2[i][0])+' ')
	outFile.write('\n')
	for i in range(numOfDim):
		for j in range(numOfDim):
			outFile.write(str(sigma[i][j])+' ')