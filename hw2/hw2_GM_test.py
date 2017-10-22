import sys
import csv
import numpy as np

numOfData = 16281
#numOfData = 32561
numOfDim = 106

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def findClass(x):
	x = x.reshape(numOfDim,1)
	C = 1
	if np.log(P[0])-0.5*np.dot(np.dot((x-mu1).transpose(),invSigma),(x-mu1)) > np.log(P[1])-0.5*np.dot(np.dot((x-mu2).transpose(),invSigma),(x-mu2)):
		C = 0
	return C

def Normalize(X):
	mu = np.sum(X ,axis = 0) ; mu = mu/X.shape[0]
	sigma = np.sqrt(np.sum(np.square(X-mu),axis = 0)/X.shape[0])+0.000001;
	X = (X-mu)/sigma;
	return X

np.set_printoptions(suppress=True)
TestFilePath = (sys.argv)[1]
outFilePath = (sys.argv)[2]


X = np.ones(shape = (numOfData,numOfDim)) # initialize to 0

paramFile = open('parameters_GM','r')
P = np.array(paramFile.readline().split()).reshape(2,).astype('float64')
mu1 = np.array(paramFile.readline().split()).reshape(numOfDim,1).astype('float64')
print('mu1 shape:',mu1.shape)
mu2 = np.array(paramFile.readline().split()).reshape(numOfDim,1).astype('float64')
print('mu2 shape:',mu2.shape)
sigma = np.array(paramFile.readline().split()).reshape(numOfDim,numOfDim).astype('float64')
print('sigma shape:',sigma.shape)
invSigma = np.linalg.pinv(sigma)

print('Reading File...')
num = 0
with open(TestFilePath,'r') as inF:
	inF.readline()
	for line in inF:
		lineList = line.split()[0].split(',')
		X[num][:] = np.array(lineList)
		#X[num][0] /= 100; X[num][1] /= 500000; X[num][3] /= 20000; X[num][4] /= 10000; X[num][5] /= 100
		num += 1

#X[:,1:] = Normalize(X[:,1:]);
print('Testing and Writing to File...')

y_discrete = np.zeros(shape = (numOfData,1))

for i in range(numOfData):
	if i % 5000 == 0:
		print('Evaluate #'+str(i),'Done')
	y_discrete[i] = findClass(X[i])


# writing output csv 
with open(outFilePath, 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['id,label'])
	for i in range(numOfData):
		row_list = [','.join([str(i+1),str(int(y_discrete[i][0]))])]
		spamwriter.writerow(row_list)