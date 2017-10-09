##################################
#	AMB_TEMP	0
#	CH4			1
#	CO 			2
#	NMHC		3
#	NO 			4
#	NO2 		5
#	NOx 		6
#	O3 			7
#	PM10 		8
#	PM2.5 		9
#	RAINFALL 	10
#	RH 			11
#	SO2 		12
#	THC 		13
#	WD_HR 		14
#	WIND_DIREC 	15
#	WIND_SPEED 	16
#	WS_HR 		17
##################################

import sys
import csv
import numpy as np

np.set_printoptions(suppress=True)
inFilePath = (sys.argv)[1]
data = []

# parameters
numOfType = 18
numOfFeature = 9	# last numOfFeature hours of pm2.5 (== m)
offset  = 1			# offsere of the input file

# defining type of attribute for training
param_list = [2,7,9,14,16]
#param_list = [2,7,8,9,14]
#param_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

# linear(1) or quadratic(2) model
model_order = 2

# regulataion parameter lambda
regulate = 100

# reading csv to data
with open(inFilePath, 'rt',  encoding = 'Big5') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spamreader:
		rowList = row[0].split(',')
		rowList = ['0' if x == 'NR' else x for x in rowList]
		data.append(rowList)

# first dimension: a total of 18 attribute per day, 10th attribute is pm2.5
# second dimension: date, location, feature(ex:PM2.5), 0~23 hours
print("Dimension of data = ("+str(len(data))+","+str(len(data[0]))+")")

numOfDays = (len(data)-offset)//numOfType
numOfData = (24-numOfFeature)*(numOfDays) 	# == n
print("Num of Data = ", numOfData)

# Error is defined as ||y-C*a||^2
# where y: gt result (n*1) ; C : input data matrix (n*(m+1)) ; a : weights ((m+1)*1)
y = np.zeros(shape = (numOfData,1))
C = np.ones(shape = (numOfData, model_order*len(param_list)*numOfFeature+1))
a = np.zeros(shape = (model_order*len(param_list)*numOfFeature+1,1))  #initialize to 0

# Storing data in C and y
start = -1+offset ; end = -1+offset
index = 0
for i in range(numOfDays):
	start = end + 1
	end = end + (len(data)-offset)//numOfDays
	dayData = data[start:end+1]
	PM25 = dayData[9][3:] # dimension = 24
	for j in range(24-numOfFeature):
		for k in range(len(param_list)):
			feature_data = dayData[param_list[k]][3:] # dimension = 24
			C[index][k*numOfFeature+1:(k+1)*numOfFeature+1] = np.array(feature_data[j:j+numOfFeature]).reshape((1,numOfFeature)).astype(np.float64)
		if model_order == 2:
			for k in range(len(param_list),2*len(param_list)):
				feature_data = dayData[param_list[k-len(param_list)]][3:] # dimension = 24
				C[index][k*numOfFeature+1:(k+1)*numOfFeature+1] = np.square(np.array(feature_data[j:j+numOfFeature]).reshape((1,numOfFeature)).astype(np.float64))
		y[index] = np.array(PM25[j+numOfFeature]).astype(np.float64)
		#print(PM25[j:j+numOfFeature])
		#print(PM25[j+numOfFeature])
		#input()
		index += 1

# Parameters for GD
numOfIter = 700000
ita = 1

# Gradient Descent For Training
a_grad_norm = np.zeros(shape = (model_order*len(param_list)*numOfFeature+1,1))  #initialize to 0
for i in range(numOfIter):
	if i % 1000 == 0:
		error_tmp = np.sqrt(np.dot((y-np.dot(C,a)).transpose(),(y-np.dot(C,a)))/y.shape[0])
		print('Iteration #'+str(i)+' Done...', 'Current loss = ',str(error_tmp[0][0]))
	# Computing Gradient in vector form (all features are considered in the same time)
	a_grad = 2*(np.dot(np.dot(C.transpose(),C)+regulate*np.identity(C.shape[1]),a)-np.dot(C.transpose(),y))
	if(i % 1000 == 0 and np.sum(np.absolute(a_grad)) < 0.001):
		break
	a_grad_norm += np.square(a_grad)
	a -= ita*a_grad/np.sqrt(a_grad_norm)

#print(a)
# Overall error of train data
error_GD = np.sqrt(np.dot((y-np.dot(C,a)).transpose(),(y-np.dot(C,a)))/y.shape[0])
print("Root mean Squared Error (GD) = ",error_GD[0][0])

a_dir = np.dot(np.dot(np.linalg.inv(np.dot(C.transpose(),C)+regulate*np.identity(C.shape[1])),C.transpose()),y) # Direct solution
# Overall error of train data
error_DIR = np.sqrt(np.dot((y-np.dot(C,a_dir)).transpose(),(y-np.dot(C,a_dir)))/y.shape[0])
print("Error of train data = ",error_DIR[0][0])

# Writing parameters (model order, number of stributes, trained parameters)
outFile = open('parameters_best.txt','w')
outFile.write(str(model_order))
outFile.write(" ")
outFile.write(str(len(param_list)))
outFile.write(" ")
for i in range(len(param_list)):
	outFile.write(str(param_list[i]))
	outFile.write(" ")
for i in range(len(param_list)*model_order*numOfFeature+1):
	outFile.write(str(a[i][0]))
	outFile.write(" ")
outFile.close()