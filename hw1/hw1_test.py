import sys
import csv
import numpy as np

np.set_printoptions(suppress=True)
inFilePath = (sys.argv)[1]
outFilePath = (sys.argv)[2]
paramPath = (sys.argv)[3]
data =[]

numOfType = 18

# reading parameters
paramFile = open(paramPath,'rt')
params = paramFile.read().split()

model_order = int(params[0])
numOfUsedType = int(params[1])
param_list = list(map(int,params[2:2+numOfUsedType]))
params = params[2+numOfUsedType:]

numOfFeature = (len(params)-1)//(numOfUsedType*model_order)	# last numOfFeature hours of pm2.5 (== m)
a = np.array(params).reshape((len(params),1)).astype(np.float64)

# reading csv to data
with open(inFilePath, 'rt',  encoding = 'Big5') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spamreader:
		rowList = row[0].split(',')
		rowList = ['0' if x == 'NR' else x for x in rowList]
		data.append(rowList)
		#print(rowList)
		#input()

numOfData = len(data)//numOfType
C = np.ones(shape = (numOfData, model_order*len(param_list)*numOfFeature+1))
#a = np.zeros(shape = (numOfFeature+1,1))

# Storing data in C and y
start = -1 ; end = -1
index = 0
for i in range(numOfData):
	start = end + 1
	end = end + numOfType
	dayData = data[start:end+1]
	for j in range(len(param_list)):
		C[index][j*numOfFeature+1:(j+1)*numOfFeature+1] = np.array(dayData[param_list[j]][-numOfFeature:]).reshape((1,numOfFeature)).astype(np.float64)
	if model_order == 2:
		for j in range(len(param_list),2*len(param_list)):
			C[index][j*numOfFeature+1:(j+1)*numOfFeature+1] = np.square(np.array(dayData[param_list[j-len(param_list)]][-numOfFeature:]).reshape((1,numOfFeature)).astype(np.float64))
	index += 1

#evaluate
y_hat = np.dot(C,a)
y_hat[y_hat < 0] = 0
#print(y_hat)

# writing output csv 
with open(outFilePath, 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['id,value'])
	for i in range(numOfData):
		row_list = [','.join(["id_"+str(i),str(y_hat[i][0])])]
		spamwriter.writerow(row_list)