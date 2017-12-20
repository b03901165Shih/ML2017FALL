import csv
import sys
import keras
import numpy as np
from keras.models import Sequential, model_from_json, Model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Embedding, Dropout, Bidirectional, Flatten
from keras.layers import Dot,Add,Concatenate,Input,Reshape,Merge

	
def Normalize_all(data):
	data_return = np.zeros(shape = data.shape)
	for i in range(data.shape[1]):
		mean = np.mean(data[:,i])
		std = np.mean(data[:,i])
		data_return[:,i] = (data[:,i]-mean)/std
	return data_return
	
def deNormalize(y):
	norm_y = (y*std)+mean
	norm_y = norm_y.reshape(y.shape[0],1)
	return norm_y
	
np.set_printoptions(suppress=True)
np.random.seed(20171209)
TestFilePath = (sys.argv)[1]
outFilePath =  (sys.argv)[2]

user_test = []
movies_test = []
data = []

batch_size = 256
mean = 0#3.58171
std = 1#1.1169

print('Parsing Data...')
# reading csv to data
with open(TestFilePath, 'r') as csvfile:
 spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
 count = 0
 for row in spamreader:	
  if(count == 0):
    count += 1
    continue
  data.append(row)
  
data = np.array(data).astype('int32')
#np.random.shuffle(data)

numOfUser = 6040 #np.max(data[:,1])
numOfMovies = 3952 #np.max(data[:,2]) 
#print(len(userIdMap)) # 6040
#print(len(movieIdMap)) # 3688
	
user_test = np.array(data[:,1]).reshape(data.shape[0],1).astype('int32')
movies_test = np.array(data[:,2]).reshape(data.shape[0],1).astype('int32')

print('user_test.shape :',user_test.shape)
print('movies_test.shape :',movies_test.shape)

# load json and create model
json_file = open('MF_model_frame.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("MF_model_weight.h5")

model.summary()

model.compile(loss='mse', optimizer=keras.optimizers.Adamax())#,decay=0.0001


y_pred = model.predict([user_test, movies_test], verbose = 1, batch_size=batch_size)
y_pred = deNormalize(y_pred)

print('y_pred.shape = ', y_pred.shape)
print(y_pred[:20,:])


# writing output csv 
with open(outFilePath, 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['TestDataID,Rating'])
	for i in range(y_pred.shape[0]):
		row_list = [','.join([str(i+1),str(float(y_pred[i,0]))])]
		spamwriter.writerow(row_list)
