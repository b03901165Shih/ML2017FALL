import csv
import sys
import keras
import numpy as np
from keras.models import Sequential, model_from_json, Model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Embedding, Dropout, Bidirectional, Flatten
from keras.layers import Dot,Add,Concatenate,Input,Activation,Merge,concatenate
from keras.regularizers import l2,l1
		
def Normalize(y_train):
	mean_y = np.mean(y_train)
	std_y = np.std(y_train)
	norm_y_train = (y_train-mean_y)/std_y
	norm_y_train = norm_y_train.reshape(y_train.shape[0],1)
	return (norm_y_train,mean_y,std_y)
	
def Normalize_all(data):
	data_return = np.zeros(shape = data.shape)
	for i in range(data.shape[1]):
		mean = np.mean(data[:,i])
		std = np.mean(data[:,i])
		data_return[:,i] = (data[:,i]-mean)/std
	return data_return

def Acc(model):
	valid_start = (int)(data.shape[0]*(1-validation))
	y_pred = model.predict([user_train[valid_start:,:], movies_train[valid_start:,:],user_train_trait[valid_start:,:]], verbose = 1, batch_size=batch_size)
	y_pred = np.clip((y_pred*std_y)+mean_y,0,5)
	y_true = np.clip((y_train[valid_start:,:]*std_y)+mean_y,0,5)
	score = np.sqrt(np.sum((y_pred-y_true)**2)/y_true.shape[0])
	return score
	
class ValidCallback(Callback):
	def __init__(self, test_data):
		self.test_data = test_data

	def on_epoch_end(self, epoch, logs={}):
		accurate = self.test_data
		score = Acc(self.model)
		print('\nTesting loss: {}\n'.format(score))
		if score+0.0005 < accurate:
			self.model.stop_training = True
	
np.set_printoptions(suppress=True)
np.random.seed(20171209)
TrainFilePath = (sys.argv)[1]
UserFilePath = (sys.argv)[2]

user_train = []
movies_train = []
y_train = []
data = []

userTrait = {}

batch_size = 256
validation = 0.1
dimOfEmbed = 32

print('Parsing Data...')
# reading csv to data
with open(TrainFilePath, 'r') as csvfile:
 spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
 count = 0
 for row in spamreader:	
  if(count == 0):
    count += 1
    continue
  data.append(row)
	
with open(UserFilePath, 'r') as csvfile:
 spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
 count = 0
 for row in spamreader:	
  if(count == 0):
    count += 1
    continue
  row = np.array(row[0].split('::'))
  if row[1]=='F':
    row[1]='0'
  else:
    row[1]='1'
  row[4] = row[4][:5]
  row = row.astype('int32')
  userTrait[row[0]] = row[1:] 
  	
data = np.array(data).astype('int32')
np.random.shuffle(data)

numOfUser = np.max(data[:,1]) # 6040
numOfMovies = np.max(data[:,2]) # 3952

print('numOfUser:',numOfUser)
print('numOfMovies:',numOfMovies)
#print(len(userIdMap)) # 6040
#print(len(movieIdMap)) # 3688
	
user_train = np.array(data[:,1]).reshape(data.shape[0],1).astype('int32')
movies_train = np.array(data[:,2]).reshape(data.shape[0],1).astype('int32')
y_train = np.array(data[:,3]).reshape(data.shape[0],1).astype('float32')
#(y_train,mean_y,std_y) = Normalize(y_train)
mean_y = 0 ; std_y = 1

'''
user_train_trait = np.zeros(shape = (data.shape[0],5))
for i in range(user_train.shape[0]):
  user_train_trait[i] = np.concatenate(([user_train[i][0]],userTrait[user_train[i][0]]),axis=0)
'''
  
user_train_trait = np.zeros(shape = (data.shape[0],4))
for i in range(user_train.shape[0]):
  user_train_trait[i] = userTrait[user_train[i][0]]
user_train_trait = Normalize_all(user_train_trait).astype('float32')

print('Mean:',mean_y)
print('Std:',std_y)

#user_train = Normalize_all(user_train).astype('float32')
#movies_train = Normalize_all(movies_train).astype('float32')

print('user_train.shape :',user_train.shape)
print('movies_train.shape :',movies_train.shape)
print('y_train.shape :',y_train.shape)

# load json and create model
json_file = open('NN_model_frame.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("NN_model_weight.h5")

model.summary()

model.compile(loss='mse', optimizer=keras.optimizers.Adamax(lr = 0.001))#lr = 0.002,decay=1e-5

accuracy = Acc(model)
print('\nAccuracy before:', accuracy)

model.fit([user_train, movies_train,user_train_trait], y_train,
          batch_size=batch_size,
          epochs=100,
		  validation_split = validation,
		  callbacks=[ValidCallback(accuracy)])
  

#score = model.evaluate([user_train[valid_start:,:], movies_train[valid_start:,:],bias_train[valid_start:,:]], y_train[valid_start:,:], batch_size=batch_size)
							
valid_start = (int)(data.shape[0]*(1-validation))
			
y_pred = model.predict([user_train[valid_start:,:], movies_train[valid_start:,:] ,user_train_trait[valid_start:,:]], verbose = 1, batch_size=batch_size)
y_pred = np.clip((y_pred*std_y)+mean_y,0,5)
y_true = np.clip((y_train[valid_start:,:]*std_y)+mean_y,0,5)
score = np.sqrt(np.sum((y_pred-y_true)**2)/y_true.shape[0])


print('\nTest score:', score)

# serialize model to JSON
model_json = model.to_json()
with open("NN_model_frame.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("NN_model_weight.h5")
print("Saved model to disk")

