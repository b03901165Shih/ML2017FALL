import csv
import sys
import keras
import numpy as np
from keras.models import Sequential, model_from_json, Model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Embedding, Dropout, Bidirectional, Flatten
from keras.layers import Dot,Add,Concatenate,Input,Activation
		
def Normalize(y_train):
	mean_y = np.mean(y_train)
	std_y = np.std(y_train)
	norm_y_train = (y_train-mean_y)/std_y
	norm_y_train = norm_y_train.reshape(y_train.shape[0],1)
	return (norm_y_train,mean_y,std_y)
	
class ValidCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		valid_start = (int)(data.shape[0]*(1-validation))
		y_pred = self.model.predict([user_train[valid_start:,:], movies_train[valid_start:,:]], verbose = 1, batch_size=batch_size)
		y_pred = np.clip((y_pred*std_y)+mean_y,0,5)
		y_true = np.clip((y_train[valid_start:,:]*std_y)+mean_y,0,5)
		score = np.sqrt(np.sum((y_pred-y_true)**2)/y_true.shape[0])
		print('\nTesting loss: {}\n'.format(score))
	
np.set_printoptions(suppress=True)
np.random.seed(20171209)
TrainFilePath = (sys.argv)[1]

user_train = []
movies_train = []
y_train = []
data = []
userIdMap = {}
movieIdMap = {}

batch_size = 256
validation = 0.1
dimOfEmbed = 64

print('Parsing Data...')
# reading csv to data
with open(TrainFilePath, 'r') as csvfile:
 spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
 count = 0
 user_count = 0
 movie_count = 0
 for row in spamreader:	
  if(count == 0):
    count += 1
    continue
  if(not row[1] in userIdMap):
    userIdMap[row[1]] = user_count
    user_count+=1
  if(not row[2] in movieIdMap):
    movieIdMap[row[2]] = movie_count
    movie_count+=1
  data.append(row)
  if(len(data)%100000==0):
    print(len(data))
	
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

print('Mean:',mean_y)
print('Std:',std_y)

bias_train = np.ones(shape = (data.shape[0],1))

print('user_train.shape :',user_train.shape)
print('movies_train.shape :',movies_train.shape)
print('y_train.shape :',y_train.shape)
'''
divide_pnt = (int)(x_train_encode.shape[0]*(1-validation))
x_train = np.copy(x_train_encode[:divide_pnt,:])
y_train = np.copy(y_train_encode[:divide_pnt])
x_test = np.copy(x_train_encode[divide_pnt:,:])
y_test = np.copy(y_train_encode[divide_pnt:])
'''
drop_rate = 0.2

bias = Input(shape = [1])
users = Input(shape = [1])
movies = Input(shape = [1])
userEmbed = Embedding(numOfUser+1, dimOfEmbed, input_length=1)(users)
flat_userEmbed = Dropout(drop_rate)(Flatten()(userEmbed))
movieEmbed = Embedding(numOfMovies+1, dimOfEmbed, input_length=1)(movies)
flat_movieEmbed = Dropout(drop_rate)(Flatten()(movieEmbed))
dot_out = Dot(axes=1)([flat_userEmbed,flat_movieEmbed])


userEmbed_bias = Embedding(numOfUser+1, 1, input_length=1)(users)
flat_userEmbed_bias =Flatten()(userEmbed_bias) 
movieEmbed_bias = Embedding(numOfMovies+1, 1, input_length=1)(movies)
flat_movieEmbed_bias =Flatten()(movieEmbed_bias)
bias_bias = Embedding(1, 1, input_length=1)(bias)
flat_bias_bias =Flatten()(bias_bias) 
add_out = Add()([dot_out,flat_userEmbed_bias,flat_movieEmbed_bias])
#add_out = Activation('tanh')(add_out)

model = Model([users,movies],add_out)

model.summary()


model.compile(loss='mse', optimizer=keras.optimizers.Adamax())#lr = 0.002,decay=1e-5

early_stopping = EarlyStopping(monitor='val_loss', patience=2)


model.fit([user_train, movies_train], y_train,
          batch_size=batch_size,
          epochs=100,
		  validation_split = validation,
		  callbacks=[early_stopping,ValidCallback()])
  

#score = model.evaluate([user_train[valid_start:,:], movies_train[valid_start:,:],bias_train[valid_start:,:]], y_train[valid_start:,:], batch_size=batch_size)
							
valid_start = (int)(data.shape[0]*(1-validation))
			
y_pred = model.predict([user_train[valid_start:,:], movies_train[valid_start:,:]], verbose = 1, batch_size=batch_size)
y_pred = np.clip((y_pred*std_y)+mean_y,0,5)
y_true = np.clip((y_train[valid_start:,:]*std_y)+mean_y,0,5)
score = np.sqrt(np.sum((y_pred-y_true)**2)/y_true.shape[0])


print('\nTest score:', score)

# serialize model to JSON
model_json = model.to_json()
with open("MF_model_frame.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("MF_model_weight.h5")
print("Saved model to disk")

