import csv
import sys
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Embedding, Dropout, Bidirectional
from keras.layers import LSTM, GRU, Flatten, AveragePooling1D, Conv1D
from gensim.models import Word2Vec
		
np.set_printoptions(suppress=True)
TrainFilePath = (sys.argv)[1]
TrainNoLabelFilePath = (sys.argv)[2]

x_train = []
y_train = []
batch_size = 2560
validation = 0.1

# reading csv to data
with open(TrainFilePath, 'r', errors='ignore', encoding = 'Big5') as file:
 spamreader = file.read().split('\n')
 for row in spamreader[:-1]:
  rowList = row.split(' +++$+++ ')
  x_train.append(rowList[1])
  y_train.append(rowList[0])
  

x_train = np.array(x_train)
y_train = np.array(y_train)
#print(x_train.shape)
#print(y_train.shape)

embed_model = Word2Vec.load('model_embed.bin')

# To Sequence
print('Text To Sequence...')
x_train_tokens = []

for i in range(x_train.shape[0]):
	x_train_token = text_to_word_sequence(x_train[i], filters= '\t\n')
	x_train_tokens.append(x_train_token)

max_length = 0
for i in range(len(x_train_tokens)):
	if max_length < len(x_train_tokens[i]):
		max_length = len(x_train_tokens[i])
print('max_length:',max_length)

x_train_encode = np.zeros(shape = (len(x_train_tokens),max_length, 32))
for i in range(len(x_train_tokens)):
	x_train_encode[i][-len(x_train_tokens[i]):][:] = embed_model[x_train_tokens[i]]

x_train_encode = np.array(x_train_encode).astype('float32')
y_train_encode = np.array(y_train).astype('float32')

print(np.array(x_train_encode).shape)
print(np.array(y_train_encode).shape)

#y_train_encode = keras.utils.to_categorical(y_train_encode, num_classes=2)

divide_pnt = (int)(x_train_encode.shape[0]*(1-validation))
x_train = np.copy(x_train_encode[:divide_pnt,:])
y_train = np.copy(y_train_encode[:divide_pnt])
x_test = np.copy(x_train_encode[divide_pnt:,:])
y_test = np.copy(y_train_encode[divide_pnt:])

########################################################################################

print('Finding Self Learning corpus')
# load json and create model
json_file = open('RNN_model_frame_forSelf.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_prev = model_from_json(loaded_model_json)
model_prev.load_weights("RNN_model_weight_forSelf.h5")

model_prev.summary()

# try using different optimizers and different optimizer configs
model_prev.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
			  
x_nolabel = []
x_nolabel_tokens = []

# reading to data
with open(TrainNoLabelFilePath, 'r', errors='ignore', encoding = 'Big5') as file:
 spamreader = file.read().split('\n')
 for row in spamreader[:-1]:
  x_nolabel.append(row)

x_nolabel = np.array(x_nolabel)

for i in range(x_nolabel.shape[0]):
	x_train_token = text_to_word_sequence(x_nolabel[i], filters= '\t\n')
	x_nolabel_tokens.append(x_train_token)
	
x_nolabel = []

for k in range(11):
	x_nolabel_encode = np.zeros(shape = (100000, max_length, 32))
	for i in range(100000):
		if(len(x_nolabel_tokens[i+k*100000]) == 0):
			continue
		if(i %10000==0):
			print('Encoding For '+str(i+k*100000)+' is Done')
		if(len(x_nolabel_tokens[i+k*100000])> max_length):
			#x_nolabel_encode[i][:][:] = embed_model[x_nolabel_tokens[i+k*100000]][-max_length:][:]
			continue
		else:
			x_nolabel_encode[i][-len(x_nolabel_tokens[i+k*100000]):][:] = embed_model[x_nolabel_tokens[i+k*100000]]

	x_nolabel_encode = np.array(x_nolabel_encode).astype('float32')
	print(x_nolabel_encode.shape)

	y_pred_raw = model_prev.predict(x_nolabel_encode, batch_size = batch_size, verbose = 1).reshape(100000,)
	y_pred = np.round(y_pred_raw)
	
	index = np.abs(y_pred_raw-y_pred) < 0.1
	
	x_nolabel_encode = x_nolabel_encode[index][:]
	y_pred = y_pred[index]
	
	x_train = np.concatenate((x_train,x_nolabel_encode),axis = 0)
	y_train = np.concatenate((y_train, y_pred), axis = 0)
	
	print(x_train.shape)
	print(y_train.shape)

print('Final x_train shape:', x_train.shape)
print('Final y_train shape:', y_train.shape)


########################################################################################

early_stopping = EarlyStopping(monitor='val_acc', patience=2)

model = Sequential()
#model.add(Conv1D(64, kernel_size = 3,padding = 'same', input_shape = (max_length, 32)))
#model.add(Conv1D(128, kernel_size = 3,padding = 'same'))
#model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)))
#model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.2)))
model.add(GRU(128, input_shape = (max_length, 32), dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True)))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])



print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
		  validation_data = (x_test,y_test),
		  callbacks=[early_stopping])
		  
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('\nTest score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("RNN_model_frame_final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("RNN_model_weight_final.h5")
print("Saved model to disk")
model.summary()