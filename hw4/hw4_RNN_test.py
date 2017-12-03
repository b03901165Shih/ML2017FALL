import csv
import sys
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, Dropout, Bidirectional
from keras.layers import LSTM, GRU, Flatten, AveragePooling1D, Conv1D
from gensim.models import Word2Vec

np.set_printoptions(suppress=True)
inFilePath = (sys.argv)[1]
outFilePath = (sys.argv)[2]

x_train = []
y_train = []
batch_size = 1280
numOfTestData = 200000

# reading csv to data
with open(inFilePath, 'r', errors='ignore', encoding = 'Big5') as file:
 spamreader = file.read().split('\n')
 for row in spamreader[1:-1]:
  rowList = row.split(',')
  x_train.append(''.join(rowList[1:]))

x_train = np.array(x_train)

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

print(np.array(x_train_encode).shape)

# load json and create model
json_file = open('RNN_model_frame_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("RNN_model_weight_final.h5")

model.summary()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


			  
#y_pred = np.argmax(model.predict(x_train_encode, batch_size = batch_size, verbose = 1),axis = 1).reshape(numOfTestData,1)
y_pred = np.round(model.predict(x_train_encode, batch_size = batch_size, verbose = 1))
 
print(y_pred.shape)
#print(y_pred[:10,:])

# writing output csv 
with open(outFilePath, 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['id,label'])
	for i in range(numOfTestData):
		row_list = [','.join([str(i),str(int(y_pred[i,0]))])]
		spamwriter.writerow(row_list)