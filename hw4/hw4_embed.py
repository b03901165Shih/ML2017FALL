import csv
import sys
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM, GRU, Flatten
from gensim.models import Word2Vec

np.set_printoptions(suppress=True)
TrainLabelFilePath = (sys.argv)[1] #(sys.argv)[1]
TrainNoLabelFilePath = (sys.argv)[2] #(sys.argv)[1]
TestFilePath = 'testing_data.txt'

x_train = []
y_train = []
batch_size = 1280
validation = 0.1

# reading to data
with open(TrainLabelFilePath, 'r', errors='ignore', encoding = 'Big5') as file:
 spamreader = file.read().split('\n')
 for row in spamreader[:-1]:
  rowList = row.split(' +++$+++ ')
  x_train.append(rowList[1])
  
# reading to data
with open(TestFilePath, 'r', errors='ignore', encoding = 'Big5') as file:
 spamreader = file.read().split('\n')
 for row in spamreader[1:-1]:
  rowList = row.split(',')
  x_train.append(''.join(rowList[1:]))
  
# reading to data
with open(TrainNoLabelFilePath, 'r', errors='ignore', encoding = 'Big5') as file:
 spamreader = file.read().split('\n')
 for row in spamreader[:-1]:
  x_train.append(row)

x_train = np.array(x_train)
#print(x_train.shape)
#print(y_train.shape)

print('Text To Sequence...')
x_train_tokens = []

for i in range(x_train.shape[0]):
	x_train_token = text_to_word_sequence(x_train[i], filters= '\t\n')
	x_train_tokens.append(x_train_token)
	
print('Word Embedding...')
model = Word2Vec(x_train_tokens, size = 32, min_count=1, workers = 4)
	
model.save('model_embed.bin')
