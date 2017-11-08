import csv
import sys
import keras
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

batch_size = 128
numOfClass = 7
epochs = 50
# input image dimensions
img_rows, img_cols = 48, 48
numOfTestData = 7178
input_shape = (img_rows,img_cols,1)

np.set_printoptions(suppress=True)
inFilePath = (sys.argv)[1]
outFilePath = (sys.argv)[2]
data = []

# reading csv to data
with open(inFilePath, 'rt') as csvfile:
 spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
 count = 0
 for row in spamreader:
  if count == 0:
   count += 1;continue		
  rowList = row[0].split(',')
  rowList += row[1:]
  data.append(rowList)

data = np.array(data)

X_test = data[:,:-1].reshape(numOfTestData,img_rows,img_cols,1)

print(X_test.shape)

X_test = X_test.astype('float32')
X_test /= 255


#best model yet
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(5, 5),
                 activation='linear',
                 input_shape=input_shape))
model.add(LeakyReLU(alpha=0.004))
model.add(Conv2D(64, kernel_size=(5, 5), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(Conv2D(128, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(Conv2D(256, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(Conv2D(256, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(Conv2D(512, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(Conv2D(512, kernel_size=(3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.004))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2560, activation='relu'))
model.add(Dense(2560, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(numOfClass, activation='softmax'))
# load weights into new model
model.load_weights("hw3_model_self.h5")
print("Loaded model from disk")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),#.Adadelta(),
              metrics=['accuracy'])
			  
outputs = np.array([layer.output for layer in model.layers])
print(outputs)

print('Testing ------------')

y_pred = np.argmax(model.predict(X_test, verbose = 1),axis = 1).reshape(numOfTestData,1)

print(y_pred.shape)

'''
import matplotlib.pyplot as plt
label_trans = ['生氣','厭惡','恐懼','高興','難過','驚訝','中立']
index = 0
while index != -1:
	image_X = X_test[index,:,:,0]
	plt.imshow(image_X,cmap = 'gray')
	print('Label = ',label_trans[y_pred[index,0]])
	plt.show()
	index += 1
'''

# writing output csv 
with open(outFilePath, 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['id,label'])
	for i in range(numOfTestData):
		row_list = [','.join([str(i),str(int(y_pred[i,0]))])]
		spamwriter.writerow(row_list)