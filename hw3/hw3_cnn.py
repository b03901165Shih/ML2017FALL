import csv
import sys
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

batch_size = 256
numOfClass = 7
epochs = 100
# input image dimensions
img_rows, img_cols = 48, 48
numOfTrainData = 28709
input_shape = (img_rows,img_cols,1)
validation_part = 0.1

np.set_printoptions(suppress=True)
inFilePath = (sys.argv)[1]
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

X_train = data[:,:-1].reshape(numOfTrainData,img_rows,img_cols,1)

y_train = np.zeros(shape = (numOfTrainData,numOfClass))
for i in range(numOfTrainData):
 y_train[i,int(data[i,0])] = 1;

print(X_train.shape)
print(y_train.shape)

X_train = X_train.astype('float32')
X_train /= 255

print('Start Training...')

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


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-4),#.Adadelta(),
              metrics=['accuracy'])
			  
outputs = np.array([layer.output for layer in model.layers])
print(outputs)

# Early stopping
early_stopping = EarlyStopping(monitor='val_acc', patience=5)

train_datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    shear_range= 0.1,
	zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
		
test_datagen = ImageDataGenerator()

numOfTrain = (int)(numOfTrainData*(1-validation_part))
numOfVad = numOfTrainData-numOfTrain


model.fit_generator(train_datagen.flow(X_train[:numOfTrain,:,:,:], y_train[:numOfTrain,:], batch_size=batch_size),
          epochs=epochs,
		  validation_data = test_datagen.flow(X_train[numOfTrain:,:,:,:], y_train[numOfTrain:,:], batch_size=batch_size),
		  validation_steps = np.floor(numOfVad/batch_size),
          verbose=1,
          steps_per_epoch= np.floor(numOfTrain/batch_size),
		  callbacks=[early_stopping])
		  
score = model.evaluate(X_train[numOfTrain:,:,:,:], y_train[numOfTrain:,:], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights("hw3_model_self.h5")
print("Saved model to disk")