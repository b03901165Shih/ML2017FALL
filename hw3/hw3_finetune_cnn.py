import csv
import sys
import keras
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, Callback
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

def Acc(y,y_pred):
	return np.sum(y==y_pred)/y.shape[0]

class ValidCallback(Callback):
	def __init__(self, test_data):
		self.test_data = test_data

	def on_epoch_end(self, epoch, logs={}):
		x, y, accurate = self.test_data
		y_pred = np.argmax(model.predict(x),axis = 1).reshape(numOfVad,1)
		y_true = np.argmax(y,axis = 1).reshape(numOfVad,1)
		acc = Acc(y_true,y_pred)
		print('\nTesting acc: {}\n'.format(acc))
		if acc > accurate+0.001:
			self.model.stop_training = True

batch_size = 256
numOfClass = 7
epochs = 40
# input image dimensions
img_rows, img_cols = 48, 48
numOfTrainData = 28709
input_shape = (img_rows,img_cols,1)
validation_part = 0.1

numOfTrain = (int)(numOfTrainData*(1-validation_part))
numOfVad = numOfTrainData-numOfTrain

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
#x_test = x_test.astype('float32')
X_train /= 255

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
model.add(Dense(numOfClass, activation='softmax'))



model.load_weights("hw3_model_self.h5")


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
			  
outputs = np.array([layer.output for layer in model.layers])
print(outputs)

for layer in model.layers[:5]:
    layer.trainable = False
	
# Early stopping
#early_stopping = EarlyStopping(monitor='val_acc', patience=5)


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
	
#train_datagen.fit(X_train[:numOfTrain,:,:,:])
#test_datagen.fit(X_train[numOfTrain:,:,:,:])


y_pred = np.argmax(model.predict(X_train[numOfTrain:,:,:,:]),axis = 1).reshape(numOfVad,1)
y = np.argmax(y_train[numOfTrain:,:],axis = 1).reshape(numOfVad,1)
accurate = Acc(y,y_pred)
print('Test accuracy:', accurate)
	
	
model.fit_generator(train_datagen.flow(X_train[:numOfTrain,:,:,:], y_train[:numOfTrain,:], batch_size=batch_size),
          epochs=epochs,
		  validation_data = test_datagen.flow(X_train[numOfTrain:,:,:,:], y_train[numOfTrain:,:], batch_size=batch_size),
		  validation_steps = np.floor(numOfVad/batch_size),
          verbose=1,
          steps_per_epoch= np.floor(numOfTrain/batch_size),
		  #callbacks=[early_stopping]
		  callbacks = [ValidCallback((X_train[numOfTrain:,:,:,:], y_train[numOfTrain:,:], accurate))]
		  )
		  
score = model.evaluate(X_train[numOfTrain:,:,:,:], y_train[numOfTrain:,:], verbose=0, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model.save_weights("hw3_model_self.h5")
print("Saved model to disk")
