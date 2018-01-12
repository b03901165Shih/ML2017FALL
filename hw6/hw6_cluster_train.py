import csv
import sys
import keras
import numpy as np
from keras.models import Sequential, model_from_json, Model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense,Dropout
from keras.layers import Input
from sklearn.cluster import KMeans

trainFlag = (sys.argv)[1]
npyPath = (sys.argv)[2]#'image.npy'#(sys.argv)[1]
testPath = (sys.argv)[3]#'test_case.csv'
outputPath = (sys.argv)[4]#'pred_old.csv'

images = np.load(npyPath)
images = images/255

'''
for i in range(images.shape[0]):
    plot(images[i].reshape(28,28))
'''
imgs = Input(shape = (784,))

dense_1 = Dense(256,activation='relu')(imgs)
#dense_2 = Dense(256,activation='relu')(dense_1)
dense_3 = Dense(48,activation='relu')(dense_1)
middle = Dense(16,activation='relu')(dense_3)

dense_9 = Dense(48,activation='relu')(middle)
dense_10 = Dense(256,activation='relu')(dense_9)
#dense_11 = Dense(512,activation='relu')(dense_10)
output = Dense(784,activation='relu')(dense_10)

auto_encoder = Model(imgs, output)
if trainFlag != 'train':
    auto_encoder.load_weights('hw6_autoencoder_16.h5')
auto_encoder.compile(loss='mse',optimizer=keras.optimizers.Adamax())

epo_count = 0

while epo_count < 50:
	print('Number of Iteration :', epo_count)
	if trainFlag=='train':		
		auto_encoder.fit(images,images,batch_size=256,epochs=10)
		#auto_encoder.save('hw6_autoencoder_64.h5')

	encoder = Model(imgs, middle)
	encoder.compile(loss='mse',optimizer=keras.optimizers.Adamax())

	dr_images = encoder.predict(images)

	n_clusters = 2
	cores = KMeans(n_clusters=n_clusters, random_state=0).fit(dr_images)
	y = cores.labels_

	labelCount = []
	for i in range(n_clusters):
		labelCount.append(np.sum(y==i)/y.shape[0])
		print('Labels split ratio of '+str(i)+' :', np.sum(y==i)/y.shape[0])
		
	labelCount = np.array(labelCount)
	epo_count += 1
	if np.abs(labelCount[0]-0.5)<0.000007 or trainFlag != 'train':
		break
		

data_test = []
# reading csv to data
with open(testPath, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    count = 0
    for row in spamreader:
        if count == 0:
            count += 1;continue		
        rowList = row[0].split(',')[1:]
        data_test.append(rowList)

y_pred = np.zeros(shape = (len(data_test),),dtype = 'int32')
for i in range(len(data_test)):
    #print('Index '+data_test[i][0]+' and '+data_test[i][1]+' :','class: ('+str(y[int(data_test[i][0])])+', '+str(y[int(data_test[i][1])])+')');input()
    if(int(y[int(data_test[i][0])]) == int(y[int(data_test[i][1])])):
        y_pred[i] = 1

print('0 and 1 ratio:',np.sum(y_pred)/y_pred.shape[0])

# writing output csv 
with open(outputPath, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|')
    spamwriter.writerow(['ID,Ans'])
    for i in range(y_pred.shape[0]):
        row_list = [','.join([str(i),str(y_pred[i])])]
        spamwriter.writerow(row_list)
