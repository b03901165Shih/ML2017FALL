import sys
import os
import numpy as np
import skimage
from skimage import io
from skimage import img_as_float
from numpy.linalg import svd,eig

def Averaging(M):
    re_M = np.zeros(shape = M.shape)
    aves = np.mean(M, axis = 0)
    for i in range(M.shape[0]):
        re_M[i] = M[i]-aves
    return (re_M,aves)

def Normalize(M):
    re_M = np.zeros(shape = M.shape)
    squr = np.sqrt(np.sum(M**2,axis = 1))
    for i in range(M.shape[0]):
        re_M[i] = M[i]/squr[i]
    return re_M
	
def EigenFacePreProcess(V):
    re_V = np.copy(V)
    re_V -= np.min(re_V)
    re_V /= np.max(re_V)
    re_V = (re_V*255).astype(np.uint8)
    #V = V.reshape(600,600)
    return re_V

dirPath = (sys.argv)[1]+'/'#'Aberdeen'#
numOfImage = len(os.walk(dirPath).__next__()[2])

toReconPath = dirPath+(sys.argv)[2]#'Aberdeen/10.jpg' #

images = np.zeros(shape = (numOfImage,600,600,3))

test_image = np.zeros(shape = (numOfImage,600,600,3))

print('Reading Images...')
for i in range(numOfImage):
    path = dirPath+'/'+str(i)+'.jpg'
    images[i] = io.imread(path)/255
			
(images,averages) = Averaging(images.reshape(numOfImage,600*600*3))
test_image= (io.imread(toReconPath)/255).reshape(1,600*600*3)-averages

print('Eigen vecs and Eigien vals process...')
eig_val,eig_vec = eig(np.dot(images,images.transpose())/numOfImage)
real_eig_vec = np.dot(images.transpose(),eig_vec)
real_eig_vec = Normalize(real_eig_vec.transpose()).transpose()

#array([ 0.04144625,  0.02948732,  0.02387711,  0.02207842])
# Draw Top 4 eigen faces
'''
for i in range(8,10):
    face = EigenFacePreProcess(real_eig_vec[:,i])
    io.imshow(face.reshape(600,600,3))
    #io.imsave('eig1_'+str(i)+'.png',face.reshape(600,600,3))
    io.show()
    face = -EigenFacePreProcess(real_eig_vec[:,i])
    io.imshow(face.reshape(600,600,3))
    #io.imsave('eig2_'+str(i)+'.png',face.reshape(600,600,3))
    io.show()


all_val = np.sum(eig_val)
for i in range(4):
    print('eig val '+str(i)+' weight: ', eig_val[i]/all_val)
'''

print('Reconstruction...')
k = 4
dim_reduct = np.dot(test_image,real_eig_vec[:,:k])

face = np.copy(averages)
for j in range(k):
    face += dim_reduct[0,j]*(real_eig_vec[:,j])
	
face_norm = EigenFacePreProcess(face)
face_norm = face_norm.reshape(600,600,3)
io.imsave('reconstruction.jpg',face_norm,quality=100)
'''
face_truth = ((test_image[0]+averages)*255).reshape(600,600,3).astype('uint8')
fig=np.concatenate((face_norm,face_truth),axis=1)
io.imshow(fig)
io.show()
'''
