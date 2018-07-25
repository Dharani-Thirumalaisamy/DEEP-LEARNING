### classification of fruit dataset without augmentation

# importing the libraries 
import os
import PIL.Image #pillow library
import numpy as np
import keras 
import glob
#import skimage.io
#from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D , MaxPooling2D , Dropout , Flatten ,Dense
from keras.utils import np_utils

## importing the training and testing  dataset from local machine
# Training dataset
files = os.listdir(r'..\fruits-360_dataset\fruits-360\Training')
images =[]
i = 0
while (i<75):
    directory = r'..\fruits-360_dataset\fruits-360\Training'
    separator = '\\'
    for filename in glob.glob(directory+separator+files[i]+separator+'*.jpg'): # to access all the jpg files in that folder
        im=PIL.Image.open(filename)
        images.append(im)
    i+=1
  
# Testing dataset
files = os.listdir(r'..\fruits-360_dataset\fruits-360\Test')
images_test =[]
n = 0
while (n<75):
    directory = r'..\fruits-360_dataset\fruits-360\Test'
    separator = '\\'
    for filename in glob.glob(directory+separator+files[i]+separator+'*.jpg'):
        im=PIL.Image.open(filename)
        images_test.append(im)
    n+=1    

## formatting the training and the test set
img_height = 100
img_width = 100
scale = 3 # RGB
categories = 75 # number of output categories
# TRAINING SET 
scaled_train = []
for pic in range(0,2):
    np_array = list(images[pic].getdata())
    np_array = np.asarray(np_array)     # converting the list of images to array so that it could be normalized
    #scaled_data = np_array/255.
    scaled_train.append(np_array)

# TEST SET 
scaled_test = []        
for pic in range(len(images_test)):
    np_array = list(images_test[pic].getdata())
    np_array = np.asarray(np_array)
    #scaled_data = np_array/255.
    scaled_test.append(np_array)

# splitting and reshaping    
x_train , y_train = train_test_split(scaled_train , test_size = 0.3)
x_test , y_test = train_test_split(scaled_test , test_size = 0.3)

x_train = np.asarray(x_train)
x_train = x_train/255.

x_test = np.asarray(x_test)
x_test = x_test/255.

x_train = x_train.reshape(x_train[0],img_height,img_width,scale)
x_test = x_test.reshape(x_test[0],img_height,img_width,scale)

y_train = y_train.utils.to_categorical(y_train , categories)
y_test = y_test.utils.to_categorical(y_test , categories)

## CNN ARCHITECTURE
model = Sequential()
model.add(Convolution2D(32,3,3,input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(32,3,3 ,activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(output_dim = 1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim = categories ,activation='softmax'))

# compilation
model.compile(optimizer = 'adam' , loss='categorical_crossentropy' , metrics=['accuracy'])

# fitting the model 
model.fit(x_train , y_train , batch_size = 32 ,epochs=100 , validation_split=(x_test , y_test))

score = model.evaluate(x_test ,y_test)
print(score)
