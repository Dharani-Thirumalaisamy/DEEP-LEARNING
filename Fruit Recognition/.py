# importing libraries 
import keras 
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout , Dense , Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# cnn architecture
model = Sequential()
model.add(Convolution2D(32 , 3,3 , input_shape=(100,100,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32 , 3,3 , activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(output_dim=1024 ,activation = 'relu'))
model.add(Dense(output_dim = 75 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss='categorical_crossentropy' , metrics=['accuracy'])

# loading the trainig and test set using image data generator
train_set = r'C:\Users\Dell-pc\DEEP LEARNING\fruits-360_dataset\fruits-360\Training'
test_set = r'C:\Users\Dell-pc\DEEP LEARNING\fruits-360_dataset\fruits-360\Test'
batch_size = 32
img_height = 100
img_width = 100

train = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range =0.2,
        horizontal_flip = True)
test = ImageDataGenerator( rescale = 1./255)

training_generator = train.flow_from_directory(
        train_set,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test.flow_from_directory(
        test_set , 
        target_size = (img_height,img_width),
        batch_size = batch_size,
        class_mode = 'categorical')

# fitting the model
model.fit_generator(
        training_generator,
        steps_per_epoch=110,
        epochs=50,
        validation_data=test_generator,
        validation_steps=800)
