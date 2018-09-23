
import keras 
import numpy as np
from keras.models import Model
from keras.layers import Convolution2D , MaxPooling2D ,Flatten,Dense,Dropout,BatchNormalization, Activation
import pandas as pd
import cv2
import glob
from skimage.transform import resize
from sklearn.model_selection import train_test_split 
from keras.layers.convolutional import Conv2DTranspose 



# calling the test set and training set 
#train_set = r"/media/dharani/New Volume/all(1)/train/images"
cv_img = []
for img in glob.glob("/media/dharani/New Volume/all (1)/train/images/*.png"):
    n= cv2.imread(img)
    cv_img.append(n)

cv_image_target = []
for img in glob.glob("/media/dharani/New Volume/all (1)/train/masks/*.png"):
    target = cv2.imread(img)
    cv_image_target.append(target)
    
cv_image_test = []
for img in glob.glob("/media/dharani/New Volume/all (1)/test/images/*.png"):
    test = cv2.imread(img)
    cv_image_test.append(test)

# resizing the input image
cv_img = np.asarray(cv_img)
cv_img = cv_img/255.
#cv_image = resize(cv_img,(4000,128,128,3))

# resizing the target image 
cv_image_target = np.asarray(cv_image_target)
cv_image_target = cv_image_target/255.
#cv_image_target = resize(cv_image_target,(4000,128,128,3))

# resizing the test image
cv_image_test = np.asarray(cv_image_test)
cv_image_test = cv_image_test/255.
#cv_image_test = resize(cv_image_test,(18000,128,128,3))

# splitting the train and test images 
train_x, train_y,test-x, test_y = train_test_split(cv_img,cv_image_target,test_size=0.3)

# building the u-net architecture 
# 2 layer with 3*3 convolution layer which has maxpooling layer and a batchnormalization layer too.
def convolution_layer(input_images, num_filters, batch_normalisation = True):
# defining first layer
    layer = convolution2D(filters= num_filters, kernal =(3,3),padding='same')(input_images)
    if batch_normalization:
        layer = batch_normalization()(layer)   
    layer = Activation("Relu")(layer)

# defining second layer    
    layer = convolution2D(filters= num_filters, kernal =(3,3),padding='same')(input_images)    
    if batch_normalization:
        layer = batch_normalization()(layer)    
    layer = Activation("Relu")(layer)
    
    return layer

def unet(input_images, num_filters,dropout,batch_normalization=True):
    # contracting path 
    convolution_layer_1 = convolution_layer(input_img, num_filters=num_filters, kernel_size=3, batch_normalisation =batch_normalisation )
    pooling_layer_1 = MaxPooling2D((2, 2)) (convolution_layer_1)
    pooling_layer_1 = Dropout(dropout)(pooling_layer_1)

    convolution_layer_2 = convolution_layer(input_img, num_filters=num_filters, kernel_size=3, batch_normalisation =batch_normalisation )
    pooling_layer_2 = MaxPooling2D((2, 2)) (convolution_layer_2)
    pooling_layer_2 = Dropout(dropout)(pooling_layer_2)

    convolution_layer_3 = convolution_layer(input_img, num_filters=num_filters, kernel_size=3, batch_normalisation =batch_normalisation )
    pooling_layer_3 = MaxPooling2D((2, 2)) (convolution_layer_3)
    pooling_layer_3 = Dropout(dropout)(pooling_layer_3)

    convolution_layer_4 = convolution_layer(input_img, num_filters=num_filters, kernel_size=3, batch_normalisation =batch_normalisation )
    pooling_layer_4 = MaxPooling2D((2, 2)) (convolution_layer_4)
    pooling_layer_4 = Dropout(dropout)(pooling_layer_4)
    
    convolution_layer_5 = convolution_layer(input_img, num_filters=num_filters, kernel_size=3, batch_normalisation =batch_normalisation 
    
    # expansive path
    expansion_layer_1 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (convolution_layer_5)
    expansion_layer_1 = concatenate([expansion_layer_1, convolution_layer_4])
    expansion_layer_1 = Dropout(dropout)(expansion_layer_1)
    convolution_layer_6 = convolution_layer(expansion_layer_1, n_filters=n_filters*8, kernel_size=3, batch_normalisation =batch_normalisation )

    expansion_layer_2 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (convolution_layer_6)
    expansion_layer_2= concatenate([expansion_layer_2, convolution_layer_3])
    expansion_layer_2 = Dropout(dropout)(expansion_layer_2)
    convolution_layer_7 = convolution_layer(expansion_layer_2, n_filters=n_filters*4, kernel_size=3, batch_normalisation =batch_normalisation )

    expansion_layer_3 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (convolution_layer_6)
    expansion_layer_3= concatenate([expansion_layer_3, convolution_layer_2])
    expansion_layer_3 = Dropout(dropout)(expansion_layer_3)
    convolution_layer_8 = convolution_layer(expansion_layer_3, n_filters=n_filters*4, kernel_size=3, batch_normalisation =batch_normalisation )


    expansion_layer_4 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (convolution_layer_6)
    expansion_layer_4= concatenate([expansion_layer_4, convolution_layer_1])
    expansion_layer_4 = Dropout(dropout)(expansion_layer_4)
    convolution_layer_9 = convolution_layer(expansion_layer_4, n_filters=n_filters*4, kernel_size=3, batch_normalisation =batch_normalisation )

    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (convolution_layer_9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

model = unet((128,128,3), n_filters=12, dropout=0.5, batchnormalization=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# fitting the model 
results = model.fit(x_train, y_train, batch_size=32, epochs=100,
                    validation_data=(x_test, y_test))

# evaluating the model 
model.evaluate(x_test, y_test)


