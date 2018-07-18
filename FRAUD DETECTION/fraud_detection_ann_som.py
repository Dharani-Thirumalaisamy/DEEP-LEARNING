# using supervised model with unsupervised model to rank the probability of fraudluent 

#unsupervised model 
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:15:04 2018

@author: Dell-pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout


# dataset 
df = pd.read_csv('Credit_Card_Applications.csv')
x = df.iloc[:,0:15].values
y = df.Class

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# SOM model
detector = MiniSom(x=10 , y=10 , input_len=15, learning_rate = 0.5 , sigma=1.0) #initialising the model
detector.random_weights_init(x) # generate random weights
detector.train_random(x , num_iteration=50) # train the model on the data ie the input

#visualising the model 
from pylab import bone , pcolor , colorbar , plot , show
bone()
pcolor(detector.distance_map().T)
colorbar()
markers = ['o','s']
color = ['r','g']
for i ,n in enumerate(x):
    bmu = detector.winner(x)
    plot(bmu[0]+0.5,
         bmu[1]+0.5,
         markers[y[i]],
         markeredgecolor = color[y[i]],
         markerfacecolor='None',
         markersize = 3,
         markeredgewidth = 5)
show()
    
mappings = detector.win_map(x)
frauds = np.concatenate((mappings[(5,8)] , mappings[(1,1)]),axis = 0)
frauds = scaler.inverse_transform(frauds)

# supervised model 
customer_info = df.iloc[:,1:].values

cheated  = np.zeros(len(df))
for i in range (len(df)):
    if df.iloc[i,0] in frauds :
        cheated[i] =1

sc = StandardScaler()
customer_info = sc.fit_transform(customer_info)

classifier = Sequential()
classifier.add(Dense(units = 2 ,init = 'uniform',activation = 'relu',input_dim = 15 ))
#classifier.add(Dropout(0.1))
#classifier.add(Dense(output_dim =6 ,init = 'uniform',activation = 'relu'))
#classifier.add(Dropout(0.1))
classifier.add(Dense(output_dim =1 ,init = 'uniform',activation = 'sigmoid'))
classifier.compile(optimizer='adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])        

classifier.fit(customer_info , cheated , batch_size = 1 , epochs = 1)

fraud_prediction = classifier.predict(customer_info)
fraud_prediction = np.concatenate((df.iloc[:,0:1].values , fraud_prediction),axis=0)
fraud_prediction = fraud_prediction[fraud_prediction[:,1].argsort()]
