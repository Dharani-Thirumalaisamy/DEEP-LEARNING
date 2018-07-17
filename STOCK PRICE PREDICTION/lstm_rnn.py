## PREDICTING THE STOCK PRICE OF GOOGLE USING LSTM MODEL


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset 
df = pd.read_csv('Google_Stock_Price_Train.csv')
df_required = df.iloc[:,1:2]

# feature scaling 
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_required)

# formatting the trainig dataset with 60 timesteps
x_train = []
y_train =[]

for i in range(60 , 1258):
     x_train.append(df_scaled[i-60 :i , 0])
     y_train.append(df_scaled[i,0])

x_train = np.array(x_train)
y_train = np.array(y_train)

#formating the input into 3D array 
x_train = np.reshape(x_train  , (x_train.shape[0],x_train.shape[1],1))

# building the LSTM model
import keras 
from keras.models import Sequential
from keras.layers import Dense , LSTM , Dropout

stock_predictor = Sequential()
# first LSTM layer
stock_predictor.add(LSTM(units = 50 , return_sequences =True ,input_shape =(x_train.shape[1],1)))
stock_predictor.add(Dropout(0.2))
# second LSTM layer
stock_predictor.add(LSTM(units = 50 , return_sequences =True))
stock_predictor.add(Dropout(0.2))
#third LSTM layer 
stock_predictor.add(LSTM(units = 50 , return_sequences =True))
stock_predictor.add(Dropout(0.2))
# forth LSTM layer 
stock_predictor.add(LSTM(units = 50 , return_sequences =False))
stock_predictor.add(Dropout(0.2))
# output layer
stock_predictor.add(Dense(units = 1))

# compliling the model 
stock_predictor.compile(optimizer = 'adam' , loss='mean_squared_error')

# fit the model 
stock_predictor.fit(x_train , y_train , epochs= 100 , batch_size = 32 )

## ---check if the prediction is correct ---
# importing the test data 

df_test = pd.read_csv('Google_Stock_Price_Test.csv')
df_test_required = df_test.iloc[:,1:2].values

# get the predicted values 
df_predict = pd.concat((df.Open,df_test.Open), axis = 0)
df_predict = df_predict[len(df_predict) - len(df_test) - 60 :].values
df_predict = np.reshape(df_predict , (-1,1))
df_predict = scaler.transform(df_predict)

# creating the array and the prediction model 
x_test = []
for i in range(60 , 80):
    x_test.append(df_predict[i-60 :i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test , (x_test.shape[0],x_test.shape[1],1))

df_test_predicted = stock_predictor.predict(x_test)
df_test_predicted = scaler.inverse_transform(df_test_predicted)
