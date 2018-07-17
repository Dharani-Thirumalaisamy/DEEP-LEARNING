
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from minisom import MiniSom

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
