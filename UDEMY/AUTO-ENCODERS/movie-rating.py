# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:13:18 2018

@author: Dell-pc
"""
import pandas as pd 
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# importing datasets , training and test set
movies = pd.read_csv('ml-1m/movies.dat' , sep ='::' , header = None , engine = 'python',encoding = 'latin-1' )
users =  pd.read_csv('ml-1m/users.dat' , sep ='::' , header = None , engine = 'python',encoding = 'latin-1' )
ratings =  pd.read_csv('ml-1m/ratings.dat' , sep ='::' , header = None , engine = 'python',encoding = 'latin-1' )
train  = pd.read_csv('ml-100k/u1.base' ,delimiter = '\t')
train = np.asarray(train)
train = train.fillna(0)
test  = pd.read_csv('ml-100k/u1.test' , delimiter = '\t')
test = np.asarray(test)

# getting the nummber of users and movies 
n_movies =  int(max(max(train[:,1]) , max(test[:,1])))
n_users = int(max(max(train[:,0]) , max(test[:,0])))

# rows - users ; columns - movies 
def formatting(dataset):
    final_list=[]
    for i in range (1 , n_users+1):
        movie_column = dataset[:,1][dataset[:,0] == i]
        ratings_row = dataset[:,2][dataset[:,0] == i]
        ratings = np.zeros(n_movies)
        ratings[movie_column - 1] = ratings_row
        final_list.append(list(ratings))
    return final_list
    
train_list = formatting(train)
test_list = formatting(test)

# converting them into torch tensors
train_tensor = torch.FloatTensor(train_list)
test_tensor = torch.FloatTensor(test_list)

# the architecture
class SAE(nn.Module):
    def __init__(self,):
        super(SAE , self).__init__() # to use all the modules from the module class
    # full connectio between the input vectors and a hidden layer
        self.layer1 == nn.Linear(n_movies,20)  #it will detect 20 features 
        self.layer2 == nn.Linear(20,10)     # it will detect 10 more features based on the first 20 nodes
        self.layer3 == nn.Linear(10,20)     # starting to decode , so making it symmetric
        self.layer4 == nn.Linear(20,n_movies)
        self.activation == nn.Sigmoid()
    def forward(self , x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x)) # started decoding 
        x = self.layer4(x) # last fully connected layer 
        return x 
    
sae = SAE()
loss_function = nn.MSELoss() 
optimizer = optim.RMSprop(sae.parameters() , lr = 0.1 , weigh_decay = 0.5)

# Trainig the model 
nb_epochs  = 10
for n in range(1 , nb_epochs+1):
    training_loss = 0
    counter = 0.
    for n in range(n_users):
       input = Variable(train_tensor[n]).unsqueeze(0) # to create a new dimension  0 - index of the dimension
       target = input.clone()
       if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0 
            loss = criterion(output ,target)
            corrector = n_movies/float(torch.sum(target.data >0) + le-10) # le -10 :will not create any bias and also will take sure that the denominator is not zero.
            loss.backward()
            training_loss += np.sqrt(loss.data[0]*corrector)
            counter += 1.
            optimizer.step()
     print('epoch :' str(nb_epoch) + 'loss :' str(training_loss/counter))
     
# Testing the model 

test_loss = 0
counter = 0.
for n in range(n_users):
    input = Variable(train_tensor[n]).unsqueeze(0) # to create a new dimension  0 - index of the dimension
    target = Variable(test_tensor[n])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0 
        loss = criterion(output ,target)
        corrector = n_movies/float(torch.sum(target.data >0) + le-10) # le -10 :will not create any bias and also will take sure that the denominator is not zero.
            #loss.backward()
        test_loss += np.sqrt(loss.data[0]*corrector)
        counter += 1.
            #optimizer.step()
print('loss :' str(training_loss/counter))
