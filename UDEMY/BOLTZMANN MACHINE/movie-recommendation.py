# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 01:51:03 2018

@author: Dell-pc
"""
# importing the necessary libraries - pytorch
import torch 
import numpy as np
import pandas as pd 
import torch.nn as nn           #  pytorch library for neural networks
import torch.nn.parallel        # for parallel computing 
import torch.optim as optim     # for optimization
import torch.utils.data         
from torch.autograd import Variable  # for optimizer

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

train_tensor[train_tensor == 0] = -1
train_tensor[train_tensor == 1] = 0
train_tensor[train_tensor == 2] = 0
train_tensor[train_tensor >= 3] = 1

test_tensor[test_tensor == 0] = -1
test_tensor[test_tensor == 1] = 0
test_tensor[test_tensor == 2] = 0
test_tensor[test_tensor >= 3] = 1

# The architucture
class RBM():
    def __init__ (self,vn,hn):
        self.w = torch.randn(vn,hn)
        self.a = torch.randn(1,vn)
        self.b = torch.randn(1,hn)
    def hidden_sampling(self , x):
        wx = torch.mm(x,self.w.t())
        activation = wx +self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v , torch.bernoulli(p_h_given_v)
    def visible_sampling(self , y):
        wy = torch.mm(y,self.w)
        activation = wy +self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h , torch.bernoulli(p_v_given_h)
    def training(self , v0 , h0 , vk,hk):
        self.w += torch.mm(v0.t(),h0) - torch.mm(vk.t(),hk)
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((h0-hk),0)
        
vn = len(train_tensor[0])
hn = 100
batch_size = 100
model = RBM()

# Trainig the model 
nb_epochs  = 10
for n in range(1 , nb_epochs+1):
    training_loss = 0
    counter = 0.
    for n in range(0,n_users-batch_size , batch_size):
        vk = train_tensor[n:n+batch_size]
        v0 = train_tensor[n:n+batch_size]
        h0,_ = model.hidden_sampling(v0)
        for k in range(10):
            _,hk =model.hidden_sampling(vk)
            _.vk = model.visible_sampling(hk)
            vk[v0<0] = v0[v0<0]
        hk,_ = model.hidden_sampling(vk)
        model.training(v0,vk,h0,hk)
        training_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        counter +=1.
    print('epoch :' str(nb_epoch) + 'loss :' str(training_loss/counter))
    
# testing the model
test_loss = 0
counter = 0.
for n in range(n_users):
    v_input = train_tensor[n:n+1]
    v_target = test_tensor[n:n+1]
    #h0,_ = model.hidden_sampling(v_input)
    if len(v_target[v_target>=0]) > 0:
        _,h =model.hidden_sampling(v_input)
        _.v_input = model.visible_sampling(h)
        #vk[v0<0] = v0[v0<0]
    #hk,_ = model.hidden_sampling(vk)
    #model.training(v0,vk,h0,hk)
        test_loss += torch.mean(torch.abs(v_target[v_target>=0] - v_input[v_target>=0]))
    counter +=1.
print('loss :' str(test_loss/counter))

        
    
