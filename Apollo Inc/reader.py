# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import 
# python will look for the standard library's version for any module that is imported 
from __future__ import division
# python will change '/' operator to true division and use '//' operator when necessary
from __future__ import print_function
# brings print function from python 3 to python2.6+


import collections # collections contains all the data-types in python which is imported as a module
import os # importing this module so that the local machine Operating system can be used especially when it comes to file storing or loading
import sys # this module provides access to modules that are used by the interpretor

import tensorflow as tf # importing the tensor-flow module as tf

Py3 = sys.version_info[0] == 3 # checking for python version and then storing the boolean result in Py3 variable.

def _read_words(filename): # function is defined with filename as the argument 
  with tf.gfile.GFile(filename, "r") as f: # # the filename is given the permission to write with file I/O wrappers without thread locking 
	  # tf.gfile.GFile provides an API for python file objects and an API for c++ file system. 
	  # C++ file system supports multiple file implementation system such as local files , GCP , Hadoop's HDFS. 
	  # tf uses all these functions in tf.gfile format to load or save the file. It can also access files form other users.
      #	If in case all the files are local , then there is no use of this.  
    if Py3: # if Python version used is 3
      return f.read().replace("\n", "<eos>").split()
	  # this line of code reads the .txt files from the path and replaces '\n'(space) with '<eos>' , splits the string into lists and returns the list
    else: # if python version used in not 3 and above this part of the code gets executed 
      return f.read().decode("utf-8").replace("\n", "<eos>").split()
		''' in this first the .txt file that is read is decoded using 'utf-8' , mainly for the character that might be present in the .txt file 
		 which will not be recognized by the compiler(Converts characters to text). Once it's decoded , the same procedure as the above 
		 takes place and the list is returned''' 

def _build_vocab(filename): # another function with filename as argument . filename is passed by the ptb_raw_data function (train_path) 
  data = _read_words(filename) # calls the _read_words function with filename as the argument passed on to it.
  # the list that is returned is passed stored in the variable called data.

  counter = collections.Counter(data) # collections has a class called counter which counts the occurrences of words.
  # here the counter data type is used to count the number of times each word occurs in the list( IE., data) and stores it in counter
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  # this line of code gives list data type and sorts the words according to their number of occurrences and alphabetically order. 
  words, _ = list(zip(*count_pairs)) # this converts the list to tuple that contains only the words in the descending order of their occurrences and alphabetically order. 
  word_to_id = dict(zip(words, range(len(words))))
  # this line of code converts the tuple into a dictionary with key as the word and the value as the index of the word.
  return word_to_id # returns a dictionary of (key,value) pair
  ''' for example :
     data = ['apple','apple','apple','apple','green','red','red','red','red','orange','orange','orange',]
	
	counter = collections.Counter(data) - gives a counter data type 
	
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) 
	o/p : [('apple', 4), ('red', 4), ('orange', 3), ('green', 1)]
	
	words, _ = list(zip(*count_pairs))
	o/p : ('apple', 'red', 'orange', 'green')
	
	word_to_id = dict(zip(words, range(len(words))))
	o/p : {'apple': 0, 'red': 1, 'orange': 2, 'green': 3}
  '''

def _file_to_word_ids(filename, word_to_id): # function with 2 arguments
  data = _read_words(filename)# calls the _read_words function with filename as the argument passed on to it.
  # the list that is returned is passed stored in the variable called data.
  return [word_to_id[word] for word in data if word in word_to_id] # this line of code converts the words to numbers. 
  # package used in Natural Language Processing 

def ptb_raw_data(data_path=None): # the data_path argument in this function is given by the tmpdir in the reader_test.py file

  train_path = os.path.join(data_path, "ptb.train.txt") # "ptb.train.txt" is joined with the path that tmpdir holds and the text file in that is stored in train_path
  valid_path = os.path.join(data_path, "ptb.valid.txt") # "ptb.valid.txt" is joined with the path that tmpdir holds and the text file in that is stored in valid_path
  test_path = os.path.join(data_path, "ptb.test.txt")# "ptb.test.txt" is joined with the path that tmpdir holds and the text file in that is stored in test_path

  word_to_id = _build_vocab(train_path) # the train.txt path is given as the argument to _build_vocab function which returns a dictionary and that is stored in word_to_id variable.
  train_data = _file_to_word_ids(train_path, word_to_id)# the train.txt path is passed to the _file_to_word_ids function along with word_to_id which converts words to numbers.  
  valid_data = _file_to_word_ids(valid_path, word_to_id)# similarly for valid_data and test_data
  test_data = _file_to_word_ids(test_path, word_to_id) 
  vocabulary = len(word_to_id) # stores the length of the dictionary
  return train_data, valid_data, test_data, vocabulary # returns all numeric values (which is stored in the output variable in reader_test.py also used in main function.)


def ptb_producer(raw_data, batch_size, num_steps, name=None): # function defined with 4 arguments. used in reader_test.py file
 
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
  # name_scope simply acts as prefix to the name.
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
	# converting the raw_data from the reader_test script into tensor of data type int.

    data_len = tf.size(raw_data) 
	# storing the size of the raw_data
    batch_len = data_len // batch_size 
	# from __future__ import division implies here floor division is done. 
    data = tf.reshape(raw_data[0 : batch_size * batch_len], 
                      [batch_size, batch_len])
	# reshaping the raw_data using corresponding values of the variable.

    epoch_size = (batch_len - 1) // num_steps 
	# floor division
    assertion = tf.assert_positive(    
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
	# adding dependency to an operation. Checking for boolean value . If it is positive assertion : 1 else 0.
    with tf.control_dependencies([assertion]): 
	# if assertion is 1 IE,. true then the next line of code gets executed.
      epoch_size = tf.identity(epoch_size, name="epoch_size") 
	# tf.identity is usually used to switch tensors between GPUs and CPUs but in this case tf.identity is used as a dummy node which makes the epoch_size 
	# tensor to be executed after the execution of the assertion and store that value in epoch_size.

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
	# this line of code produces numbers between the range (0 , epoch_size) without the numbers being shuffled.
	# dequeue here makes sure that number gets appended on either side of the list and can be removed from either side too.
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
	# strided_slice in tensor-flow is used for slicing a tensor based on three values which is begin (here ,= [0, i * num_steps])
	# end (here,=batch_size) , stride(here,=(i + 1) * num_steps). 
	# what it does is that it moves through the tensor with the stride that is specified , starting from the begin position and ending at end.
	# this gives the value of x.
	''' for example :
	consider a 3D tensor :
	tensor = [[[1,2,3],[4,5,6],[7,8,9]],
			  [[10,11,12],[13,14,15],[16,17,18]],
			  [[19,20,21],[22,23,24],[25,26,27]]]
	tensor_x = tf.strided_slice(tensor , begin=(0,0,0),end=(3,3,3),stride=(2,2,2))
	so tensor_x will be the values present at the position : (0,0,0) ,(0,2,0),(0,2,2),(2,0,0),(2,0,2),(2,2,0),(2,2,2)
	This can be considered as 3 for loop with condition : i=0,i<3,i+2 (first loop) ; j=0,j<3,j+2(second loop) ; k=0,k<3,k+2(third loop)
	'''
	
    x.set_shape([batch_size, num_steps])
	# reshaping the x tensor.
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
	# same execution as x 
    y.set_shape([batch_size, num_steps])
	# reshaping the tensor.
    return x, y
	# returning x and y tensor 
