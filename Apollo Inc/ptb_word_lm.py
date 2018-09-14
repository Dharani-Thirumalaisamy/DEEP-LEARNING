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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""

from __future__ import absolute_import 
# python will look for the standard library's version for any module that is imported 
from __future__ import division
# python will change '/' operator to true division and use '//' operator when necessary
from __future__ import print_function
# brings print function from python 3 to python2.6+

import time
# time module which uses the local machine's time

import numpy as np
# numpy module for various execution

import tensorflow as tf
# calling tensorflow module

import reader
# calling the reader module (reader.py)
import util
# calling the util module(util.py)


from tensorflow.python.client import device_lib
# lists out the available devices in the local process (GPUs if necessary)

flags = tf.flags
#argument that can be passes during runtime
logging = tf.logging
# keeps track of when the execution occurs with time stamp

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
# flag.DEFINE_string(name,default , help) name - what the flag is called , default value is small and the third argument is help which describes the way the flag is called.
# flag.DEFINE_integer(batch size , data to be processed at a time , help)
# flag.DEFINE_bool(what to use , default,help)
FLAGS = flags.FLAGS
#argument that can be passes during runtime(command line)
BASIC = "basic" # these three are different rnn_modes in which the program is executed.
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32
 # this defines a function called data_type which returns float16 value if the argument passed is use_fp16 which is 16 bit float(for GPUs computations)
 # or else returns float32 value.


class PTBInput(object): # class is defined 
  """The input data."""

  def __init__(self, config, data, name=None): 
  # class has a init method which is the initialization function
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
	# the hyper-parameters are assigned values using config method which is instantiated in the main function .
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
	# the ptb_producer returns x and y value which is stored in the input_data and targets variable respectively.

# this class gives the input data necessary to run the model.

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_): 
  # method with 3 arguments where is_training is a boolean value , config = config which is defined in main function and input_ keeps changing)
    self._is_training = is_training # value of is_training stored in _is_training
    self._input = input_ # data in input_ stored in _input
    self._rnn_params = None # hynper-parameters
    self._cell = None # hyper-parameters
    self.batch_size = input_.batch_size # batch_size , num_steps , size and vocab_size is assigned by the config function.
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"): # running on cpu with one thread
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
	# the above line of code is used to create a variable named 'embedding' of size = [vocab_size,size] and of the data type returned by the data_type() function.
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
	# embedding_lookup performs parallel lookup on the tensors here embedding. It returns the elements of embedding based on the IDs provided by the 
	# input_.input_data.

    if is_training and config.keep_prob < 1: # if is_training is true and config.keep_prob is < 1 which makes the if statement true , then the next line 
	# of code will be executed
      inputs = tf.nn.dropout(inputs, config.keep_prob) # keep_prob - controls dropout layer
	# inputs is assigned a dropout layer in the LSTM model
    output, state = self._build_rnn_graph(inputs, config, is_training)
	# the previous line of code executes the _build_rnn_graph method which based on the mode executes either _build_rnn_graph_cudnn or _build_rnn_graph_lstm method.
	# both the function outputs output and state which is stored in the output and state variable defined in this function respectively.
    
	# softmax and the loss function is defined
	# y = wx+b
	softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
	# creating a variable named 'softmax_w' of size = [size, vocab_size] and of data type that is returned in the data_type() function.
	# softmax_w - weight 
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
	# # creating a variable named 'softmax_b' of size = [size, vocab_size] and of data type that is returned in the data_type() function.
	# softmax_b - biases
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
	# logits = y (final output)
	# y = xw_plus_b = wx+b which is calculated by taking the output from the last but one layer multiplying the weights and adding bias to it. 
 
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
	# Reshape logits to be a 3-D tensor for sequence loss
    
	# Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits, # the final output
        input_.targets, # tensors at different time stamp 
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()), # tensors with all elements set to 1 
        average_across_timesteps=False, #sum the cost across the sequence dimension and divide the cost by the total label weight across time steps.
        average_across_batch=True) #sum the cost across the batch dimension and divide the returned cost by the batch size.
	# loss function calculated here is weighted cross_entropy
	# it returns a float tensor of rank 0 or 1 or 2 depending on average_across_timesteps and average_across_batch.
		
    # Update the cost
    self._cost = tf.reduce_sum(loss) # sums up the loss along all its dimensions and stores it in _cost
    self._final_state = state # state returned by the _build_rnn_graph function is stored in _final_state

    if not is_training: # if is_training is false , the program pointer returns back to where this class was initiated.
      return            # If training is true , we go ahead and create the entities required to train the RNN network such as learning_rate,optimizer etc.

	# defining learning_rate 
    self._lr = tf.Variable(0.0, trainable=False)
	# creating a variable with initial value 0.0 
    tvars = tf.trainable_variables()
	# store all the trainable parameters in tvars.
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
	# calculates gradients for all the elements in tvars and also clips the gradients to a certain value defined in config.max_grad_norm.
	# this is usually done to address the exploding the gradient issue.
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
	# defining Stochastic Gradient Descent optimizer to update the weights and bias.
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())
	# creating a class member train_op to update the weights in tvars corresponding to the gradients stored in grads.
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
	# creating a placeholder to get the learning_rate as user input.
    self._lr_update = tf.assign(self._lr, self._new_lr)
	#updating the learning rate with the one specified by the user 
	
	# building a function to build rnn graph based on the rnn mode.
  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)
	

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2]) 
	# takes the inputs from the _build_rnn_graph and permutes according to the perm (here , =[1,0,2]) and stores it in input
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
	# defining a LSTM cell which is CUDNN compatible 
    params_size_t = self._cell.params_size()
	# the parameter size of the cell is stored in params_size_t variable	
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
	# initializing a variable named 'lstm_params' with initial values sampled randomly with uniform probability distribution and storing it in _rnn_params.
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
	# creating tensor c and tensor h of size [batch_size,hidden_size] and of type float32.
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
	# stores c and h which is hidden state and output  respectively in the variable _initial_state
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
	# calling the CudnnLSTM and passing the parameters such as inputs , h , c , rnn_params and is_training.
	# it returns output and states which is stored in outputs , h and c.
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
	# taking transpose of the output and reshaping it.
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
	# function returns output and states(h and c)

  # function defined o build a LSTM architecture when rnn mode is BASIC or BLOCK
 def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
	# if rnn_mode in config is BASIC the below line of code gets executed
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
	# the above line of code returns a basic LSTM layer and instantiates variables to all gates
	# forget bias is added in-order to reduce the scale of forgetting during the training. usually it is 1.
    if config.rnn_mode == BLOCK:
	# if rnn_mode in config is BLOCK , the below line of code gets executed.
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
	# this returns a block lstm-layer and instantiates variables to all gates.
	# forget bias is added in-order to reduce the scale of forgetting during the training. usually it is 1
	# the difference between BASIC and BLOCK is , BLOCK uses a fused kernal which results in better performance.
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)
	# if the mode given in config doesn't match any of the following , then ValueError will be raised.

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training) 
	  # it calls the _get_lstm_cell function which returns either BasicLSTMCell or BLOCK cell based on the mode rnn mode specified in config.
      if is_training and config.keep_prob < 1: 
	  # if is_training is True and keep_prob is less than 1 , then the below lines of code will be executed.
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
	# it adds dropout to the input and output cell. The cell here is updated with the dropout layer. 
	# keep_prob value is the one that controls the dropout rate in the cell. 
      return cell
	  # this function returns the updated cell.

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
	# if the IF condition is false , then the program counter skips the IF statement and executes this line of code.
	# in this line of code , cell that is returned by the _get_lstm_cell is stacked with multiple layers of simple cells. The number of cells that is 
	# sequentially stacked is given by the parameter num_layers in config method.
	# the cell is updated with the stacked layer.

    self._initial_state = cell.zero_state(config.batch_size, data_type())
	# the updated cell has an object named zero_state whose value is stored in the _initial_state variable
	# here it means , the initial state remains a constant zero vector.
    state = self._initial_state
	# the _initial_state value is passed on to state variable here.
    
	# Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    
	outputs = []
	# an empty list name outputs is defined.
    with tf.variable_scope("RNN"):
	# creating namespace for both operators and variables. Here 'RNN' will act as prefix.
      for time_step in range(self.num_steps):
	  # this loop works for the range of number of time steps that is defined 
        if time_step > 0: tf.get_variable_scope().reuse_variables() 
		# the above line of code will always return None as result. Its only function is to set the attribute reuse to True.
        (cell_output, state) = cell(inputs[:, time_step, :], state)
		# values from the inputs of the cell are stored in cell_output and the value of state in the LSTM cell is stored in the state variable.
        outputs.append(cell_output)
		# the cell_output is stored in the outputs array.
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
	# reshaping the output and updating the output.
    return output, state
	# this function returns the output and state variable.

	
	# this is function which takes session and lr_value( which is the learning rate) as arguments
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
	# the operation _lr_update is executed in this session and lr_value is the value that is passed to the _new_lr placeholder. 

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
	# value stored in name is passed on to _name.
    ops = {util.with_prefix(self._name, "cost"): self._cost}
	# calls the with_prefix function in util.py.
	# It joins the prefix and name with '/' in between and returns that value.
	# That value acts as key and the value to that is given by _cost.
	# This dictionary is stored in ops variable.
    
	if self._is_training:
	# if _is_training is true the below lines of code gets executed which implies these lines of code are part of training the model.
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
	  # ops gets updated with new values from the model.
      if self._rnn_params: 
	  # _rnn_params is not None
        ops.update(rnn_params=self._rnn_params)
		# the params in ops gets updated with _rnn_params
    for name, op in ops.items():
	# for name and op items in ops
      tf.add_to_collection(name, op)
	  # name and op are added to the graph collection 
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
	# Above two lines of code calls with_prefix function from util script and returns a value which has _name as prefix to 'initial' and 'final'.
	# They are stored as initial_state_name and final_state_name respectively.
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)
	#the above two lines of code calls the export_state_tuples function from utils and passes the arguments to the function.
	# these arguments that is _initial_state_name and _final_state_name are added to the graph as name and c state from initial_state and h from final_state are added to the graph. 

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
	# is _is_training is true the below lines of code gets executed which implies these lines of code are part of training the model.
      self._train_op = tf.get_collection_ref("train_op")[0] # train_op contains the updated weight in tvars corresponding to the gradients stored in grads.
      # the value at the first position is stored to _train_op.
	  self._lr = tf.get_collection_ref("lr")[0]
	  # 'lr' hold the learning rate for the model which is stored to _lr.
      self._new_lr = tf.get_collection_ref("new_lr")[0]
	  # 'new_lr' holds the new learning rate given as user input in the collection from which the value is stored to _new_lr.
      self._lr_update = tf.get_collection_ref("lr_update")[0]
	  # 'lr_update' holds the value of updated learning rate and now that value is stored to _lr_update
      rnn_params = tf.get_collection_ref("rnn_params")
	  # all the parameters from the model is stored in rnn_params which is now stored to rnn_params.
      if self._cell and rnn_params:
	  # _cell anf rnn_params holds some value when rnn mode is defined , then the below lines of code will be executed. Those two are hyper-parameters
	  # which are initially given a value of None and takes value only when rnn_mode is CUDNN.
        
		params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable( # creating a RNNParamsSaveable object - used for saving the weights and parameters in canonical format.
            self._cell,  # _cell is the LSTM cell that is created in CudnnLSTM method in _build_rnn_graph_cudnn function.
            self._cell.params_to_canonical, # this is a function that converts the params from one format to canonical form.
            self._cell.canonical_to_params, # this function converts canonical form to a specific format.
            rnn_params, # parameters stored in CUDNNLSTM cell.
            base_variable_scope="Model/RNN") # used as part of prefix of names.
			# all these are saved to params_saveable .
			
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
		# tf.GraphKeys is a library which is used to save values associated with graphs , here the params_saveable object is added to the graph collection
		# in such a way that it can be retrieved later.
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
	# with_prefic function returns prefix/name here _name/cost. So _name[0] value is stored to _cost object.
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
	# assigning values to num_replicas. If _name is 'Train' (IE,. train.txt) then number of replicas will be equal to number of GPUs , else it is considered as 1 .
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)
	'''The above two lines of code calls the import_state_tuples function from util passing on the required parameters to that function.
		The function takes in values passed by these parameters and returns a tuple of (c,h) which is stored as _initial_state and _final_state'''

		
	# property is nothing but a class which usually calls the __init__ function but sometimes it acts as getter and setter. 
	# here property means :
	# for example :  def input(self): return self._input
	#                input = property(input)
	
  @property
  def input(self):                          
    return self._input # class the input function and returns the value as _input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

# All the other functions also word the same way like the input function.


''' Three different class SmallConfig , MediumConfig ,LargeConfig ,TestConfig which lists out the hyperparameters of the LSTM model.'''
class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK

 """Runs the model on the given data."""
def run_epoch(session, model, eval_op=None, verbose=False):
  start_time = time.time() 
  # stores the system time and not the cpu time
  costs = 0.0 
  # initially cost and iters is initialized to 0.
  iters = 0
  state = session.run(model.initial_state) 
  # the operation initial_state runs in this session and the output which is _initial_state is stored in state variable.

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  # dictionary named fetches stores the cost and final_state. 
  
  if eval_op is not None: 
    fetches["eval_op"] = eval_op # adding third key:value pair to the dictionary if eval_op is assigned a value.

  for step in range(model.input.epoch_size): # for the number of steps in range epoch_size from the input object of model which is equal to m which in return is the output of PTB Model
    feed_dict = {} # initializing a dictionary 
    for i, (c, h) in enumerate(model.initial_state): # for loop for storing the key value pair of the 2 states that is the result of initial state in the PTBModel
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
# the operation fetches is executed in this session for which key,value pair in feed_dict is passed as parameters and stored to vals
    cost = vals["cost"] # cost in the vals is stored to cost 
    state = vals["final_state"] # final_state in vals is stored to state

    costs += cost # incrementing cost by cost times
    iters += model.input.num_steps # incrementing iterations by number of steps in the PTBModel times

    if verbose and step % (model.input.epoch_size // 10) == 10: # if this statement is true print the below statement 
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))
	# the print statement prints the perplexity which is the measure of uncertainty in the prediction model , 
	# speed of the process and wps - words per second (predicted words)

  return np.exp(costs / iters)
  # returns exponential function


def get_config(): # defining config 
  """Get model config."""
  config = None # initially given a value of None
  if FLAGS.model == "small": # if the user inputs small then SmallConfig parameters defined in class SmallConfig are taken for the model
    config = SmallConfig()
  elif FLAGS.model == "medium": #if the user inputs medium then MediumConfig parameters defined in class MediumConfig are taken for the model
    config = MediumConfig()
  elif FLAGS.model == "large":#if the user inputs large then largeConfig parameters defined in class LargeConfig are taken for the model
    config = LargeConfig()
  elif FLAGS.model == "test":#if the user inputs test then testConfig parameters defined in class TestConfig are taken for the model
    config = TestConfig()
  else: # if none of the above input is given by the user  , a ValueError will be raised .
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode: # if rnn mode is set to true then the config model is also se to true
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" : # if the version of python is <1.3 or number of GPUs is not 1 then BASIC LSTM cell will be created.
    config.rnn_mode = BASIC
  return config
  # returning the config method.


def main(_): # this is the main function which controls the whole implementation
  if not FLAGS.data_path: # if no path is given , the data will not be accessible , so it raises a ValueError .
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [ 
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ] # getting the list of local devices available for running this program and checking if the device type is GPU. If it is GPU , then it is stores in the gpus array.
  if FLAGS.num_gpus > len(gpus): # if the number of gpus taken as input is greater than the number of gpus present in the local machine
    raise ValueError(   # a value error will be raised asking the user to input a lesser number as the local machine has gpu less than the number given by the user.
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  raw_data = reader.ptb_raw_data(FLAGS.data_path) 
  # calls the ptb_raw_data function from reader script. The function returns train_data,test_data,valid_data and vocabulary which stores numeric values of the txt data 
  train_data, valid_data, test_data, _ = raw_data 
  # storing train, test and validation data in train_data, test_data and valid_data respectively and discarding vocabulary
  
  config = get_config() # calling the get_config function and storing the config value in config.
  eval_config = get_config() # calling the get_config function and storing the config value in eval_config.
  eval_config.batch_size = 1 # assigning batch_size and number of steps in eval_config to 1.
  eval_config.num_steps = 1

  with tf.Graph().as_default(): # creates a new graph and places everything in that graph 
    initializer = tf.random_uniform_initializer(-config.init_scale, 
                                                config.init_scale)
	# its a initializer which generates tensor with uniform distribution with min value as -config.init_scale and maximum value as config.init_scale.

    with tf.name_scope("Train"): # Train acts as prefix to all the operators 
      train_input = PTBInput(config=config, data=train_data, name="TrainInput") # creating an object "train_input" for PTBInput class which stores 
	  #batch_size, step size, number of epochs, input_data and its corresponding targets as its members
      
	  with tf.variable_scope("Model", reuse=None, initializer=initializer): 
	  # Model here is a prefix for all the variables and operators
        m = PTBModel(is_training=True, config=config, input_=train_input) # creating an object "m" for PTBModel class which creates the model.
      tf.summary.scalar("Training Loss", m.cost) # creating summary for the cost that can be written to the file writer to be later used for visualizing the loss
      tf.summary.scalar("Learning Rate", m.lr) # creating summary for learning rate 

    with tf.name_scope("Valid"): # doing the same operation as above but without training the model and using the same weights learnt
	#above to evaluate the performance of the network on the validation data.
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"): # doing the same operation as above but without training the model and using the same weights learnt
	#above to evaluate the performance of the network on the test data.
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest} # creating a dictionary of models.
    for name, model in models.items(): # iterating through the models dictionary
      model.export_ops(name) # calling each of the model's (m,mvalid,mtest) export_ops method to add all the operations to the tensorflow graph collection
    metagraph = tf.train.export_meta_graph() # saving the graph information created above in the metagraph variable 
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1: # checking for tensorflow version and the number of GPUs given as user input
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0") # raising ValueError.
    soft_placement = False # assigning boolean value to a variable
    if FLAGS.num_gpus > 1: # checking for the number of GPUs
      soft_placement = True # updating the soft_placement variable based on condition
      util.auto_parallel(metagraph, m) # calling the auto_parallel function in util script to rewrite the configurations to better optimize the 
	  # execution of the model based on the number of GPUs available. 

  with tf.Graph().as_default(): # creates a new graph  
    tf.train.import_meta_graph(metagraph) # places the metagraph created above in the new graph 
    for model in models.values(): # iterating through the models dictionary
      model.import_ops() # calling the import_ops method 
    sv = tf.train.Supervisor(logdir=FLAGS.save_path) # creating a tensorflow supervisor object to monitor and save the model checkpoint at FLAGS.save_path 
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement) # it checks for the value of soft_placement and it it is true it places an op in CPU provided there are no GPUs.
    with sv.managed_session(config=config_proto) as session: # checks if the model is initialized and creates a session if it is true.
      for i in range(config.max_max_epoch): # iterating through the epochs
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0) # calculating the learning rate decay
        m.assign_lr(session, config.learning_rate * lr_decay) # calling the assign_lr method for model m 

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr))) # running the lr operation for model m, calculating Epoch, learning rate  and printing it.
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
		# the uncertainty in the training model is calculated by running the run_epoch function.
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))# printing the train_perplexity for each epoch
        valid_perplexity = run_epoch(session, mvalid)
		# the uncertainty in the validation model is calculated by running the run_epoch function.
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
		# printing the validation_perplexity for each epoch
		
      test_perplexity = run_epoch(session, mtest)
	  # the uncertainty in the test model is calculated by running the run_epoch function.
      print("Test Perplexity: %.3f" % test_perplexity)
	  # printing the test_perplexity 

      if FLAGS.save_path: # if there exist a value for FLAGS.save_path , the model is saved 
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run() # it calls the main() function passing the command line FLAG arguments  
