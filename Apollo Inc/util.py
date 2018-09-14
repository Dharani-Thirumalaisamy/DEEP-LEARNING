# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for Grappler autoparallel optimizer."""
# models here represented by combination of graph in tensor flow. ( uses the concept of metaGraph in tensor-flow)

from __future__ import absolute_import 
# python will look for the standard library's version for any module that is imported 
from __future__ import division
# python will change '/' operator to true division and use '//' operator when necessary
from __future__ import print_function
# brings print function from python 3 to python2.6+

import tensorflow as tf
# importing tensor-flow module 
from tensorflow.core.framework import variable_pb2
# importing API for the metaGraph that is to be used in this implementation which already has all the operations defined in it in .h files.
# metaGraph contains different fields from which the variable protocol buffer is imported. 
from tensorflow.core.protobuf import rewriter_config_pb2
# protobuf - serializing structured data 
# importing API to serialize data. RewriterConfig is a constructor in .proto file with all the descriptions.
# the above two APIs are required for Grappler method.

FLAGS = tf.flags.FLAGS
# pass runtime parameters
''' say this code is first run on the local machine and n number of epochs are used. If the model is used on server and if we want to increase the number
    epochs, this can be used.'''

# there are 2 states in metaGraph - exporting and importing a metaGraph
def export_state_tuples(state_tuples, name): # defining a function export_state_tuples ( exporting metaGraph) for 2 arguments
  for state_tuple in state_tuples: 
# adds (key , value) pair to the graph collection. c hidden state; h - output. name - key for the collection , state_tuple - value. 
    tf.add_to_collection(name, state_tuple.c) 
    tf.add_to_collection(name, state_tuple.h)
# the graph collection then keeps track of constructed graphs and how it is to be executed.

def import_state_tuples(state_tuples, name, num_replicas): # defining a function import_state_tuples with 3 arguments passed to it.
  restored = [] # initializing an array 
  for i in range(len(state_tuples) * num_replicas): # defining a range of numbers
    c = tf.get_collection_ref(name)[2 * i + 0] 
	# get the names from the stored collection. 
	# store the corresponding names to hidden state.
	# say i = 1 , so names[2] will be stored in cell.
    h = tf.get_collection_ref(name)[2 * i + 1]  
	# get the name from the stored collection.
	# store the corresponding names in output.
	# say i = 1 , name[3] ie the 4th name will be stored in hidden state.
    restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
	# tf.contrib.rnn.LSTMStateTuple(c,h) - stores two elements which is the hidden state and the output as tuple.
	# this tuple is appended to the array restored for range of i.
  return tuple(restored)
  # the output of this function is a tuple which contains two value.


def with_prefix(prefix, name):
# defines another function named with_prefix with 2 arguments passed to it.
  """Adds prefix to name."""
  return "/".join((prefix, name))
  # the function returns a output which joins prefix that is added to the name with '/' in between them.
  # say prefix - tensor , name - flow . it returns tensor/flow as output.  


def with_autoparallel_prefix(replica_id, name):
  return with_prefix("AutoParallel-Replica-%d" % replica_id, name)
# this is another function which takes two arguments and returns a output which works as input to the with_prefix function.
# two arguments which is replica_id , name are passed the with_prefix function.
# it joins two arguments and gives it to with_autoparallel_prefix function.
# now this function returns "AutoParallel-Replica-replica_id/name"
# this is implemented in Grappler/optimizer/auto_parallel.h file

class UpdateCollection(object):
# defining  a class named UpdateColection with an object named object.
  """Update collection info in MetaGraphDef for AutoParallel optimizer."""

  def __init__(self, metagraph, model):
  # defining the initializing function by passing two arguments to it.
    self._metagraph = metagraph # metagraph descriptions are passed to the _metagraph.
# replicate_states and update_snapshot_name are functions that are defined later 
    self.replicate_states(model.initial_state_name) # initial_state_name and final_state_name are defined later in the PTBModel in ptb_word_lm.py script.
    self.replicate_states(model.final_state_name) 
    self.update_snapshot_name("variables") # var_coll_name = variable now
    self.update_snapshot_name("trainable_variables") # var_coll_name = trainable_variables now

  def update_snapshot_name(self, var_coll_name): # function definition with 1 argument 
    var_list = self._metagraph.collection_def[var_coll_name]  # all the variable names from the metagraph is passed on to var_list
    for i, value in enumerate(var_list.bytes_list.value): 
      var_def = variable_pb2.VariableDef() 
      var_def.ParseFromString(value) # it doesn't return anything , but fills itself with the parsed data ( used with protobuf)
      # Somehow node Model/global_step/read doesn't have any fanout and seems to
      # be only used for snapshot; this is different from all other variables.
      if var_def.snapshot_name != "Model/global_step/read:0": # checks the snapshot_name in the var_def 
        var_def.snapshot_name = with_autoparallel_prefix(
            0, var_def.snapshot_name) # passing arguments to with_autoparallel_prefix function which update the snapshot_name with the output from with_autoparallel_prefix.
      value = var_def.SerializeToString() # serializing the data present in var_def and assigning it to value (used with protobuf) 
      var_list.bytes_list.value[i] = value # the value in the var_list's bytes_list method is updated with value in the previous line by incrementing i.

  def replicate_states(self, state_coll_name): # defining a function with one argument 
    state_list = self._metagraph.collection_def[state_coll_name] # assigning only the state column names to state_list from collection_def method in metagraph
    num_states = len(state_list.node_list.value) # length of nodes in the state_list  is assigned to num_states
    for replica_id in range(1, FLAGS.num_gpus): # the replica_id is looped from range of 1 to number of GPUs
      for i in range(num_states): # from the length of states 
        state_list.node_list.value.append(state_list.node_list.value[i]) # appending the value for the range to the value object in the node_list method in state_list
    for replica_id in range(FLAGS.num_gpus): # for the id in range of 0 to number of GPUs which is given as runtime parameter 
      for i in range(num_states): 
        index = replica_id * num_states + i # defining a new variable called index and assigning it with a value
        state_list.node_list.value[index] = with_autoparallel_prefix( #'''updating the value in the node list method in the state_list with the 
            replica_id, state_list.node_list.value[index])            # return function from the with_autoparallel_prefix which returns a string'''


def auto_parallel(metagraph, model): # function defined which is used in the ptb_word_lm.py script.
  from tensorflow.python.grappler import tf_optimizer # importing the Grappler/optimizer/auto_parallel
  rewriter_config = rewriter_config_pb2.RewriterConfig() # calling the RewriterConfig description file from profile buffer
  rewriter_config.optimizers.append("autoparallel") # appending the 'autoparallel' method to the optimizer method in rewriter_config constructor
  rewriter_config.auto_parallel.enable = True # enabling the suto_parallel optimizer in the description file
  rewriter_config.auto_parallel.num_replicas = FLAGS.num_gpus ''' flags which is a runtime parameter , can change the number of GPU's used which is assigned 
                                                                to num_replicas '''
  optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph) # all the changes that is made in the previous lines are stored in optimized_graph.
  metagraph.graph_def.CopyFrom(optimized_graph) # the description changes are copied from optimized_graph to metagraph.
  UpdateCollection(metagraph, model) # arguments are passed to UpdateColection class.
   # model is assigned  in the PTBModel in the ptb_word_lm.py script.