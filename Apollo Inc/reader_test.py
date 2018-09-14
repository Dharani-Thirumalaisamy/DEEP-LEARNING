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

"""Tests for models.tutorials.rnn.ptb.reader."""

from __future__ import absolute_import 
# python will look for the standard library's version for any module that is imported 
from __future__ import division
# python will change '/' operator to true division and use '//' operator when necessary
from __future__ import print_function
# brings print function from python 3 to python2.6+

import os.path
# for pathname manipulations
import tensorflow as tf
# tensorflow module 
import reader
# import reader.py script 

class PtbReaderTest(tf.test.TestCase): '''class ptbReaderTest in which tf.test.Testcase is called. This tf.test.TestCase is a method inherited from 
										unittest.TestCase used for unit testing in tensorflow. '''

  def setUp(self):                   
    self._string_data = "\n".join(  # a function in which _string_data is assigned a value
        [" hello there i am",
         " rain as day",
         " want some cheesy puffs ?"])

  def testPtbRawData(self): 
    tmpdir = tf.test.get_temp_dir() # tf.test.get_temp_dir() returns a temporary directory which is to be used during the testcase. This is assigned to tmpdir.
    for suffix in "train", "valid", "test": # giving values to suffix
      filename = os.path.join(tmpdir, "ptb.%s.txt" % suffix) ''' creating a path where,the path in which this program is running on the local machine is
																joined with temporary directory that is created and with ptb.suffix.txt
																 eg. c:/users/dell-pc/desktop/start_kernalvdhfgd/ptb.test.txt 
																 where start_kernalvdhfgd is tmpdir.'''
      with tf.gfile.GFile(filename, "w") as fh: # the filename is given the permission to write with file I/O wrappers without thread locking 
	  # tf.gfile.GFile provides an API for python file objects and an API for c++ file system. 
	  # C++ file system supports multiple file implementation system such as local files , GCP , Hadoop's HDFS. 
	  # tf uses all these functions in tf.gfile format to load or save the file. It can also access files form other users. If in case all the files are local , then there is no use of this.  
        fh.write(self._string_data) # writing the contents stored in _string_data in that file ( filename)
    # Smoke test - checking the critical functionalities of this program
    output = reader.ptb_raw_data(tmpdir) # calls the ptb_raw_data function from reader.py which returns a tuple of train_data, valid_data, test_data and vocabulary.
    # that tuple is passed to output in this program.
	self.assertEqual(len(output), 4) # assertEqual is called to check for the expected result in unit test python framework.
	# here is it used to check if the length of the output is equal to 4 which is train _data , test_data , valid_data and vocabulary.
  
  def testPtbProducer(self): # another function 
    raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1] # assigning an array of integers to a variable named raw_data
    batch_size = 3 # assigning values to batch size and num_steps which are hyper-parameters to ptb_producer function in reader.py script
    num_steps = 2
    x, y = reader.ptb_producer(raw_data, batch_size, num_steps) 
	# ptb_producer iterates on the raw_data.
	# ptb_producer function converts raw_data to tensors and splits it into x , y which is reshaped later and is returned.
	# the x , y returned by the ptb_producer is stored in this x , y which are nothing but tensors.
    
	with self.test_session() as session: # setting up a session in which this will be executed 
      # this part of the code covers the queuing and threading concepts of tensor-flow
	  coord = tf.train.Coordinator() # coordinator is an object which makes sure that all the threads that is used by the program stops at the same time.
	  # and also if any exception arises in any thread , it gets broad-casted to all other threads and they all stop at sync. 
	  # it is mainly used to bring all the threads at the end of the program and join it with the main program.
      tf.train.start_queue_runners(session, coord=coord)
	  # this line of the code , starts the queue_runner which takes care of the asynchronous execution of enqueues and also takes care of creating threads.
	  # this runs in the session that was created and the coordination of the threads will be taken care by coord which was assigned in the previous line.
      try:
        xval, yval = session.run([x, y]) # the test is run in this session created with values x , y returned by the ptb_producer function which is 
		# assigned to xval , yval respectively.
        self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]]) # this is used to check if all the values in xval and yval hold the same value
        self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]]) # as mentioned . 
        xval, yval = session.run([x, y])						
        self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
        self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
      finally: # once the try is done with execution , the interpretor compiles this part of the code where
        coord.request_stop() # the coord variable which is the coordination object asks the queuing to stop 
        coord.join() # and all the threads are joined together with the main program.


if __name__ == "__main__":
  tf.test.main() # calling this function to run the test case in tensor-flow IE., runs the unit test.
