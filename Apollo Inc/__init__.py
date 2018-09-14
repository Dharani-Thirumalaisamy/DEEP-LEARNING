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

"""Makes helper libraries available in the ptb package."""

from __future__ import absolute_import 
# python will look for the standard library's version for any module that is imported 
from __future__ import division
# python will change '/' operator to true division and use '//' operator when necessary
from __future__ import print_function
# brings print function from python 3 to python2.6+

import reader
# calling the reader module (reader.py)
import util
# calling the util module(util.py)
