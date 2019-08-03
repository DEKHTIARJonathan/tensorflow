# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for deterministic BiasAdd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests import bias_op_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class BiasAddDeterministicTest(bias_op_base.BiasAddTestBase):

  def _make_shape_tuple(self, batch_size, channel_count, data_rank, data_dim,
                        channels_position):
    data_dims = data_rank * (data_dim,)
    if channels_position == 'first':
      shape = (batch_size,) + (channel_count,) + data_dims
    elif channels_position == 'last':
      shape = (batch_size,) + data_dims + (channel_count,)
    else:
      raise ValueError("Unknown data format")
    return shape

  def _data_format_from_channels_position(self, channels_position=None):
    if channels_position == 'first':
      return 'NCHW'
    elif channels_position == 'last':
      return 'NHWC'
    else:
      raise ValueError("Unknown channels_position")

  def _random_data_op(self, shape, data_type):
    return constant_op.constant(
        2 * np.random.random_sample(shape) - 1, dtype=data_type)

  def _random_ndarray(self, shape):
    return 2 * np.random.random_sample(shape) - 1

  def _assert_reproducible(self, operation, feed_dict={}):
    with self.cached_session(force_gpu=True):
      result_a = operation[0].eval(feed_dict=feed_dict)
      result_b = operation[0].eval(feed_dict=feed_dict)
      self.assertAllEqual(result_a, result_b)

  def _testGradientsCase(self, channels_position, data_rank, data_type):
    np.random.seed(3)
    batch_size = 10
    channel_count = 8
    data_dim = 14
    in_shape = self._make_shape_tuple(batch_size, channel_count, data_rank,
                                      data_dim, channels_position)
    bias_shape = (channel_count,)
    out_shape = in_shape
    in_op = self._random_data_op(in_shape, data_type)
    bias_op = self._random_data_op(bias_shape, data_type)
    data_format = self._data_format_from_channels_position(channels_position)
    bias_add_op = nn_ops.bias_add(in_op, bias_op, data_format=data_format)
    upstream_gradients = array_ops.placeholder(data_type, shape=out_shape,
                                               name='upstream_gradients')
    gradient_injector_op = bias_add_op * upstream_gradients
    # The gradient function behaves as if grad_ys is multiplied by the op
    # gradient result, not passing the upstram gradients through the op's
    # gradient generation graph. This is the reason for using the
    # gradient_injector_op
    grad_ys = None
    bias_gradients_op = gradients_impl.gradients(
        gradient_injector_op, bias_op, grad_ys=grad_ys,
        colocate_gradients_with_ops=True)
    for i in range(5):
      feed_dict = {upstream_gradients: self._random_ndarray(out_shape)}
      self._assert_reproducible(bias_gradients_op, feed_dict=feed_dict)

  @test_util.run_cuda_only
  def testGradients(self):
    for channels_position in ('first', 'last'):
      for data_rank in (1, 2, 3):
        for data_type in (dtypes.float16, dtypes.float32, dtypes.float64):
          self._testGradientsCase(channels_position, data_rank, data_type)

  # TODO(duncanriach): Re-enable the following three tests for the error checks
  #   after deterministic functionality is implemented at the CUDA kernel level.
  def testInputDims(self):
    pass

  def testBiasVec(self):
    pass

  def testBiasInputsMatch(self):
    pass


if __name__ == "__main__":
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require this file to be made into a base
  # and then two more test files to be created.
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  test.main()
