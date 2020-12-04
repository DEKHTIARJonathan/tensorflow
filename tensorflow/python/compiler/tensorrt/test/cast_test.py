# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test conversion of graphs involving INT32 tensors and operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CastFp32Fp16Test(trt_test.TfTrtIntegrationTestBase):
  """Tests cast to back and forth between FP16 and FP32."""

  def GraphFn(self, net):

    # Convert FP16 => FP32
    net = math_ops.cast(net, dtypes.float32, name="cast_fp16_to_fp32")
    net = math_ops.mul(net, net, name="mul_fp32_1")
    net = math_ops.mul(net, net, name="mul_fp32_2")
    # Convert FP32 => FP16
    net = math_ops.cast(net, dtypes.float16, name="cast_fp32_to_fp16")
    net = math_ops.add(net, net, name="add_fp16_1")
    net = math_ops.add(net, net, name="add_fp16_2")
    # Convert FP16 => FP32
    net = math_ops.cast(net, dtypes.float32, name="cast_fp16_to_fp32")
    net = math_ops.mul(net, net, name="add_fp32_1")
    net = math_ops.mul(net, net, name="add_fp32_2")
    # Convert FP16 => FP32
    net = math_ops.cast(net, dtypes.float32, name="cast_fp32_to_fp32")

    return array_ops.identity(net, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float16, [[1, 10]], [[1, 10]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    if run_params.precision_mode != "FP32":
      return {
        "TRTEngineOp_0": [
          "cast_fp16_to_fp32", "mul_fp32_1", "mul_fp32_2",
          "cast_fp32_to_fp16", "add_fp16_1", "add_fp16_2",
          "cast_fp16_to_fp32", "add_fp32_1", "add_fp32_2",
          "cast_fp32_to_fp32"  # This OP is removed anyway by the grappler
        ]
      }
    else:
      return {
        "TRTEngineOp_0": ["mul_fp32_1", "mul_fp32_2"],
        "TRTEngineOp_1": ["add_fp16_1", "add_fp16_2"],
        "TRTEngineOp_2": ["add_fp32_1", "add_fp32_2"]
      }

  def GetConversionParams(self, run_params):
    """Returns a TrtConversionParams for test."""
    conversion_params = super(CastFp32Fp16Test, self).GetConversionParams(
      run_params
    )
    conversion_params = conversion_params._replace(minimum_segment_size=2)
    # raise Exception(conversion_params)
    return conversion_params

  def ShouldRunTest(self, run_params):
    should_run, reason = super().ShouldRunTest(run_params)
    # Only run for TRT 7.0.0 and above.
    return should_run and \
           trt_test.IsTensorRTVersionGreaterEqual(7), \
           reason + ' and TRT Version >= 7.0.0'


if __name__ == "__main__":
  test.main()
