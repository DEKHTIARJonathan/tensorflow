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
# =============================================================================
"""Exposes the Python wrapper conversion to trt_graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils import version

from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2


def disable_non_trt_optimizers_in_rewriter_config(rewriter_config):
  """Modifies rewriter_config to disable all non-TRT optimizations."""
  off = rewriter_config_pb2.RewriterConfig.OFF

  rewriter_config.arithmetic_optimization = off
  rewriter_config.auto_mixed_precision = off
  rewriter_config.auto_parallel.enable = False
  rewriter_config.constant_folding = off
  rewriter_config.debug_stripper = off
  rewriter_config.dependency_optimization = off
  # This one needs to be ON to allow TF-TRT
  rewriter_config.disable_meta_optimizer = False
  rewriter_config.disable_model_pruning = True
  rewriter_config.function_optimization = off
  rewriter_config.implementation_selector = off
  rewriter_config.layout_optimizer = off
  rewriter_config.loop_optimization = off
  rewriter_config.memory_optimization = (
      rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
  rewriter_config.min_graph_nodes = -1
  rewriter_config.pin_to_host_optimization = off
  rewriter_config.remapping = off
  rewriter_config.scoped_allocator_optimization = off
  rewriter_config.shape_optimization = off


def _IsTensorRTVersionGreaterEqual(trt_ver, major, minor=0, patch=0):
  assert isinstance(trt_ver, tuple)
  assert len(trt_ver) == 3

  trt_ver = [str(x) for x in trt_ver]
  trt_ver = version.LooseVersion(".".join(trt_ver))
  target_ver = version.LooseVersion("{}.{}.{}".format(major, minor, patch))

  return trt_ver >= target_ver


def IsLinkedTensorRTVersionGreaterEqual(major, minor=0, patch=0):
  ver = _pywrap_py_utils.get_linked_tensorrt_version()
  return _IsTensorRTVersionGreaterEqual(ver, major, minor, patch)
