# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Exposes the python wrapper for TensorRT graph transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.platform import tf_logging as logging


if _pywrap_py_utils.is_tensorrt_enabled():
  """Check compatibility of TensorRT version.
  Raises:
    RuntimeError: if the TensorRT library version is incompatible.
  """
  linked_version = _pywrap_py_utils.get_linked_tensorrt_version()
  loaded_version = _pywrap_py_utils.get_loaded_tensorrt_version()

  logging.info("Linked TensorRT version: %s" % str(linked_version))
  logging.info("Loaded TensorRT version: %s" % str(loaded_version))

  def raise_trt_deprecation_warning(version_type, trt_version):
    assert version_type in ["linked", "loaded"], "Incorrect value received " \
        "for version_type: %s. Accepted: ['linked', 'loaded']" % version_type

    logging.error(
        "The {version_type} version of TensorRT: `{trt_version}` has now "
        "been removed. Please upgrade to TensorRT 7 or more recent.".format(
            version_type=version_type,
            trt_version=trt_utils.versionTupleToString(trt_version)
        )
    )

    raise RuntimeError("Incompatible %s TensorRT versions" % version_type)

  if not trt_utils.IsLinkedTensorRTVersionGreaterEqual(7, 0, 0):
    raise_trt_deprecation_warning("linked", linked_version)

  if not trt_utils.IsLoadedTensorRTVersionGreaterEqual(7, 0, 0):
    raise_trt_deprecation_warning("loaded", loaded_version)

  if (loaded_version[0] != linked_version[0] or
      not trt_utils.IsLoadedTensorRTVersionGreaterEqual(*linked_version)):
    logging.error(
        "Loaded TensorRT %s but linked TensorFlow against TensorRT %s. A few "
        "requirements must be met:\n"
        "\t-It is required to use the same major version of TensorRT during "
        "compilation and runtime.\n"
        "\t-TensorRT does not support forward compatibility. The loaded "
        "version has to be equal or more recent than the linked version." % (
          trt_utils.versionTupleToString(loaded_version),
          trt_utils.versionTupleToString(linked_version)
    ))
    raise RuntimeError("Incompatible TensorRT major version")

  elif loaded_version != linked_version:
    logging.info(
        "Loaded TensorRT %s and linked TensorFlow against TensorRT %s. This is "
        "supported because TensorRT minor/patch upgrades are backward "
        "compatible." % (
          trt_utils.versionTupleToString(loaded_version),
          trt_utils.versionTupleToString(linked_version)
    ))
