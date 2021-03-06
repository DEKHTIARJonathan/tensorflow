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
"""Distribution tests for keras.layers.preprocessing.discretization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute import strategy_combinations
from tensorflow.python.keras.layers.preprocessing import discretization
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.platform import test


@ds_combinations.generate(
    combinations.combine(
        distribution=strategy_combinations.strategies_minus_tpu,
        mode=["eager", "graph"]))
class DiscretizationDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution(self, distribution):
    input_array = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])

    expected_output = [[0, 1, 3, 1], [0, 3, 2, 0]]
    expected_output_shape = [None, 4]

    config.set_soft_device_placement(True)

    with distribution.scope():
      input_data = keras.Input(shape=(4,))
      layer = discretization.Discretization(bin_boundaries=[0., 1., 2.])
      bucket_data = layer(input_data)
      self.assertAllEqual(expected_output_shape, bucket_data.shape.as_list())

      model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
  test.main()
