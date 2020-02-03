# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the loss scaling optimizer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision.experimental import \
  loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2.wrapping_interface import \
  WrappingInterfaceOptimizer
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training.experimental import \
  loss_scale as loss_scale_module
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.mixed_precision.experimental.LossScaleOptimizer')
class LossScaleOptimizer(WrappingInterfaceOptimizer):
  """An optimizer that applies loss scaling.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:

  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used. By
  multiplying the loss, each intermediate gradient will have the same multiplier
  applied.

  The loss scale can either be a fixed constant, chosen by the user, or be
  dynamically determined. Dynamically determining the loss scale is convenient
  as a loss scale does not have to be explicitly chosen. However it reduces
  performance.

  This optimizer wraps another optimizer and applies loss scaling to it via a
  `LossScale`. Loss scaling is applied whenever gradients are
  computed, either through `minimize()` or `get_gradients()`. The loss scale is
  updated via `LossScale.update()` whenever gradients are applied, either
  through `minimize()` or `apply_gradients()`. For example:

  ```python
  opt = tf.keras.optimizers.SGD(0.1)
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
  # 'minimize' applies loss scaling to the loss and updates the loss sale.
  opt.minimize(loss_fn)
  ```

  If a `tf.GradientTape` is used to compute gradients instead of
  `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, the loss
  and gradients must be scaled manually. This can be done by calling
  `LossScaleOptimizer.get_scaled_loss` before passing the loss to
  `tf.GradientTape`, and `LossScaleOptimizer.get_unscaled_gradients` after
  computing the gradients with `tf.GradientTape`. For example:

  ```python
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(...)
  vars = ...
  with tf.GradientTape() as tape:
    loss = ...
    scaled_loss = opt.get_scaled_loss(loss)
  scaled_grads = tape.gradient(scaled_loss, vars)
  grads = opt.get_unscaled_gradients(scaled_grads)
  opt.apply_gradients(zip(grads, vars))  # Loss scale will be updated here
  ```
  """

  # =============== PUBLIC METHODS API - CAN OVERWRITTEN ============== #

  def setup(self, loss_scale):
    """Initializes this loss scale optimizer.
    Args:
      optimizer: The Optimizer instance to wrap.
      loss_scale: The loss scale to scale the loss and gradients. This can
      either be an int/float to use a fixed loss scale, the string "dynamic"
      to use dynamic loss scaling, or an instance of a LossScale. The string
      "dynamic" equivalent to passing `DynamicLossScale()`, and passing an
      int/float is equivalent to passing a FixedLossScale with the given loss
      scale.
    """

    self._loss_scale = keras_loss_scale_module.get(loss_scale)

    if self._loss_scale is None:
      raise ValueError('loss_scale cannot be None.')

    for weight in loss_scale_module.get_loss_scale_weights(
        self._loss_scale):
      # We cannot call `track_variable` in the LossScale class itself, because a
      # file outside of Keras cannot depend on a Keras file. Calling it here
      # instead is OK, because a variable only needs to be tracked if used with
      # a Keras class, and the only way to use LossScale with a Keras class is
      # through the LossScaleOptimizer.
      backend.track_variable(weight)

    self._track_trackable(self._loss_scale, 'loss_scale')

  def before_compute_gradients_hook(self, loss):
    return self.get_scaled_loss(loss)

  def after_compute_gradients_hook(self, grads_and_vars):
    grads = [g for g, _ in grads_and_vars]
    variables = [v for _, v in grads_and_vars]
    unscaled_grads = self.get_unscaled_gradients(grads)
    return list(zip(unscaled_grads, variables))

  def cond_apply_step_hook(self, grads_and_vars):
    grads = [g for g, _ in grads_and_vars]
    return self._loss_scale.update(grads)

  def get_scaled_loss(self, loss):
    """Scales the loss by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for
    an example.

    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
      a tensor or a callable returning a tensor.

    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale()`.
    """
    loss_scale = self._loss_scale()
    if callable(loss):
      def new_loss():
        loss_val = loss()
        return loss_val * math_ops.cast(loss_scale, loss_val.dtype)

      return new_loss
    else:
      return loss * math_ops.cast(loss_scale, loss.dtype)

  def get_unscaled_gradients(self, grads):
    """Unscales the gradients by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to unscale the gradients
    after computing them with `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_scaled_loss` should also be called. See
    the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for an
    example.

    Args:
      grads: A list of tensors, each which will be divided by the loss scale.
      Can have None values, which are ignored.

    Returns:
      A new list the same size as `grads`, where every non-None value in `grads`
      is divided by `LossScaleOptimizer.loss_scale()`.
    """
    loss_scale = self._loss_scale()
    loss_scale_reciprocal = 1. / loss_scale
    return [
      g * math_ops.cast(loss_scale_reciprocal, g.dtype)
      if g is not None else None for g in grads
    ]

  @property
  def loss_scale(self):
    """The `LossScale` instance associated with this optimizer."""
    return self._loss_scale

  def get_config(self):
    config = {
      'loss_scale': keras_loss_scale_module.serialize(
        self._loss_scale
      )
    }
    config.update(super(LossScaleOptimizer, self).get_config())

    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()  # Make a copy, since we mutate config
    config['loss_scale'] = keras_loss_scale_module.deserialize(
      config['loss_scale'], custom_objects=custom_objects
    )

    return super(LossScaleOptimizer, cls).from_config(
      config,
      custom_objects=custom_objects
    )
