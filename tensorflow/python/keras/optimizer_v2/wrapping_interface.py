# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Adam for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend_config
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


class _UnwrapPreventer(object):
  """Wrapper that DistributionStrategy will not unwrap.

  Typically, DistributionStrategy will unwrap values when going from a cross-
  replica context to a replica context via `call_for_each_replica`. This class
  is a wrapper that DistributionStrategy will not unwrap, so it can be used to
  prevent it from unwrapping a value.

  TODO(reedwm): Find/implement a better way of preventing values from being
  unwrapped by DistributionStrategy
  """

  def __init__(self, value):
    self.value = value


class WrappingInterfaceOptimizer(optimizer_v2.OptimizerV2):
  # =============== PUBLIC METHODS API - CAN OVERWRITTEN ============== #

  # Developer Notes on Optimizer Priority - @DEKHTIARJonathan:
  #
  # Hooks might need to be executed in a specific order to avoid conflicts or
  # corner cases. One typical example is the combination of:
  # - collective allreduce (with Horovod for instance)
  # - LossScaling optimizer for mixed precision training
  #
  # The desirable order is: allreduce first, loss scaling last.
  # We achieve by introducing an optimizer priority that enforces this behavior.
  # An experienced user, can always override this value at run time or by
  # subclassing.
  #
  # Default order: from the highest priority to the lowest
  # If two `WrappingInterfaceOptimizers` have the same priority, order of
  # execution can not be guaranteed.
  _PRIORITY = 0  # default priority value, shall be adapted when subclassed

  def setup(self, *args, **kwargs):
    # this API exposes `setup` instead of __init__ to prevent users
    # from not calling the `super().__init__` method => might break API.
    pass

  def before_compute_gradients_hook(self, loss):
    return loss

  def after_compute_gradients_hook(self, grads_and_vars):
    return grads_and_vars

  def cond_apply_step_hook(self, grads_and_vars):
    return gen_control_flow_ops.no_op(), True

  def before_apply_gradients_hook(self, grads_and_vars):
    return grads_and_vars

  # =============== PUBLIC METHODS API - CAN BE EXTENDED ============== #
  # To be noted: calling `super()` is required for these methods

  def get_config(self):
    serialized_optimizer = optimizers.serialize(self._optimizer)
    return {
      'optimizer': serialized_optimizer,
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()  # Make a copy, since we mutate config
    config['optimizer'] = optimizers.deserialize(
      config['optimizer'],
      custom_objects=custom_objects
    )
    return cls(**config)

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
  # =============== PRIVATE API - DO NOT OVERWRITE ============== #
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

  def __init__(
      self,
      optimizer,
      *args,
      **kwargs
  ):
    """Initializes the wrapping interface optimizer.

    Args:
    optimizer: The Optimizer instance to wrap.
    """

    try:
      name = kwargs["name"]
      del kwargs["name"]
    except KeyError:
      name = self.__class__.__name__

    if not isinstance(optimizer, optimizer_v2.OptimizerV2):
      raise ValueError(
        '"optimizer" must be an instance of OptimizerV2, but '
        'got: %s' % optimizer
      )

    if hasattr(optimizer, 'clipnorm'):
      raise ValueError(
        '%s does not support wrapping optimizers with a clipnorm. '
        'Optimizer %s has clipnorm %s' % (
          self.__class__.__name__,
          optimizer,
          optimizer.clipnorm
        ))

    if hasattr(optimizer, 'clipvalue'):
      raise ValueError(
        '%s does not support wrapping optimizers with a clipvalue. '
        'Optimizer %s has clipvalue %s' % (
          self.__class__.__name__,
          optimizer,
          optimizer.clipvalue
        ))

    self._optimizer = optimizer
    self._track_trackable(self._optimizer, 'base_optimizer')

    # Needed because the superclass's __getattribute__ checks this.
    self._hyper = {}

    self.setup(*args, **kwargs)

    super(WrappingInterfaceOptimizer, self).__init__(name=name)

  # ============ Automatic Nested Optimizers Hooks Apply Methods =========== #

  def apply_before_compute_gradients_hooks(self, loss):
    for opt in self.all_wrapping_optimizers:
      loss = opt.before_compute_gradients_hook(loss)
    return loss

  def apply_after_compute_gradients_hooks(self, grads_and_vars):
    for opt in self.all_wrapping_optimizers:
      grads_and_vars = opt.after_compute_gradients_hook(grads_and_vars)
    return grads_and_vars

  def apply_cond_apply_step_hooks(self, grads_and_vars):

    should_apply_bool = True
    should_apply_op = list()

    for opt in self.all_wrapping_optimizers:
      _should_apply_op, _should_apply_bool = opt.cond_apply_step_hook(
        grads_and_vars
      )

      if isinstance(should_apply_bool, bool) and should_apply_bool:
        should_apply_bool = _should_apply_bool
      else:
        should_apply_bool = math_ops.logical_and(
          should_apply_bool,
          _should_apply_bool
        )

      should_apply_op.append(_should_apply_op)

      return should_apply_op, should_apply_bool

  def apply_before_apply_gradients_hooks(self, grads_and_vars):
    for opt in self.all_wrapping_optimizers:
      grads_and_vars = opt.before_apply_gradients_hook(grads_and_vars)
    return grads_and_vars

  def _compute_gradients(self, loss, var_list, grad_loss=None):

    loss = self.apply_before_compute_gradients_hooks(loss)

    grads_and_vars = self._optimizer._compute_gradients(
      loss,
      var_list,
      grad_loss
    )

    grads_and_vars = self.apply_after_compute_gradients_hooks(grads_and_vars)

    return grads_and_vars

  def get_gradients(self, loss, params):

    loss = self.apply_before_compute_gradients_hooks(loss)

    grads = self._optimizer.get_gradients(loss, params)

    grads_and_vars = list(zip(grads, params))

    grads_and_vars = self.apply_after_compute_gradients_hooks(grads_and_vars)

    return [g for g, _ in grads_and_vars]

  def apply_gradients(self, grads_and_vars, name=None):
    if distribution_strategy_context.in_cross_replica_context():
      raise ValueError(
        'apply_gradients() must be called in a replica context.')

    grads_and_vars = tuple(grads_and_vars)

    grads_and_vars = self.apply_before_apply_gradients_hooks(grads_and_vars)

    return distribution_strategy_context.get_replica_context().merge_call(
      self._apply_gradients_cross_replica,
      args=(grads_and_vars, name)
    )

  def _apply_gradients(self, grads, wrapped_vars, name):
    return self._optimizer.apply_gradients(
      list(zip(grads, wrapped_vars.value)),
      name
    )

  def _apply_gradients_cross_replica(self, distribution, grads_and_vars, name):

    should_apply_op, should_apply_bool = self.apply_cond_apply_step_hooks(grads_and_vars)

    def apply_fn():
      # We do not want DistributionStrategy to unwrap any
      # MirroredVariables in grads_and_vars, because even in a replica
      # context, the wrapped optimizer expects mirrored variables. So
      # we wrap the variables with an _UnwrapPreventer, preventing
      # DistributionStrategy from unwrapping the MirroredVariables.
      wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])
      grads = [g for g, _ in grads_and_vars]

      return distribution.extended.call_for_each_replica(
        self._apply_gradients,
        args=(grads, wrapped_vars, name)
      )

    # Note: We must call this cond() in a cross-replica context.
    # DistributionStrategy does not support having a cond in a replica
    # context with a branch that calls `merge_call`, and
    # self._optimizer.apply_gradients calls `merge_call`.
    maybe_apply_op = smart_cond.smart_cond(
      should_apply_bool,
      apply_fn,
      gen_control_flow_ops.no_op
    )

    return control_flow_ops.group(*[maybe_apply_op, *should_apply_op])

  # For the most part, we only expose methods in the base OptimizerV2, not
  # individual subclasses like Adam. However, although "learning_rate" and "lr"
  # properties are not part of the base OptimizerV2 class, they are part of most
  # subclasses, so we expose them here for convenience.

  # =============== PRIVATE API - DO NOT OVERWRITE ============== #
  # GETTERS & SETTERS - Shall eventually be implemented using:
  # - __getattr__
  # - __setattr__

  @property
  def iterations(self):
    return self._optimizer.iterations

  @iterations.setter
  def iterations(self, variable):
    self._optimizer.iterations = variable

  @property
  def learning_rate(self):
    return self._optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, lr):
    self._optimizer.learning_rate = lr

  @property
  def lr(self):
    return self._optimizer.lr

  @lr.setter
  def lr(self, lr):
    self._optimizer.lr = lr

  def get_slot_names(self):
    return self._optimizer.get_slot_names()

  def variables(self):
    return self._optimizer.variables()

  @property
  def weights(self):
    return self._optimizer.weights

  def get_weights(self):
    return self._optimizer.get_weights()

  def set_weights(self, weights):
    return self._optimizer.set_weights(weights)

  # shortcut to access all underlying optimizers
  @property
  def all_wrapping_optimizers(self):
    wrapping_opts = list()

    optimizer = self
    while issubclass(optimizer.__class__, WrappingInterfaceOptimizer):
      wrapping_opts.append(optimizer)
      optimizer = optimizer._optimizer

    wrapping_opts = sorted(
      wrapping_opts,
      key=lambda opt: opt._PRIORITY,
      reverse=True
    )

    return wrapping_opts

  # Delegations: We delegate most OptimizerV2 methods to the wrapped optimizer
  # below.
  def get_slot(self, var, slot_name):
    # We cannot implement get_slot for the following reason: When saving a
    # checkpoint, two optimizers cannot share slot variables. Since both the
    # WrappingInterfaceOptimizer and the wrapped optimizer (self and self._optimizer
    # respectively) are checkpointed, we cannot expose the wrapped
    # optimizer's slots in the WrappingInterfaceOptimizer. Otherwise,
    # a checkpoint would believe both optimizers share slot variables.
    raise AttributeError(
      'You cannot call get_slot on a %s. '
      'This limitation will be removed in the future.' %
      self.__class__.__name__
    )

  def add_slot(self, var, slot_name, initializer='zeros'):
    # We disallow adding a slot for consistency with `get_slot`.
    raise AttributeError(
      'You cannot call add_slot on a %s. '
      'This limitation will be removed in the future.' %
      self.__class__.__name__
    )

  # We do not override some OptimizerV2 methods. For each, we describe why we do
  # not delegate them to self._optimizer:
  # * get_updates: get_updates() calls get_gradients(). Since we override
  #   get_gradients(), we cannot delegate get_updates() to self._optimizer,
  #   otherwise the overridden get_gradients() method would not be called.
  #   Luckily, get_updates() does not access any OptimizerV2 fields, so
  #   inheriting the OptimizerV2 version works fine.
  # * minimize: We don't delegate for a similar as get_updates(): it calls
  #   both self._compute_gradients() and self.apply_gradients(), and both need
  #   to have the WrappingInterfaceOptimizer version called.

  # TODO(reedwm): Maybe merge this class's functionality into OptimizerV2.

  # TODO(reedwm): Maybe throw an error if mixed precision is used without this
  # optimizer being used.
