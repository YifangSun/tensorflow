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

"""AdaEA for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adagrad


class AdaEAOptimizer(adagrad.AdagradOptimizer):
  """Optimizer that extends Adagrad algorithm with embedding-aware strategies.

  """

  def __init__(self, learning_rate, initial_accumulator_value=0.1,
               min_scaled_lr=0,
               use_locking=False, name="AdaEA"):
    """Construct a new Adagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      min_scaled_lr: low bound of scaled lr (lr * rsqrt(accumulator))
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate` can be a callable that
    takes no arguments and returns the actual value to use. This can be useful
    for changing these values across different invocations of optimizer
    functions.
    @end_compatibility
    """
    super(AdaEAOptimizer, self).__init__(
      learning_rate, initial_accumulator_value,
      use_locking, name)
    self._min_scaled_lr = min_scaled_lr

  def _create_slots(self, var_list):
    for v in var_list:
      if context.executing_eagerly() or \
              not v.op.type.startswith("MutableHashTable"):
        dtype = v.dtype.base_dtype
        if v.get_shape().is_fully_defined():
          init = init_ops.constant_initializer(self._initial_accumulator_value,
                                               dtype=dtype)
        else:
          init = self._init_constant_op(v, dtype)
        self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                                "accumulator", self._name)

  def _lookuptable_apply_sparse(self, grad, var, indices):
    # \\(acc := acc + (g_t * g_t)\\)
    attr_key = "accumulator"
    init_constant = gen_array_ops.fill(var.value_shape,
                                       self._initial_accumulator_value)
    default_value = math_ops.cast(init_constant, dtypes.float32)
    acc = gen_lookup_ops.lookup_table_get_attr(var.handle,
                                               indices,
                                               attr_key,
                                               default_value)
    acc = math_ops.cast(acc, var.value_dtype)
    acc_t_slice = acc + math_ops.square(grad)
    acc_t_slice_f = math_ops.cast(acc_t_slice, dtypes.float32)
    acc_update_op = gen_lookup_ops.lookup_table_set_attr(var.handle,
                                                         indices,
                                                         attr_key,
                                                         acc_t_slice_f)

    lr_t = math_ops.cast(self._learning_rate_tensor, var.value_dtype)
    scaled_lr = lr_t * math_ops.rsqrt(acc_t_slice)
    if self._min_scaled_lr != 0:
      # clip the scaled lr to prevent gradient too small.
      min_scaled_lr = math_ops.cast(self._min_scaled_lr, var.value_dtype)
      min_scaled_lr_vec = gen_array_ops.fill(array_ops.shape(grad),
                                             min_scaled_lr)
      scaled_lr = array_ops.where(
        math_ops.less(scaled_lr, self._min_scaled_lr),
        min_scaled_lr_vec,
        scaled_lr)
    var_slice = scaled_lr * grad
    # \\(variable -= learning_rate * rsqrt(accumulator) * g_t
    var_update_op = gen_lookup_ops.lookup_table_scatter_sub_v2(var.handle,
                                                               indices,
                                                               var_slice)

    return control_flow_ops.group(var_update_op, acc_update_op)

