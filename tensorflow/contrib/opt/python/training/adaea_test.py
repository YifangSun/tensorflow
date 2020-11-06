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
"""Functional tests for aggregate operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
import six

from tensorflow.python.training import saver

from tensorflow.contrib import lookup

from tensorflow.contrib.opt.python.training import adaea
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class AdaEAOptimizerTest(test.TestCase):
  def testMutableHashTableBasic(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
        var1 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
        indices_np = np.array([0, 1], dtype=np.int64)
        var0_np = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)
        var0.insert(constant_op.constant(indices_np),
                    constant_op.constant(var0_np)).run()
        var1_np = np.array([[3.0], [4.0]], dtype=dtype.as_numpy_dtype)
        var1.insert(constant_op.constant(indices_np),
                    constant_op.constant(var1_np)).run()

        grads0 = ops.IndexedSlices(
          constant_op.constant(
            [0.1], shape=[1, 1], dtype=dtype),
          constant_op.constant([0], dtype=dtypes.int64),
          constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
          constant_op.constant(
            [0.01], shape=[1, 1], dtype=dtype),
          constant_op.constant([1], dtype=dtypes.int64),
          constant_op.constant([2, 1]))
        ada_opt = adaea.AdaEAOptimizer(3.0, initial_accumulator_value=0.1)
        ada_update = ada_opt.apply_gradients(
          zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose(
          [[1.0], [2.0]],
          self.evaluate(var0.lookup(constant_op.constant(indices_np))))
        self.assertAllClose(
          [[3.0], [4.0]],
          self.evaluate(var1.lookup(constant_op.constant(indices_np))))
        # Run 3 step of sgd
        for _ in range(3):
          ada_update.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
          np.array([[-1.6026098728179932], [2.0]]),
          self.evaluate(var0.lookup(constant_op.constant(indices_np))))
        self.assertAllCloseAccordingToType(
          np.array([[3.0], [3.715679168701172]]),
          self.evaluate(var1.lookup(constant_op.constant(indices_np))))

  def testMutableHashTableRepeatedIndices(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        repeated_index_update_var = \
          lookup.MutableHashTable(dtypes.int64, dtype, [0])
        aggregated_update_var = \
          lookup.MutableHashTable(dtypes.int64, dtype, [0])
        indices_np = np.array([0, 1], dtype=np.int64)
        var_np = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)
        repeated_index_update_var.insert(constant_op.constant(indices_np),
                                         constant_op.constant(var_np)).run()
        aggregated_update_var.insert(constant_op.constant(indices_np),
                                     constant_op.constant(var_np)).run()
        grad_repeated_index = ops.IndexedSlices(
          constant_op.constant(
            [0.1, 0.1], shape=[2, 1], dtype=dtype),
          constant_op.constant([1, 1], dtype=dtypes.int64),
          constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
          constant_op.constant(
            [0.2], shape=[1, 1], dtype=dtype),
          constant_op.constant([1], dtype=dtypes.int64),
          constant_op.constant([2, 1]))
        repeated_update = adaea.AdaEAOptimizer(3.0).apply_gradients(
          [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = adaea.AdaEAOptimizer(3.0).apply_gradients(
          [(grad_aggregated, aggregated_update_var)])
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose(
          aggregated_update_var.lookup(constant_op.constant(indices_np)).eval(),
          repeated_index_update_var.lookup(constant_op.constant(indices_np))
            .eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(
            aggregated_update_var.lookup(constant_op.constant(indices_np))
              .eval(),
            repeated_index_update_var.lookup(constant_op.constant(indices_np))
              .eval())

  def testMutableHashTableStability(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        shape = [1, 6]
        var0 = lookup.MutableHashTable(
          dtypes.int64, dtype,
          [0, 0, 0, 0, 0, 0])
        indices_np = np.array([0], dtype=np.int64)
        var0_np = np.array([[
          0.00872496, -0.106952, 0.110467, 0.226505, -0.0147257,
          -0.0105945
        ]], dtype=dtype.as_numpy_dtype)
        var0.insert(constant_op.constant(indices_np),
                    constant_op.constant(var0_np)).run()
        grads0 = ops.IndexedSlices(
          constant_op.constant(
            [[
              -5.91278e-05, 5.31673e-05, -2.5779e-06, 4.29153e-05,
              -8.4877e-05, -9.48906e-05
            ]],
            shape=shape,
            dtype=dtype),
          constant_op.constant([0], dtype=dtypes.int64),
          constant_op.constant(shape))
        ada_opt = adaea.AdaEAOptimizer(1.0, initial_accumulator_value=0.1)
        ada_update = ada_opt.apply_gradients(zip([grads0], [var0]))
        init = variables.global_variables_initializer()
        init.run()
        ada_update.run()
        self.assertAllCloseAccordingToType(
          np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),
          self.evaluate(var0.get_attr(constant_op.constant(indices_np),
                                      "accumulator", [0, 0, 0, 0, 0, 0])))
        self.assertAllCloseAccordingToType(
          np.array([[
            0.00891194, -0.10712013, 0.11047515, 0.22636929, -0.0144573,
            -0.01029443
          ]]), self.evaluate(var0.lookup(constant_op.constant(indices_np))))

  def testSharingMutableHashTable(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
        var1 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
        indices_np = np.array([0, 1], dtype=np.int64)
        var0_np = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)
        var0.insert(constant_op.constant(indices_np),
                    constant_op.constant(var0_np)).run()
        var1_np = np.array([[3.0], [4.0]], dtype=dtype.as_numpy_dtype)
        var1.insert(constant_op.constant(indices_np),
                    constant_op.constant(var1_np)).run()

        grads0 = ops.IndexedSlices(
          constant_op.constant(
            [[0.1], [0.1]], shape=[2, 1], dtype=dtype),
          constant_op.constant([0, 1], dtype=dtypes.int64),
          constant_op.constant([2]))
        grads1 = ops.IndexedSlices(
          constant_op.constant(
            [[0.01], [0.01]], shape=[2, 1], dtype=dtype),
          constant_op.constant([0, 1], dtype=dtypes.int64),
          constant_op.constant([2]))
        ada_opt = adaea.AdaEAOptimizer(3.0)
        # Apply the optimizer twice.  Both applications will use
        # the same accums.
        ada_update1 = ada_opt.apply_gradients(
          zip([grads0, grads1], [var0, var1]))
        ada_update2 = ada_opt.apply_gradients(
          zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values.
        self.assertAllClose(
          [[1.0], [2.0]],
          self.evaluate(var0.lookup(constant_op.constant(indices_np))))
        self.assertAllClose(
          [[3.0], [4.0]],
          self.evaluate(var1.lookup(constant_op.constant(indices_np))))
        # Mix the first and the second adagrad for 3 steps.
        ada_update1.run()
        ada_update2.run()
        ada_update1.run()
        # Validate updated params (the same as with only 1 Adagrad).
        self.assertAllCloseAccordingToType(
          np.array([[-1.6026098728179932], [-0.6026098728179932]]),
          self.evaluate(var0.lookup(constant_op.constant(indices_np))))
        self.assertAllCloseAccordingToType(
          np.array([[2.715679168701172], [3.715679168701172]]),
          self.evaluate(var1.lookup(constant_op.constant(indices_np))))

  def testMutableHashTableGradClip(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
        var1 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
        indices_np = np.array([0, 1], dtype=np.int64)
        var_np = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)
        var0.insert(constant_op.constant(indices_np),
                    constant_op.constant(var_np)).run()
        var1.insert(constant_op.constant(indices_np),
                    constant_op.constant(var_np)).run()

        grads0 = ops.IndexedSlices(
          constant_op.constant(
            [[0.1], [0.01]], shape=[2, 1], dtype=dtype),
          constant_op.constant([0, 1], dtype=dtypes.int64),
          constant_op.constant([2]))
        ada_opt0 = adaea.AdaEAOptimizer(0.01, initial_accumulator_value=0.1)
        ada_opt1 = adaea.AdaEAOptimizer(0.01, initial_accumulator_value=0.1,
                                        min_scaled_lr=0.02)
        ada_update0 = ada_opt0.apply_gradients(
          zip([grads0], [var0]))
        ada_update1 = ada_opt1.apply_gradients(
          zip([grads0], [var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose(
          [[1.0], [2.0]],
          self.evaluate(var0.lookup(constant_op.constant(indices_np))))
        self.assertAllClose(
          [[1.0], [2.0]],
          self.evaluate(var1.lookup(constant_op.constant(indices_np))))
        # Run 10 step without grad clip
        for _ in range(10):
          ada_update0.run()
          ada_update1.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
          self.evaluate(var0.lookup(constant_op.constant(indices_np))),
          self.evaluate(var1.lookup(constant_op.constant(indices_np))))
        # Run 10 step with grad clip
        for _ in range(10):
          ada_update0.run()
          ada_update1.run()
        self.assertAllCloseAccordingToType(
          np.array([[0.9543586772273744], [1.9937083125514325]]),
          self.evaluate(var0.lookup(constant_op.constant(indices_np))))
        self.assertAllCloseAccordingToType(
          np.array([[0.9538168570812222], [1.9937083125514325]]),
          self.evaluate(var1.lookup(constant_op.constant(indices_np))))

  def testSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "adaea")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      dtype = dtypes.float32
      var0 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
      var1 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
      indices_np = np.array([0, 1], dtype=np.int64)
      var0_np = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)
      var0.insert(constant_op.constant(indices_np),
                  constant_op.constant(var0_np)).run()
      var1_np = np.array([[3.0], [4.0]], dtype=dtype.as_numpy_dtype)
      var1.insert(constant_op.constant(indices_np),
                  constant_op.constant(var1_np)).run()

      grads0 = ops.IndexedSlices(
        constant_op.constant(
          [0.1], shape=[1, 1], dtype=dtype),
        constant_op.constant([0], dtype=dtypes.int64),
        constant_op.constant([2, 1]))
      grads1 = ops.IndexedSlices(
        constant_op.constant(
          [0.01], shape=[1, 1], dtype=dtype),
        constant_op.constant([1], dtype=dtypes.int64),
        constant_op.constant([2, 1]))

      ada_opt = adaea.AdaEAOptimizer(3.0, initial_accumulator_value=0.1)
      ada_update = ada_opt.apply_gradients(
        zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())
      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      # Fetch params to validate initial values
      self.assertAllClose(
        [[1.0], [2.0]],
        self.evaluate(var0.lookup(constant_op.constant(indices_np))))
      self.assertAllClose(
        [[3.0], [4.0]],
        self.evaluate(var1.lookup(constant_op.constant(indices_np))))
      # Run 3 step of adaea
      for _ in range(3):
        ada_update.run()
      # Validate updated params
      self.assertAllCloseAccordingToType(
        np.array([[-1.6026098728179932], [2.0]]),
        self.evaluate(var0.lookup(constant_op.constant(indices_np))))
      self.assertAllCloseAccordingToType(
        np.array([[3.0], [3.715679168701172]]),
        self.evaluate(var1.lookup(constant_op.constant(indices_np))))

      save = saver.Saver()
      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      dtype = dtypes.float32
      var0 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
      var1 = lookup.MutableHashTable(dtypes.int64, dtype, [0])
      grads0 = ops.IndexedSlices(
        constant_op.constant(
          [0.1], shape=[1, 1], dtype=dtype),
        constant_op.constant([0], dtype=dtypes.int64),
        constant_op.constant([2, 1]))
      grads1 = ops.IndexedSlices(
        constant_op.constant(
          [0.01], shape=[1, 1], dtype=dtype),
        constant_op.constant([1], dtype=dtypes.int64),
        constant_op.constant([2, 1]))

      ada_opt = adaea.AdaEAOptimizer(3.0, initial_accumulator_value=0.1)
      ada_update = ada_opt.apply_gradients(
        zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllCloseAccordingToType(
        np.array([[-1.6026098728179932], [2.0]]),
        self.evaluate(var0.lookup(constant_op.constant(indices_np))))
      self.assertAllCloseAccordingToType(
        np.array([[3.0], [3.715679168701172]]),
        self.evaluate(var1.lookup(constant_op.constant(indices_np))))

      # Run 3 step of adaea
      for _ in range(3):
        ada_update.run()

      self.assertAllCloseAccordingToType(
        np.array([[-3.9289901268343295], [2.0]]),
        self.evaluate(var0.lookup(constant_op.constant(indices_np))))
      self.assertAllCloseAccordingToType(
        np.array([[3.0], [3.4317829142662]]),
        self.evaluate(var1.lookup(constant_op.constant(indices_np))))


if __name__ == "__main__":
  test.main()
