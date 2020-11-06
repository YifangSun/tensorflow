# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.lookup.lookup."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import numpy as np
import six
import time

from tensorflow.contrib import lookup
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training.checkpointable import util as checkpointable

class HashTableOpTest(test.TestCase):

  def testHashTable(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

      exported_keys_tensor, exported_values_tensor = table.export()

      self.assertItemsEqual([b"brain", b"salad", b"surgery"],
                            exported_keys_tensor.eval())
      self.assertItemsEqual([0, 1, 2], exported_values_tensor.eval())

  def testHashTableFindHighRank(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(
          [["brain", "salad"], ["tank", "tarkus"]])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testHashTableInitWithPythonArrays(self):
    with self.cached_session():
      default_val = -1
      keys = ["brain", "salad", "surgery"]
      values = [0, 1, 2]
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(
              keys, values, value_dtype=dtypes.int64),
          default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableInitWithNumPyArrays(self):
    with self.cached_session():
      default_val = -1
      keys = np.array(["brain", "salad", "surgery"], dtype=np.str)
      values = np.array([0, 1, 2], dtype=np.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testMultipleHashTables(self):
    with self.cached_session() as sess:
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)

      table1 = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table2 = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table3 = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual(3, table1.size().eval())
      self.assertAllEqual(3, table2.size().eval())
      self.assertAllEqual(3, table3.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testHashTableWithTensorDefault(self):
    with self.cached_session():
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableWithSparseTensorInput(self):
    with self.cached_session() as sess:
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      sp_indices = [[0, 0], [0, 1], [1, 0]]
      sp_shape = [2, 2]
      input_tensor = sparse_tensor.SparseTensor(
          constant_op.constant(sp_indices, dtypes.int64),
          constant_op.constant(["brain", "salad", "tank"]),
          constant_op.constant(sp_shape, dtypes.int64))
      output = table.lookup(input_tensor)

      out_indices, out_values, out_shape = sess.run(output)

      self.assertAllEqual([0, 1, -1], out_values)
      self.assertAllEqual(sp_indices, out_indices)
      self.assertAllEqual(sp_shape, out_shape)

  def testSignatureMismatch(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      # Ref types do not produce a lookup signature mismatch.
      input_string_ref = variables.Variable("brain")
      variables.global_variables_initializer().run()
      self.assertEqual(0, table.lookup(input_string_ref).eval())

      input_string = constant_op.constant([1, 2, 3], dtypes.int64)
      with self.assertRaises(TypeError):
        table.lookup(input_string)

      with self.assertRaises(TypeError):
        lookup.HashTable(
            lookup.KeyValueTensorInitializer(keys, values), "UNK")

  def testDTypes(self):
    with self.cached_session():
      default_val = -1
      with self.assertRaises(TypeError):
        lookup.HashTable(
            lookup.KeyValueTensorInitializer(["a"], [1], [dtypes.string],
                                             dtypes.int64), default_val)

  def testNotInitialized(self):
    with self.cached_session():
      default_val = -1
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(
              ["a"], [1], value_dtype=dtypes.int64),
          default_val)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      with self.assertRaisesOpError("Table not initialized"):
        output.eval()

  def testInitializeTwice(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      with self.assertRaisesOpError("Table already initialized"):
        table.init.run()

  def testInitializationWithInvalidDimensions(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)

      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.KeyValueTensorInitializer(keys, values), default_val)

  def testMultipleSessions(self):
    # Start a server
    server = server_lib.Server(
        {
            "local0": ["localhost:0"]
        }, protocol="grpc", start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup.HashTable(
        lookup.KeyValueTensorInitializer(keys, values),
        default_val,
        name="t1")

    # Init the table in the first session.
    with session1:
      table.init.run()
      self.assertAllEqual(3, table.size().eval())

    # Init the table in the second session and verify that we do not get a
    # "Table already initialized" error.
    with session2:
      table.init.run()
      self.assertAllEqual(3, table.size().eval())

  def testHashTableInt32String(self):
    with self.cached_session():
      default_val = "n/a"
      keys = constant_op.constant([0, 1, 2], dtypes.int32)
      values = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.init.run()

      input_tensor = constant_op.constant([0, 1, -1])
      output = table.lookup(input_tensor)

      result = output.eval()
      self.assertAllEqual([b"brain", b"salad", b"n/a"], result)


class MutableHashTableOpTest(test.TestCase):

  def testMutableHashTable(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

      exported_keys, exported_values = table.export()
      self.assertAllEqual([None], exported_keys.get_shape().as_list())
      self.assertAllEqual([None], exported_values.get_shape().as_list())

      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(exported_keys.eval())
      sorted_values = np.sort(exported_values.eval())
      self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
      self.assertAllEqual([0, 1, 2], sorted_values)

  def testSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant(["b", "c", "d"], dtypes.string)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = -1
      table = lookup.MutableHashTable(
          dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)
      table.insert(
          constant_op.constant(["a", "c"], dtypes.string),
          constant_op.constant([12, 24], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                          dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], output.eval())

  def testSaveRestoreTensor(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([1, 2, 3], dtypes.int64)
      values = constant_op.constant([[1, 1], [2, 2], [3, 3]], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64, default_val, name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64, default_val, name="t1", checkpoint=True)
      table.insert(
          constant_op.constant([1, 3], dtypes.int64),
          constant_op.constant([[12, 12], [24, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant([0, 1, 2, 3, 4],
                                          dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[-1, -1], [1, 1], [2, 2], [3, 3], [-1, -1]],
                          output.eval())

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(20.0, name="v1")

    default_val = -1
    keys = constant_op.constant(["b", "c", "d"], dtypes.string)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup.MutableHashTable(
        dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)

    checkpoint = checkpointable.Checkpoint(table=table, v0=v0, v1=v1)
    self.evaluate([v0.initializer, v1.initializer])

    # Check that the parameter nodes have been initialized.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(0, self.evaluate(table.size()))
    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    save_path = checkpoint.save(save_prefix)
    del table, checkpoint, v0, v1

    v0 = variables.Variable(-1.0, name="v0")
    v1 = variables.Variable(-1.0, name="v1")
    default_val = -1
    table = lookup.MutableHashTable(
        dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)
    self.evaluate(table.insert(
        constant_op.constant(["a", "c"], dtypes.string),
        constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(table.size()))

    checkpoint = checkpointable.Checkpoint(table=table, v0=v0, v1=v1)

    # Restore the saved values in the parameter nodes.
    checkpoint.restore(save_path).run_restore_ops()
    # Check that the parameter nodes have been restored.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                        dtypes.string)
    output = table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  def testSharing(self):
    # Start a server to store the table state
    server = server_lib.Server(
        {
            "local0": ["localhost:0"]
        }, protocol="grpc", start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    table = lookup.MutableHashTable(
        dtypes.int64, dtypes.string, "-", name="t1")

    # Tensor value
    default_val = constant_op.constant([-1, -1], dtypes.int64)
    table2 = lookup.MutableHashTable(
        dtypes.int64, dtypes.int64, default_val, name="t2")

    # Populate the table in the first session
    with session1:
      self.assertAllEqual(0, table.size().eval())

      keys = constant_op.constant([11, 12], dtypes.int64)
      values = constant_op.constant(["a", "b"])
      table.insert(keys, values).run()
      self.assertAllEqual(2, table.size().eval())

      output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
      self.assertAllEqual([b"a", b"b", b"-"], output.eval())

      self.assertAllEqual(0, table2.size().eval())

      keys = constant_op.constant([11, 12], dtypes.int64)
      values = constant_op.constant([[0, 0], [1, 1]], dtypes.int64)
      table2.insert(keys, values).run()
      self.assertAllEqual(2, table2.size().eval())

      output = table2.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
      self.assertAllEqual([[0, 0], [1, 1], [-1, -1]], output.eval())

    # Verify that we can access the shared data from the second session
    with session2:
      self.assertAllEqual(2, table.size().eval())

      output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
      self.assertAllEqual([b"-", b"a", b"b"], output.eval())

      self.assertAllEqual(2, table2.size().eval())

      output = table2.fast_lookup(constant_op.constant([10, 11, 12], dtypes.int64))
      self.assertAllEqual([[0, 0], [0, 0], [1, 1]], output.eval())


  def testMutableHashTableExportInsert(self):
    with self.cached_session():
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
      table1 = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                       default_val)
      self.assertAllEqual(0, table1.size().eval())
      table1.insert(keys, values).run()
      self.assertAllEqual(3, table1.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      expected_output = [[0, 1], [2, 3], [-1, -1]]
      output1 = table1.lookup(input_string)
      self.assertAllEqual(expected_output, output1.eval())

      exported_keys, exported_values = table1.export()
      self.assertAllEqual(3, exported_keys.eval().size)
      self.assertAllEqual(6, exported_values.eval().size)

      # Populate a second table from the exported data
      table2 = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                       default_val)
      self.assertAllEqual(0, table2.size().eval())
      table2.insert(exported_keys, exported_values).run()
      self.assertAllEqual(3, table2.size().eval())

      # Verify lookup result is still the same
      output2 = table2.lookup(input_string)
      self.assertAllEqual(expected_output, output2.eval())

  def testMutableHashTableOfTensorsInvalidShape(self):
    with self.cached_session():
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)

      # Shape [6] instead of [3, 2]
      values = constant_op.constant([0, 1, 2, 3, 4, 5], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Shape [2,3] instead of [3, 2]
      values = constant_op.constant([[0, 1, 2], [3, 4, 5]], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Shape [2, 2] instead of [3, 2]
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Shape [3, 1] instead of [3, 2]
      values = constant_op.constant([[0], [2], [4]], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Valid Insert
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

  def testMutableHashTableInvalidDefaultValue(self):
    with self.cached_session():
      default_val = constant_op.constant([[-1, -1]], dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)
      with self.assertRaisesOpError("Default value must be a vector"):
        self.assertAllEqual(0, table.size().eval())

  def testMutableHashTableDuplicateInsert(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery", "brain"])
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([3, 1, -1], result)

  def testMutableHashTableFindHighRank(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(
          [["brain", "salad"], ["tank", "tarkus"]])
      output = table.lookup(input_string)
      self.assertAllEqual([2, 2], output.get_shape())

      result = output.eval()
      self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testMutableHashTableInsertHighRank(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant([["brain", "salad"], ["surgery", "tank"]])
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank", "tarkus"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, 3, -1], result)

  def testMutableHashTableOfTensorsFindHighRank(self):
    with self.cached_session():
      default_val = constant_op.constant([-1, -1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(
          [["brain", "salad"], ["tank", "tarkus"]])
      output = table.lookup(input_string)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = output.eval()
      self.assertAllEqual(
          [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def testMultipleMutableHashTables(self):
    with self.cached_session() as sess:
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)

      table1 = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                       default_val)
      table2 = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                       default_val)
      table3 = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                       default_val)
      table1.insert(keys, values).run()
      table2.insert(keys, values).run()
      table3.insert(keys, values).run()

      self.assertAllEqual(3, table1.size().eval())
      self.assertAllEqual(3, table2.size().eval())
      self.assertAllEqual(3, table3.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testMutableHashTableWithTensorDefault(self):
    with self.cached_session():
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testSignatureMismatch(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableHashTable(dtypes.string, dtypes.int64,
                                      default_val)

      # insert with keys of the wrong type
      with self.assertRaises(ValueError):
        table.insert(constant_op.constant([4, 5, 6]), values).run()

      # insert with values of the wrong type
      with self.assertRaises(ValueError):
        table.insert(keys, constant_op.constant(["a", "b", "c"])).run()

      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string_ref = variables.Variable("brain")
      input_int64_ref = variables.Variable(-1, dtype=dtypes.int64)
      variables.global_variables_initializer().run()

      # Ref types do not produce an insert signature mismatch.
      table.insert(input_string_ref, input_int64_ref).run()
      self.assertAllEqual(3, table.size().eval())

      # Ref types do not produce a lookup signature mismatch.
      self.assertEqual(-1, table.lookup(input_string_ref).eval())

      # lookup with keys of the wrong type
      input_string = constant_op.constant([1, 2, 3], dtypes.int64)
      with self.assertRaises(ValueError):
        table.lookup(input_string).eval()

      # default value of the wrong type
      with self.assertRaises(TypeError):
        lookup.MutableHashTable(dtypes.string, dtypes.int64, "UNK")

  def testMutableHashTableStringFloat(self):
    with self.cached_session():
      default_val = -1.5
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1.1, 2.2], dtypes.float32)
      table = lookup.MutableHashTable(dtypes.string, dtypes.float32,
                                      default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllClose([0, 1.1, default_val], result)

  def testMutableHashTableIntFloat(self):
    with self.cached_session():
      default_val = -1.0
      keys = constant_op.constant([3, 7, 0], dtypes.int64)
      values = constant_op.constant([7.5, -1.2, 9.9], dtypes.float32)
      table = lookup.MutableHashTable(dtypes.int64, dtypes.float32,
                                      default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant([7, 0, 11], dtypes.int64)
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllClose([-1.2, 9.9, default_val], result)

  def testMutableHashTableInt64String(self):
    with self.cached_session():
      default_val = "n/a"
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.MutableHashTable(dtypes.int64, dtypes.string,
                                      default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual((b"brain", b"salad", b"n/a"), result)

  def testToFromProto(self):
    with self.cached_session():
      v = lookup.MutableHashTable(key_dtype=dtypes.string,
                                  value_dtype=dtypes.int64,
                                  default_value=[-2]*3)
      w = lookup.MutableHashTable.from_proto(v.to_proto())
      self.assertAllEqual([-2, -2, -2], w.lookup("hello").eval())
      self.assertEquals(v._resource_handle, w._resource_handle)

  def testComplexSaveRestoreTensor(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    keys_values = None
    values_values = None
    update_ts_values_map = {}
    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64, default_val, name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())

      num_keys = 10
      target_keys = []
      insert_timestamp = time.time()
      for key in range(num_keys):
        keys = constant_op.constant([key], dtypes.int64)
        values = constant_op.constant([[key, key]], dtypes.int64)
        sub_value = constant_op.constant([[0, 0]], dtypes.int64)
        table.insert(keys, values).run()
        table.scatter_sub(keys, sub_value).run()  # update feature timestamp
        time.sleep(1)
        target_keys.append(key)

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(num_keys, len(values_values))
      self.assertEqual(num_keys, len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertAllEqual([key_value] * 2, values_values[i])
        self.assertTrue(update_ts_values[i][0] >= int(insert_timestamp),
                        str(update_ts_values[i][0]) + " vs " + str(insert_timestamp))
        self.assertTrue(update_ts_values[i][0] <= int(time.time()))
        update_ts_values_map[key_value] = update_ts_values[i]

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64, default_val, name="t1", checkpoint=True)
      table.insert(
          constant_op.constant([1, 3], dtypes.int64),
          constant_op.constant([[12, 12], [24, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(num_keys, table.size().eval())

      input_string = constant_op.constant(range(num_keys),
                                          dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[v] * 2 for v in range(num_keys)],
                          output.eval())

      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      new_update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(num_keys, len(values_values))
      self.assertEqual(num_keys, len(new_update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertEqual(new_update_ts_values[i], update_ts_values_map[key_value])

  def testEraseSaveRestoreTensor(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    keys_values = None
    values_values = None
    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = constant_op.constant([-1, -1], dtypes.int64)
      target_keys = [1, 2, 3]
      keys = constant_op.constant(target_keys, dtypes.int64)
      values = constant_op.constant([[1, 1], [2, 2], [3, 3]], dtypes.int64)
      sub_values = constant_op.constant([[0, 0], [0, 0], [0, 0]], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64, default_val, name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      insert_timestamp = time.time()
      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      table.scatter_sub(keys, sub_values).run()  # update feature timestamp
      self.assertAllEqual(3, table.size().eval())

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(3, len(values_values))
      self.assertEqual(3, len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertAllEqual([key_value] * 2, values_values[i])
        self.assertTrue(update_ts_values[i][0] >= int(insert_timestamp),
                        str(update_ts_values[i][0]) + " vs " + str(insert_timestamp))
        self.assertTrue(update_ts_values[i][0] <= int(time.time()))

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64, default_val, name="t1", checkpoint=True)
      table.insert(
          constant_op.constant([1, 3], dtypes.int64),
          constant_op.constant([[12, 12], [24, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant([0, 1, 2, 3, 4],
                                          dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[-1, -1], [1, 1], [2, 2], [3, 3], [-1, -1]],
                          output.eval())

      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      new_update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(3, len(values_values))
      self.assertEqual(3, len(new_update_ts_values))
      for i in keys_values:
        self.assertEqual(new_update_ts_values[i-1], update_ts_values[i-1])

      # clear
      threshold = int(time.time()) + 1
      table.erase_by_threshold(threshold).run()
      save.save(sess, save_path)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64, default_val, name="t1", checkpoint=True)
      table.insert(
          constant_op.constant([1, 3], dtypes.int64),
          constant_op.constant([[12, 12], [24, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())

  def testComplexEraseSaveRestoreTensor(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")
    meta_path = save_path + ".meta"
    feature_clean_num = 3

    def update_features(date, target_keys):
      with self.session(graph=ops.Graph()) as sess:
        default_val = constant_op.constant([-1, -1], dtypes.float32)
        table = lookup.MutableHashTable(dtypes.int64,
                                        dtypes.float32,
                                        default_val,
                                        feature_clean_num=feature_clean_num,
                                        name="t1")
        self.assertAllEqual(0, table.size().eval())

        save = saver.Saver()
        if os.path.exists(meta_path):
          save.restore(sess, save_path)
        for i in target_keys:
          keys = [i]
          table.lookup_and_insert(keys, 1.0).eval()

        for i in target_keys:
          keys = [i]
          sub_values = constant_op.constant([[0.5] * 2], dtypes.float32)
          table.scatter_sub(keys, sub_values).run()  # update feature timestamp

        val = save.save(sess, save_path)
        self.assertTrue(isinstance(val, six.string_types))
        self.assertEqual(save_path, val)

    def erase_features(timestamp, expected_keys):
      with self.session(graph=ops.Graph()) as sess:
        default_val = constant_op.constant([-1, -1], dtypes.float32)
        table = lookup.MutableHashTable(dtypes.int64,
                                        dtypes.float32,
                                        default_val,
                                        feature_clean_num=feature_clean_num,
                                        name="t1")
        self.assertAllEqual(0, table.size().eval())

        save = saver.Saver()
        save.restore(sess, save_path)
        table.erase_by_threshold(timestamp).run()
        new_keys, new_values = table.export()
        self.assertAllEqual(expected_keys, np.sort(new_keys.eval()))

    #
    # day 0
    #
    each_day_ts = [int(time.time())]
    update_features(date=0, target_keys=range(0, 12))
    time.sleep(1)
    #
    # day 1
    #
    each_day_ts.append(int(time.time()))
    update_features(date=1, target_keys=range(3, 12))
    time.sleep(1)
    #
    # day 2
    #
    each_day_ts.append(int(time.time()))
    update_features(date=2, target_keys=range(6, 12))
    time.sleep(1)
    #
    # day 3
    #
    each_day_ts.append(int(time.time()))
    update_features(date=3, target_keys=range(9, 15))
    time.sleep(1)
    #
    # before train in day 4, erase expired data.
    #
    erase_features(each_day_ts[1], expected_keys=range(9, 15))

  def testSegment(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    target_keys = range(1, 14)
    num_segment = 7
    update_ts_values_map = {}
    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64,
          default_val, hash_table_segments=num_segment,
          name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      insert_timestamp = time.time()
      self.assertAllEqual(0, table.size().eval())
      for i in target_keys:
        keys = constant_op.constant([i], dtypes.int64)
        values = constant_op.constant([[i] * 2], dtypes.int64)
        sub_values = constant_op.constant([[0] * 2], dtypes.int64)
        table.insert(keys, values).run()
        table.scatter_sub(keys, sub_values).run()  # update feature timestamp
        time.sleep(1)
      self.assertAllEqual(len(target_keys), table.size().eval())

      input_keys_raw = range(1, 14, 2)
      input_keys = constant_op.constant(input_keys_raw,
                                        dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllEqual([[i] * 2 for i in input_keys_raw], output.eval())

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(len(target_keys), len(values_values))
      self.assertEqual(len(target_keys), len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertAllEqual([key_value] * 2, values_values[i])
        self.assertTrue(update_ts_values[i][0] >= int(insert_timestamp),
                        str(update_ts_values[i][0]) + " vs " + str(insert_timestamp))
        self.assertTrue(update_ts_values[i][0] <= int(time.time()))
        update_ts_values_map[key_value] = update_ts_values[i]

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64,
          default_val, hash_table_segments=num_segment,
          name="t1", checkpoint=True)

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(len(target_keys), table.size().eval())

      input_keys_raw = range(1, 14, 2)
      input_keys = constant_op.constant(input_keys_raw,
                                        dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllEqual([[i] * 2 for i in input_keys_raw], output.eval())

      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      new_update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(len(target_keys), len(values_values))
      self.assertEqual(len(target_keys), len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertAllEqual([key_value] * 2, values_values[i])
        self.assertEqual(new_update_ts_values[i], update_ts_values_map[key_value])

      # clear
      threshold = int(time.time()) + 1
      table.erase_by_threshold(threshold).run()
      save.save(sess, save_path)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
          dtypes.int64, dtypes.int64,
          default_val, hash_table_segments=num_segment,
          name="t1", checkpoint=True)
      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())

  def testTruncatedNormalInitializer(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    target_keys = range(1, 100)
    num_segment = 7
    update_ts_values_map = {}
    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      target_mean = 0.0
      target_stddev = 0.5
      initializer = lookup.HashTableInitializer("truncated_normal", [4],
        target_mean, target_stddev)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        initializer=initializer,
        hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      insert_timestamp = time.time()
      self.assertAllEqual(0, table.size().eval())
      sub_value = constant_op.constant([[0., 0., 0., 0.]], dtypes.float32)
      for i in target_keys:
        keys = constant_op.constant([i], dtypes.int64)
        table.lookup_and_insert(keys, 1.0).eval()
        table.scatter_sub(keys, sub_value).run()  # update feature timestamp
      self.assertAllEqual(len(target_keys), table.size().eval())

      input_keys = constant_op.constant(target_keys,
        dtypes.int64)
      output = np.array(table.lookup(input_keys).eval()).flatten()
      mean = np.mean(output)
      stddev = np.std(output)
      self.assertAllClose(mean, target_mean, atol=1e-1)
      self.assertAllClose(stddev, target_stddev, atol=1e-1)

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(len(target_keys), len(values_values))
      self.assertEqual(len(target_keys), len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertTrue(update_ts_values[i][0] >= int(insert_timestamp),
                        str(update_ts_values[i][0]) + " vs " + str(insert_timestamp))
        self.assertTrue(update_ts_values[i][0] <= int(time.time()))
        update_ts_values_map[key_value] = [values_values[i], update_ts_values[i]]

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      target_mean = 0.0
      target_stddev = 0.5
      initializer = lookup.HashTableInitializer("truncated_normal", [4],
        target_mean, target_stddev)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        initializer=initializer,
        hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(len(target_keys), table.size().eval())

      # check store and restore correct
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(len(target_keys), len(values_values))
      self.assertEqual(len(target_keys), len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertAllEqual(values_values[i], update_ts_values_map[key_value][0])
        self.assertAllEqual(update_ts_values[i], update_ts_values_map[key_value][1])
        # check normal distribution
      new_keys = range(100, 200)
      for i in new_keys:
        keys = constant_op.constant([i], dtypes.int64)
        table.lookup_and_insert(keys, 1.0).eval()
      self.assertAllEqual(len(target_keys) + len(new_keys),
        table.size().eval())

      _, values = table.export()
      output = np.array(values.eval()).flatten()
      mean = np.mean(output)
      stddev = np.std(output)
      self.assertAllClose(mean, target_mean, atol=1e-1)
      self.assertAllClose(stddev, target_stddev, atol=1e-1)

  def testParallel(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    target_size = 1000
    num_iterations = 10
    target_keys = range(0, target_size)
    insert_keys = range(0, target_size * num_iterations)
    grads = np.array([[0.01, 0.01]] * target_size, dtype=np.float32)
    offset = len(target_keys)
    target_values = [0.9, 0.9]
    default_val = [1., 1.]
    num_segment = 7
    update_ts_values_map = {}
    with self.cached_session() as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_value = constant_op.constant(default_val, dtypes.float32)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        default_value, hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      lookup_insert_op = [
        table.lookup_and_insert(
          insert_keys[i*offset:(i+1)*offset], 1.0)
        for i in range(num_iterations)
      ]
      scatter_sub_op = table.scatter_sub(target_keys, grads)
      export_op = table.export_values_and_attrs()

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      def lookup_insert(ss, i):
        ss.run([lookup_insert_op[i]])

      def scatter_sub(ss):
        ss.run([scatter_sub_op])

      def export_func(ss):
        ss.run([export_op])

      insert_threads = [
        self.checkedThread(
          target=lookup_insert, args=(sess, i)) for i in range(num_iterations)
      ]
      scatter_sub_threads = [
        self.checkedThread(
          target=scatter_sub, args=(sess,)) for i in range(num_iterations)
      ]
      export_threads = [
        self.checkedThread(
          target=export_func, args=(sess,)) for i in range(num_iterations)
      ]

      insert_timestamp = time.time()
      for i in range(num_iterations):
        insert_threads[i].start()
        time.sleep(0.1)
        scatter_sub_threads[i].start()
        time.sleep(0.1)
        export_threads[i].start()
      time.sleep(0.1)
      for t in insert_threads:
        t.join()
      for t in scatter_sub_threads:
        t.join()
      for t in export_threads:
        t.join()

      self.assertAllEqual(len(insert_keys), table.size().eval())

      input_keys_raw = range(0, 100, 10)
      input_keys = constant_op.constant(input_keys_raw,
                                        dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllCloseAccordingToType(
        [target_values for _ in input_keys_raw], output.eval())

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(insert_keys, np.sort(keys_values))
      self.assertEqual(target_size * num_iterations, len(values_values))
      self.assertEqual(target_size * num_iterations, len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        if key_value < target_size:
          self.assertAllCloseAccordingToType(target_values, values_values[i])
        else:
          self.assertAllCloseAccordingToType(default_val, values_values[i])
        self.assertTrue(update_ts_values[i][0] >= int(insert_timestamp),
                        str(update_ts_values[i][0]) + " vs " + str(insert_timestamp))
        self.assertTrue(update_ts_values[i][0] <= int(time.time()))
        update_ts_values_map[key_value] = update_ts_values[i]

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = constant_op.constant([1., 1.], dtypes.float32)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        default_val, hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(len(insert_keys), table.size().eval())

      input_keys_raw = range(1, 10, 2)
      input_keys = constant_op.constant(input_keys_raw,
                                        dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllCloseAccordingToType(
        [target_values for _ in input_keys_raw], output.eval())

      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      new_update_ts_values = update_ts.eval()
      self.assertAllEqual(insert_keys, np.sort(keys_values))
      self.assertEqual(target_size * num_iterations, len(values_values))
      self.assertEqual(target_size * num_iterations, len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        if key_value < target_size:
          self.assertAllCloseAccordingToType(target_values, values_values[i])
        else:
          self.assertAllCloseAccordingToType(default_val, values_values[i])
        self.assertEqual(new_update_ts_values[i], update_ts_values_map[key_value])

  def testAttributes(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    key_start = 0
    key_end = 14
    target_keys = range(key_start, key_end)
    num_segment = 7
    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.int64,
        default_val, hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())
      for i in target_keys:
        keys = constant_op.constant([i], dtypes.int64)
        values = constant_op.constant([[i] * 2], dtypes.int64)
        sub_value = constant_op.constant([[0, 0]], dtypes.int64)
        table.insert(keys, values).run()
        table.scatter_sub(keys, sub_value).run()  # update feature timestamp
        time.sleep(1)
      self.assertAllEqual(len(target_keys), table.size().eval())

      input_keys_raw = range(key_start, key_end, 2)
      attr_default_value = [0., 0.]

      def gen_attr(attr_key, base_value):
        keys_with_attr = constant_op.constant(input_keys_raw,
                                              dtypes.int64)
        attr_values_raw = [[i + base_value] * 2 for i in input_keys_raw]
        attr_values = constant_op.constant(attr_values_raw, dtypes.float32)
        table.set_attr(keys_with_attr, attr_key, attr_values).run()
        wanted_attr_flags = [0] * len(target_keys)
        wanted_attr_values =\
          [attr_default_value] * len(target_keys)
        for i in input_keys_raw:
          wanted_attr_flags[i] = 1
          wanted_attr_values[i] = [i + base_value] * 2
        return wanted_attr_flags, wanted_attr_values

      # set attributes
      attr_key_m = "m"
      m_flags, m_values = gen_attr(attr_key_m, 0.1)
      attr_key_n = "n"
      n_flags, n_values = gen_attr(attr_key_n, 0.2)
      wanted_attr_keys = [attr_key_m, attr_key_n]
      wanted_attr_flags = [list(pair) for pair in zip(m_flags, n_flags)]
      wanted_attr_values = [
        pair[0] + pair[1] for pair in zip(m_values, n_values)]

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

      # check get_attr correctness
      attr_m_value = table.get_attr(target_keys, attr_key_m,
                                    attr_default_value)
      attr_n_value = table.get_attr(target_keys, attr_key_n,
                                    attr_default_value)
      self.assertAllClose(m_values, attr_m_value.eval())
      self.assertAllClose(n_values, attr_n_value.eval())

      keys, _, _, attr_keys, attr_flags, attr_values = \
        table.export_values_and_attrs()
      keys_values = np.array(keys.eval())
      attr_keys_values = attr_keys.eval()
      attr_flags_values = np.array(attr_flags.eval())
      attr_values_values = np.array(attr_values.eval())
      sorted_idx = np.argsort(keys_values)
      self.assertAllEqual(target_keys, keys_values[sorted_idx])
      self.assertAllEqual(wanted_attr_keys, attr_keys_values)
      self.assertAllEqual(wanted_attr_flags, attr_flags_values[sorted_idx])
      self.assertAllClose(wanted_attr_values, attr_values_values[sorted_idx])

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.int64,
        default_val, hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(len(target_keys), table.size().eval())

      input_keys_raw = range(1, 14, 2)
      input_keys = constant_op.constant(input_keys_raw,
                                        dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllEqual([[i] * 2 for i in input_keys_raw], output.eval())

      # check restore correctness
      keys, _, _, attr_keys, attr_flags, attr_values = \
        table.export_values_and_attrs()
      keys_values = np.array(keys.eval())
      attr_keys_values = attr_keys.eval()
      attr_flags_values = np.array(attr_flags.eval())
      attr_values_values = np.array(attr_values.eval())
      sorted_idx = np.argsort(keys_values)
      self.assertAllEqual(target_keys, keys_values[sorted_idx])
      self.assertAllEqual(wanted_attr_keys, attr_keys_values)
      self.assertAllEqual(wanted_attr_flags, attr_flags_values[sorted_idx])
      self.assertAllClose(wanted_attr_values, attr_values_values[sorted_idx])

      # check get_attr correctness
      attr_m_value = table.get_attr(target_keys, attr_key_m,
                                    attr_default_value)
      attr_n_value = table.get_attr(target_keys, attr_key_n,
                                    attr_default_value)
      self.assertAllClose(m_values, attr_m_value.eval())
      self.assertAllClose(n_values, attr_n_value.eval())

      # check get_attr with not-exist keys
      new_target_keys = target_keys + [key_end+1] * 10
      attr_m_value = table.get_attr(new_target_keys, attr_key_m,
                                    attr_default_value)
      attr_n_value = table.get_attr(new_target_keys, attr_key_n,
                                    attr_default_value)
      new_target_values = m_values + [attr_default_value] * 10
      self.assertAllClose(new_target_values, attr_m_value.eval())
      new_target_values = n_values + [attr_default_value] * 10
      self.assertAllClose(new_target_values, attr_n_value.eval())

      # check get_attr with not-exist attribute's key
      new_attr_default_value = [10.0, 10.0]
      attr_not_exist_value = table.get_attr(target_keys, "temp",
                                            new_attr_default_value)
      wanted_attr_values = [new_attr_default_value] * len(target_keys)
      self.assertAllClose(wanted_attr_values, attr_not_exist_value.eval())

      # update then get
      target_key = constant_op.constant([0], dtype=dtypes.int64)
      target_value = constant_op.constant([1.0] * 2, dtype=dtypes.float32)
      table.set_attr(target_key, attr_key_m, target_value).run()
      attr_m_value = table.get_attr(target_key, attr_key_m,
                                    new_attr_default_value)
      self.assertAllClose([[1.0] * 2], attr_m_value.eval())

      # clear
      threshold = int(time.time()) + 1
      table.erase_by_threshold(threshold).run()
      save.save(sess, save_path)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.int64,
        default_val, hash_table_segments=num_segment,
        name="t1", checkpoint=True)
      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())

      # check get_attr correctness
      attr_m_value = table.get_attr(target_keys, attr_key_m,
                                    new_attr_default_value)
      attr_n_value = table.get_attr(target_keys, attr_key_n,
                                    new_attr_default_value)
      wanted_attr_values = [new_attr_default_value] * len(target_keys)
      self.assertAllClose(wanted_attr_values, attr_m_value.eval())
      self.assertAllClose(wanted_attr_values, attr_n_value.eval())


class MutableHashTableImportFromFile(test.TestCase):
  def _createIdFile(self, basename, values):
    id_file = os.path.join(self.get_temp_dir(), basename)
    with open(id_file, "w") as f:
      for k, v in values.items():
        str_values = [str(t) for t in v]
        f.write(str(k) + "\t" + " ".join(str_values) + "\n")
    return id_file

  def test_import_from_file(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    target_data_map = {}
    num_data = 101
    for i in range(num_data):
      target_data_map[i] = [i * 0.1, i * 0.2, i * 0.3]
    target_keys = target_data_map.keys()
    id_file = self._createIdFile("embedding.txt", target_data_map)
    num_segment = 7
    with self.session(graph=ops.Graph()) as sess:
      target_mean = 0.0
      target_stddev = 0.5
      initializer = lookup.HashTableInitializer("truncated_normal", [3],
                                                target_mean, target_stddev)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        initializer=initializer,
        hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      save = saver.Saver()
      variables.global_variables_initializer().run()

      self.assertAllEqual(0, table.size().eval())

      table.import_from_file(id_file).run()

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(len(target_keys), len(values_values))
      self.assertEqual(len(target_keys), len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertAllCloseAccordingToType(target_data_map[key_value],
                                           values_values[i])

    with self.session(graph=ops.Graph()) as sess:
      target_mean = 0.0
      target_stddev = 0.5
      initializer = lookup.HashTableInitializer("truncated_normal", [3],
                                                target_mean, target_stddev)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        initializer=initializer,
        hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      self.assertAllEqual(len(target_keys), table.size().eval())
      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertAllEqual(target_keys, np.sort(keys_values))
      self.assertEqual(len(target_keys), len(values_values))
      self.assertEqual(len(target_keys), len(update_ts_values))
      for i in range(len(keys_values)):
        key_value = keys_values[i]
        self.assertAllCloseAccordingToType(target_data_map[key_value],
                                           values_values[i])

  def test_import_from_not_exist_file(self):
    id_file = self.get_temp_dir() + "/embedding.txt"
    num_segment = 7
    with self.session(graph=ops.Graph()) as sess:
      target_mean = 0.0
      target_stddev = 0.5
      initializer = lookup.HashTableInitializer("truncated_normal", [3],
                                                target_mean, target_stddev)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        initializer=initializer,
        hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      variables.global_variables_initializer().run()

      self.assertAllEqual(0, table.size().eval())

      table.import_from_file(id_file).run()

      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertEqual(0, len(keys_values))
      self.assertEqual(0, len(values_values))
      self.assertEqual(0, len(update_ts_values))

  def test_import_from_empty_file(self):
    id_file = self._createIdFile("embedding.txt", {})
    num_segment = 7
    with self.session(graph=ops.Graph()) as sess:
      target_mean = 0.0
      target_stddev = 0.5
      initializer = lookup.HashTableInitializer("truncated_normal", [3],
                                                target_mean, target_stddev)
      table = lookup.MutableHashTable(
        dtypes.int64, dtypes.float32,
        initializer=initializer,
        hash_table_segments=num_segment,
        name="t1", checkpoint=True)

      variables.global_variables_initializer().run()

      self.assertAllEqual(0, table.size().eval())

      table.import_from_file(id_file).run()

      keys, values, update_ts, _, _, _ = table.export_values_and_attrs()
      keys_values = keys.eval()
      values_values = values.eval()
      update_ts_values = update_ts.eval()
      self.assertEqual(0, len(keys_values))
      self.assertEqual(0, len(values_values))
      self.assertEqual(0, len(update_ts_values))


class MutableDenseHashTableOpTest(test.TestCase):

  def testBasic(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64, dtypes.int64, default_value=-1, empty_key=0)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant([11, 12, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testBasicBool(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([True, True, True], dtypes.bool)
      table = lookup.MutableDenseHashTable(
          dtypes.int64, dtypes.bool, default_value=False, empty_key=0)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant([11, 12, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([True, True, False], result)

  def testLookupUnknownShape(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64, dtypes.int64, default_value=-1, empty_key=0)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      placeholder_keys = array_ops.placeholder(dtypes.int64)
      output = table.lookup(placeholder_keys)
      self.assertAllEqual(None, output.get_shape())
      result = output.eval({placeholder_keys: [11, 12, 15]})
      self.assertAllEqual([0, 1, -1], result)

  def testMapStringToFloat(self):
    with self.cached_session():
      keys = constant_op.constant(["a", "b", "c"], dtypes.string)
      values = constant_op.constant([0.0, 1.1, 2.2], dtypes.float32)
      default_value = constant_op.constant(-1.5, dtypes.float32)
      table = lookup.MutableDenseHashTable(
          dtypes.string,
          dtypes.float32,
          default_value=default_value,
          empty_key="")
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["a", "b", "d"], dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllClose([0, 1.1, -1.5], result)

  def testMapInt64ToFloat(self):
    for float_dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        keys = constant_op.constant([11, 12, 13], dtypes.int64)
        values = constant_op.constant([0.0, 1.1, 2.2], float_dtype)
        default_value = constant_op.constant(-1.5, float_dtype)
        table = lookup.MutableDenseHashTable(
            dtypes.int64, float_dtype, default_value=default_value, empty_key=0)
        self.assertAllEqual(0, table.size().eval())

        table.insert(keys, values).run()
        self.assertAllEqual(3, table.size().eval())

        input_string = constant_op.constant([11, 12, 15], dtypes.int64)
        output = table.lookup(input_string)
        self.assertAllEqual([3], output.get_shape())

        result = output.eval()
        self.assertAllClose([0, 1.1, -1.5], result)

  def testVectorValues(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]],
                                    dtypes.int64)
      default_value = constant_op.constant([-1, -2, -3, -4], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=0,
          initial_num_buckets=4)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(4, len(table.export()[0].eval()))

      table.insert(
          constant_op.constant([14], dtypes.int64),
          constant_op.constant([[2, 3, 4, 5]], dtypes.int64)).run()
      self.assertAllEqual(4, table.size().eval())
      self.assertAllEqual(8, len(table.export()[0].eval()))

      input_string = constant_op.constant([11, 12, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual(
          [3, 4], output.shape, msg="Saw shape: %s" % output.shape)

      result = output.eval()
      self.assertAllEqual([[0, 1, 2, 3], [3, 4, 5, 6], [-1, -2, -3, -4]],
                          result)

  def testVectorKeys(self):
    with self.cached_session():
      keys = constant_op.constant([[0, 1], [1, 2], [1, 3]], dtypes.int64)
      values = constant_op.constant([10, 11, 12], dtypes.int64)
      empty_key = constant_op.constant([0, 3], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          initial_num_buckets=8)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      table.insert(
          constant_op.constant([[0, 0]], dtypes.int64),
          constant_op.constant([13], dtypes.int64)).run()
      self.assertAllEqual(4, table.size().eval())
      self.assertAllEqual(8, len(table.export()[0].eval()))

      input_string = constant_op.constant([[0, 1], [1, 2], [0, 2]],
                                          dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([10, 11, -1], result)

  def testResize(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          initial_num_buckets=4)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(4, len(table.export()[0].eval()))

      keys2 = constant_op.constant([13, 14, 15, 16, 17], dtypes.int64)
      values2 = constant_op.constant([3, 4, 5, 6, 7], dtypes.int64)

      table.insert(keys2, values2).run()
      self.assertAllEqual(7, table.size().eval())
      self.assertAllEqual(16, len(table.export()[0].eval()))

      keys3 = constant_op.constant([10, 11, 12, 13, 14, 15, 16, 17, 18],
                                   dtypes.int64)
      output = table.lookup(keys3)
      self.assertAllEqual([-1, 0, 1, 3, 4, 5, 6, 7, -1], output.eval())

  def testExport(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([1, 2, 3], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=100,
          initial_num_buckets=8)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      exported_keys, exported_values = table.export()
      self.assertAllEqual([None], exported_keys.get_shape().as_list())
      self.assertAllEqual([None], exported_values.get_shape().as_list())

      np_keys = exported_keys.eval()
      np_values = exported_values.eval()

      self.assertAllEqual(8, len(np_keys))
      self.assertAllEqual(8, len(np_values))

      # pair up keys and values, drop extra added dimension
      pairs = np.dstack((np_keys.flatten(), np_values.flatten()))[0]
      # sort by key
      pairs = pairs[pairs[:, 0].argsort()]
      self.assertAllEqual([[11, 1], [12, 2], [13, 3], [100, 0], [100, 0],
                           [100, 0], [100, 0], [100, 0]], pairs)

  def testSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      default_value = -1
      empty_key = 0
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32)

      save = saver.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          constant_op.constant([11, 14], dtypes.int64),
          constant_op.constant([12, 24], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], output.eval())

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    default_value = -1
    empty_key = 0
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    save_table = lookup.MutableDenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=32)

    save_checkpoint = checkpointable.Checkpoint(table=save_table)

    self.assertAllEqual(0, self.evaluate(save_table.size()))
    self.evaluate(save_table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(save_table.size()))
    self.assertAllEqual(32, len(self.evaluate(save_table.export()[0])))

    save_path = save_checkpoint.save(save_prefix)
    del save_table, save_checkpoint

    load_table = lookup.MutableDenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=64)
    self.evaluate(load_table.insert(
        constant_op.constant([11, 14], dtypes.int64),
        constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(load_table.size()))
    self.assertAllEqual(64, len(self.evaluate(load_table.export()[0])))

    restore_checkpoint = checkpointable.Checkpoint(table=load_table)

    # Restore the saved values in the parameter nodes.
    restore_checkpoint.restore(save_path).run_restore_ops()

    self.assertAllEqual(3, self.evaluate(load_table.size()))
    self.assertAllEqual(32, len(self.evaluate(load_table.export()[0])))

    input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
    output = load_table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  def testVectorSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "vector_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [13, 14]], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32)

      save = saver.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([[21, 22], [23, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -2], [4, 5], [-1, -2]],
                          output.eval())

  def testVectorScalarSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "vector_scalar_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [13, 14]], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=32)

      save = saver.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([3, 4], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([0, 1, -1, 2, -1], output.eval())

  def testReprobe(self):
    with self.cached_session():
      # Insert 6 keys into a table with 8 buckets.
      # The values are chosen to make sure collisions occur when using GCC STL
      keys = constant_op.constant([11, 12, 13, 19, 20, 21], dtypes.int64)
      values = constant_op.constant([51, 52, 53, 54, 55, 56], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          initial_num_buckets=8)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(6, table.size().eval())

      input_string = constant_op.constant([10, 11, 12, 13, 14, 19, 20, 21, 22],
                                          dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([9], output.get_shape())

      result = output.eval()
      self.assertAllEqual([-1, 51, 52, 53, -1, 54, 55, 56, -1], result)

  def testCustomEmptyKey(self):
    with self.cached_session():
      keys = constant_op.constant([11, 0, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.MutableDenseHashTable(
          dtypes.int64, dtypes.int64, default_value=-1, empty_key=12)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant([11, 0, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testErrors(self):
    with self.cached_session():
      table = lookup.MutableDenseHashTable(
          dtypes.int64, dtypes.int64, default_value=-1, empty_key=0)

      # Inserting the empty key returns an error
      keys = constant_op.constant([11, 0], dtypes.int64)
      values = constant_op.constant([0, 1], dtypes.int64)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "empty_key"):
        table.insert(keys, values).run()

      # Looking up the empty key returns an error
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "empty_key"):
        table.lookup(keys).eval()

      # Arbitrary tensors of keys are not supported
      keys = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
      values = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Expected key shape"):
        table.lookup(keys).eval()
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Expected key shape"):
        table.insert(keys, values).run()

      table2 = lookup.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=17,
          initial_num_buckets=12)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Number of buckets must be"):
        self.assertAllEqual(0, table2.size().eval())


class IndexTableFromFile(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_string_index_table_from_file(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_string_index_table_from_file_tensor_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_file = constant_op.constant(vocabulary_file)
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())
      self.assertEqual(1,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  def test_string_index_table_from_file_placeholder_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_placeholder = array_ops.placeholder(dtypes.string, [])
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_placeholder, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)

      feed_dict = {vocabulary_placeholder.name: vocabulary_file}
      lookup_ops.tables_initializer().run(feed_dict=feed_dict)
      self.assertAllEqual((1, 2, 3), ids.eval())
      self.assertEqual(0,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  def test_int32_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab2.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1,
          key_dtype=dtypes.int32)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_int64_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1,
          key_dtype=dtypes.int64)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_index_table_from_file_with_default_value(self):
    default_value = -42
    vocabulary_file = self._createVocabFile("f2i_vocab4.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, default_value), ids.eval())

  def test_index_table_from_file_with_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab5.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1000)
      ids = table.lookup(
          constant_op.constant(["salad", "surgery", "tarkus", "toccata"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual(
          (
              1,  # From vocabulary file.
              2,  # From vocabulary file.
              867,  # 3 + fingerprint("tarkus") mod 300.
              860),  # 3 + fingerprint("toccata") mod 300.
          ids.eval())

  def test_index_table_from_file_fails_with_empty_vocabulary_file_name(self):
    self.assertRaises(
        ValueError,
        lookup.index_table_from_file,
        vocabulary_file="")

  def test_index_table_from_file_fails_with_empty_vocabulary(self):
    self.assertRaises(
        ValueError,
        lookup.index_table_from_file,
        vocabulary_file=None)

  def test_index_table_from_file_with_vocab_size_too_small(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=2)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, -1, -1), ids.eval())
      self.assertEqual(2, table.size().eval())

  def test_index_table_from_file_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                              "Invalid vocab_size", table.init.run)

  def test_index_table_from_file_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab8.txt")

    self.assertRaises(
        ValueError,
        lookup.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, -1), ids.eval())
      self.assertEqual(3, table.size().eval())

  def test_index_table_from_file_with_invalid_hashers(self):
    vocabulary_file = self._createVocabFile("invalid_hasher.txt")
    with self.cached_session():
      with self.assertRaises(TypeError):
        lookup.index_table_from_file(
            vocabulary_file=vocabulary_file,
            vocab_size=3,
            num_oov_buckets=1,
            hasher_spec=1)

      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=3,
          num_oov_buckets=1,
          hasher_spec=lookup.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        constant_op.constant(["salad", "surgery", "tarkus"]))


class KeyValueTensorInitializerTest(test.TestCase):

  def test_string(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table = lookup.HashTable(init, default_value=-1)
      table.init.run()

  def test_int64(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup.KeyValueTensorInitializer(
          (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64)
      table = lookup.HashTable(init, default_value=-1)
      table.init.run()

  def test_int32(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup.KeyValueTensorInitializer(
          (42, 1, -1000), (0, 1, 2), dtypes.int32, dtypes.int64)
      table = lookup.HashTable(init, default_value=-1)
      with self.assertRaisesRegexp(
          errors_impl.OpError, "No OpKernel was registered"):
        table.init.run()


class IndexTableFromTensor(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_index_table_from_tensor_with_tensor_init(self):
    table = lookup.index_table_from_tensor(
        mapping=("brain", "salad", "surgery"), num_oov_buckets=1)

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(table.lookup(
            constant_op.constant(("salad", "surgery", "tarkus"))))
    else:
      # Reinitializing a table in eager should work.
      table = lookup.index_table_from_tensor(
          mapping=("brain", "salad", "surgery"), num_oov_buckets=1)
    self.evaluate(lookup_ops.tables_initializer())
    ids = table.lookup(constant_op.constant(("salad", "surgery", "tarkus")))
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int32_index_table_from_tensor_with_tensor_init(self):
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int32)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_int64_index_table_from_tensor_with_tensor_init(self):
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int64)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_index_table_from_tensor_with_default_value(self):
    default_value = -42
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=["brain", "salad", "surgery"], default_value=default_value)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, default_value), ids.eval())

  def test_index_table_from_tensor_missing_mapping(self):
    with self.cached_session():
      with self.assertRaisesRegexp(ValueError, "mapping must be specified"):
        lookup.index_table_from_tensor(mapping=None, num_oov_buckets=1)

  def test_index_table_from_tensor_empty_mapping(self):
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=np.array([], dtype=np.str_), num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "brain"]))
      self.assertRaises(errors_impl.OpError, ids.eval)
      with self.assertRaisesRegexp(
          errors_impl.OpError, "keys and values cannot be empty"):
        lookup_ops.tables_initializer().run()

  def test_index_table_from_tensor_with_invalid_hashers(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        lookup.index_table_from_tensor(
            mapping=["brain", "salad", "surgery"],
            num_oov_buckets=1,
            hasher_spec=1)

      table = lookup.index_table_from_tensor(
          mapping=["brain", "salad", "surgery"],
          num_oov_buckets=1,
          hasher_spec=lookup.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        constant_op.constant(["salad", "surgery", "tarkus"]))


class StringToIndexTest(test.TestCase):

  def test_string_to_index(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      feats = constant_op.constant(["salad", "surgery", "tarkus"])
      indices = lookup.string_to_index(feats, mapping=mapping_strings)

      self.assertRaises(errors_impl.OpError, indices.eval)
      lookup_ops.tables_initializer().run()

      self.assertAllEqual((1, 2, -1), indices.eval())

  def test_duplicate_entries(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["hello", "hello"])
      feats = constant_op.constant(["hello", "hola"])
      _ = lookup.string_to_index(feats, mapping=mapping_strings)

      self.assertRaises(errors_impl.OpError,
                        lookup_ops.tables_initializer().run)

  def test_string_to_index_with_default_value(self):
    default_value = -42
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      feats = constant_op.constant(["salad", "surgery", "tarkus"])
      indices = lookup.string_to_index(
          feats, mapping=mapping_strings, default_value=default_value)
      self.assertRaises(errors_impl.OpError, indices.eval)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, default_value), indices.eval())


class IndexToStringTableFromFileTest(test.TestCase):

  def _createVocabFile(self, basename):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["brain", "salad", "surgery"]) + "\n")
    return vocabulary_file

  def test_index_to_string_table(self):
    vocabulary_file = self._createVocabFile("i2f_vocab1.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file)
      features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          features.eval())

  def test_index_to_string_table_with_default_value(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          features.eval())

  def test_index_to_string_table_with_vocab_size_too_small(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=2,
          default_value=default_value)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", default_value, default_value),
                          features.eval())

  def test_index_to_string_table_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))

      self.assertRaises(errors_impl.OpError, features.eval)
      init = lookup_ops.tables_initializer()
      self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                              "Invalid vocab_size", init.run)

  def test_index_to_string_table_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))

      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", b"UNK"), features.eval())


class EmbeddingLookupHashVariableTest(test.TestCase):

  def testHashVariableWithoutPartition(self):
    with self.cached_session():
      table = lookup.MutableHashTable(key_dtype=dtypes.int64,
                                          value_dtype=dtypes.float64,
                                          default_value=[0, 0])

      ti = gen_lookup_ops.lookup_table_insert_v2(
        table.handle,
        keys=np.array([0, 3, 6, 9], dtype=np.int64),
        values=np.array([[0, 0], [3, 3], [6, 6], [9, 9]], dtype=np.float64))

      res = embedding_ops.embedding_lookup(table,
                                           np.array([3, 6, 9], dtype=np.int64))

      self.evaluate(ti)
      expected = [[3, 3], [6, 6], [9, 9]]
      self.assertAllEqual(self.evaluate(res), expected)

  def testHashVariableWithDefaultValues(self):
    with self.cached_session():
      table0 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=[1])
      table1 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=[1])
      table2 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=[1])
      table0.lookup([0, 1, 2, 3])
      table1.lookup([0, 1, 2, 3])
      table2.lookup([0, 1, 2])

      res = embedding_ops.embedding_lookup([table0, table1, table2], [3, 4, 5])

      expected = [[1], [1], [1]]
      self.assertAllEqual(self.evaluate(res), expected)

  def testHashVariableLookupAndInsert(self):
    embedding_size = 10
    default_value = [1.0] * embedding_size
    with self.cached_session():
      table0 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=default_value)
      table1 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=default_value)
      table2 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=default_value)
      res = embedding_ops.embedding_lookup([table0, table1, table2],
                                           [3, 4, 5], id_insert_probability=0.0)

      expected = [default_value, default_value, default_value]
      self.assertAllEqual(self.evaluate(res), expected)
      self.assertAllEqual(0, table0.size().eval())
      self.assertAllEqual(0, table1.size().eval())
      self.assertAllEqual(0, table2.size().eval())

      res = embedding_ops.embedding_lookup([table0, table1, table2],
                                           [3, 4, 5],
                                           id_insert_probability=1.0)
      self.assertAllEqual(self.evaluate(res), expected)
      self.assertAllEqual(3, table0.size().eval() +
                          table1.size().eval() + table2.size().eval())

  def testHashVariableLookupAndInsertRandomInit(self):
    embedding_size = 10
    with self.cached_session():
      initializer = lookup.HashTableInitializer(
        "truncated_normal", [embedding_size], 0.0, 1.0)
      table0 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       initializer=initializer)
      table1 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       initializer=initializer)
      table2 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       initializer=initializer)
      res = embedding_ops.embedding_lookup([table0, table1, table2],
                                           [3, 4, 5],
                                           id_insert_probability=0.0)

      expected = [[0] * embedding_size, [0] * embedding_size,
                  [0] * embedding_size]
      self.assertAllEqual(self.evaluate(res), expected)
      self.assertAllEqual(0, table0.size().eval())
      self.assertAllEqual(0, table1.size().eval())
      self.assertAllEqual(0, table2.size().eval())

      embedding_ops.embedding_lookup([table0, table1, table2],
                                     [3, 4, 5],
                                     id_insert_probability=1.0).eval()
      self.assertAllEqual(3, table0.size().eval() +
                          table1.size().eval() + table2.size().eval())

  def testHashVariablePartitionMod(self):
    with self.cached_session():
      table0 = lookup.MutableHashTable(key_dtype=dtypes.int64,
                                       value_dtype=dtypes.float64,
                                       default_value=[1])
      table1 = lookup.MutableHashTable(key_dtype=dtypes.int64,
                                       value_dtype=dtypes.float64,
                                       default_value=[1])
      table2 = lookup.MutableHashTable(key_dtype=dtypes.int64,
                                       value_dtype=dtypes.float64,
                                       default_value=[1])

      t0i = gen_lookup_ops.lookup_table_insert_v2(
        table0.handle,
        keys=np.array([0, 3, 6, 9], dtype=np.int64),
        values=np.array([[0], [3], [6], [9]], dtype=np.float64))
      t1i = gen_lookup_ops.lookup_table_insert_v2(
        table1.handle,
        keys=np.array([1, 4, 7, 10], dtype=np.int64),
        values=np.array([[1], [4], [7], [10]], dtype=np.float64))
      t2i = gen_lookup_ops.lookup_table_insert_v2(
        table2.handle,
        keys=np.array([2, 5, 8], dtype=np.int64),
        values=np.array([[2], [5], [8]], dtype=np.float64))

      res = embedding_ops.embedding_lookup([table0, table1, table2],
                                           np.array([3, 4, 5], dtype=np.int64))

      self.evaluate([t0i, t1i, t2i])
      expected = [[3], [4], [5]]
      self.assertAllEqual(self.evaluate(res), expected)

  def testHashVariablePartitionDiv(self):
    with self.cached_session():
      table0 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=[0])
      table1 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=[0])
      table2 = lookup.MutableHashTable(key_dtype=dtypes.int32,
                                       value_dtype=dtypes.float64,
                                       default_value=[0])

      t0i = gen_lookup_ops.lookup_table_insert_v2(
        table0.handle,
        keys=np.array([715827882], dtype=np.int32),
        values=np.array([[1]], dtype=np.float64))
      t1i = gen_lookup_ops.lookup_table_insert_v2(
        table1.handle,
        keys=np.array([1431655764], dtype=np.int32),
        values=np.array([[2]], dtype=np.float64))
      t2i = gen_lookup_ops.lookup_table_insert_v2(
        table2.handle,
        keys=np.array([1431655765], dtype=np.int32),
        values=np.array([[3]], dtype=np.float64))

      res = embedding_ops.embedding_lookup(
            [table0, table1, table2],
            np.array([715827882, 1431655764, 1431655765], dtype=np.int32),
            partition_strategy="div")

      self.evaluate([t0i, t1i, t2i])
      expected = [[1], [2], [3]]
      self.assertAllEqual(self.evaluate(res), expected)


class EmbeddingLookupSparseHashVariableTest(test.TestCase):

  def _keys_values(self, num_features, num_shards, value_dim):
    keys = []
    values = []
    for id in range(num_shards):
      t = id
      key_tuple = []
      while t < num_features:
        key_tuple.append(t)
        t += num_shards
      keys.append(key_tuple)
      value_tuple = []
      for key_value in keys[-1]:
        value_tuple.append([key_value + 1] * value_dim)
      values.append(value_tuple)

    return keys, values

  def testHashVariableWithoutPartition(self):
    shape = [3, 6]
    indices = [[0, 0], [0, 1], [1, 0], [2, 3]]
    ids = np.array([1, 3, 0, 1], dtype=np.int64)
    weights = np.array([2.0, .5, 1.0, 3.], dtype=np.float64)
    sp_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64))
    sp_weights = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(weights, dtypes.float32),
      constant_op.constant(shape, dtypes.int64))
    with self.cached_session():
      table = lookup.MutableHashTable(key_dtype=dtypes.int64,
                                      value_dtype=dtypes.float64,
                                      default_value=[0, 0, 0])
      ti = gen_lookup_ops.lookup_table_insert_v2(
        table.handle,
        keys=np.array([0, 1, 3], dtype=np.int64),
        values=np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]], dtype=np.float64))

      embedding_sum = embedding_ops.embedding_lookup_sparse(
        [table],
        sp_ids,
        sp_weights,
        combiner="sum")

      self.evaluate(ti)
      expected = [[6, 6, 6], [1, 1, 1], [6, 6, 6]]
      self.assertAllEqual(self.evaluate(embedding_sum), expected)

  def testModShard(self):
    num_shards = 5
    num_features = 13
    num_values = 4
    batch = 5
    nzdim = 3
    shape = [batch, nzdim]

    indices = []
    for i in range(batch):
      indices.extend([[i, x] for x in range(0, nzdim)])
    ids = np.random.randint(num_features, size=batch * nzdim)
    weights = np.random.rand(batch * nzdim)
    sum_expected = []
    mean_expected = []
    for i in range(batch):
      sum_tuple = np.zeros(num_values)
      count = .0
      for j in range(nzdim):
        idx = i * nzdim + j
        sum_tuple += [(ids[idx] + 1) * weights[idx]] * num_values
        count += weights[idx]
      sum_expected.append(sum_tuple)
      mean_expected.append(sum_tuple / count)

    sp_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64))
    sp_weights = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(weights, dtypes.float32),
      constant_op.constant(shape, dtypes.int64))

    with self.cached_session():
      keys, values = self._keys_values(num_features, num_shards, num_values)
      tables = []
      insert_ops = []
      for i in range(num_shards):
        tb = lookup.MutableHashTable(key_dtype=dtypes.int64,
                                     value_dtype=dtypes.float64,
                                     default_value=[0] * num_values)
        tables.append(tb)
        insert_ops.append(gen_lookup_ops.lookup_table_insert_v2(
          tables[i].handle,
          keys=np.array(keys[i], dtype=np.int64),
          values=np.array(values[i], dtype=np.float64)))

      embedding_sum = embedding_ops.embedding_lookup_sparse(
        tables,
        sp_ids,
        sp_weights,
        combiner="sum")

      embedding_mean = embedding_ops.embedding_lookup_sparse(
        tables,
        sp_ids,
        sp_weights,
        combiner="mean")

      self.evaluate(insert_ops)
      self.assertAllClose(self.evaluate(embedding_sum), sum_expected)
      self.assertAllClose(self.evaluate(embedding_mean), mean_expected)

class IndexToStringTableFromTensorTest(test.TestCase):

  def test_index_to_string_table_from_tensor(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings)

      indices = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      features = table.lookup(indices)
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()

      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          features.eval())

  def test_duplicate_entries(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["hello", "hello"])
      table = lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings)
      indices = constant_op.constant([0, 1, 4], dtypes.int64)
      features = table.lookup(indices)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"hello", b"hello", b"UNK"), features.eval())

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings, default_value=default_value)
      indices = constant_op.constant([1, 2, 4], dtypes.int64)
      features = table.lookup(indices)
      self.assertRaises(errors_impl.OpError, features.eval)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          features.eval())


class IndexToStringTest(test.TestCase):

  def test_index_to_string(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      indices = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      feats = lookup.index_to_string(indices, mapping=mapping_strings)

      self.assertRaises(errors_impl.OpError, feats.eval)
      lookup_ops.tables_initializer().run()

      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          feats.eval())

  def test_duplicate_entries(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["hello", "hello"])
      indices = constant_op.constant([0, 1, 4], dtypes.int64)
      feats = lookup.index_to_string(indices, mapping=mapping_strings)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"hello", b"hello", b"UNK"), feats.eval())

      self.assertRaises(errors_impl.OpError,
                        lookup_ops.tables_initializer().run)

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      indices = constant_op.constant([1, 2, 4], dtypes.int64)
      feats = lookup.index_to_string(
          indices, mapping=mapping_strings, default_value=default_value)
      self.assertRaises(errors_impl.OpError, feats.eval)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", default_value), feats.eval())


class InitializeTableFromFileOpTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  @test_util.run_in_graph_and_eager_modes
  def testInitializeStringTable(self):
    vocabulary_file = self._createVocabFile("one_column_1.txt")
    default_value = -1
    table = lookup.HashTable(
        lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                   lookup.TextFileIndex.WHOLE_LINE,
                                   dtypes.int64,
                                   lookup.TextFileIndex.LINE_NUMBER),
        default_value)
    self.evaluate(table.init)

    output = table.lookup(constant_op.constant(["brain", "salad", "tank"]))

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testInitializeInt64Table(self):
    vocabulary_file = self._createVocabFile(
        "one_column_int64.txt", values=("42", "1", "-1000"))

    with self.cached_session():
      default_value = -1
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.int64,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value)
      table.init.run()

      output = table.lookup(
          constant_op.constant((42, 1, 11), dtype=dtypes.int64))

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testInitializeIndexTable(self):
    vocabulary_file = self._createVocabFile("one_column_2.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup.TextFileIndex.LINE_NUMBER
      value_index = lookup.TextFileIndex.WHOLE_LINE
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.int64,
                                     key_index, dtypes.string, value_index),
          default_value)
      table.init.run()

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      output = table.lookup(input_values)

      result = output.eval()
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], result)

  def testMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 1
      value_index = 2

      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     key_index, dtypes.int64, value_index),
          default_value)
      table.init.run()

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([1, 5, 6], result)

  def testInvalidDataTypeInMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 2
      value_index = 1
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     key_index, dtypes.int64, value_index),
          default_value)
      with self.assertRaisesOpError("is not a valid"):
        table.init.run()

  def testInvalidDataType(self):
    vocabulary_file = self._createVocabFile("one_column_3.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup.TextFileIndex.WHOLE_LINE
      value_index = lookup.TextFileIndex.LINE_NUMBER

      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(vocabulary_file, dtypes.int64,
                                       key_index, dtypes.string,
                                       value_index), default_value)

  def testInvalidIndex(self):
    vocabulary_file = self._createVocabFile("one_column_4.txt")
    with self.cached_session():
      default_value = -1
      key_index = 1  # second column of the line
      value_index = lookup.TextFileIndex.LINE_NUMBER
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     key_index, dtypes.int64, value_index),
          default_value)

      with self.assertRaisesOpError("Invalid number of columns"):
        table.init.run()

  def testInitializeSameTableWithMultipleNodes(self):
    vocabulary_file = self._createVocabFile("one_column_5.txt")

    with self.cached_session() as sess:
      shared_name = "shared-one-columm"
      default_value = -1
      table1 = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)
      table2 = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)
      table3 = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)

      lookup_ops.tables_initializer().run()

      input_string = constant_op.constant(["brain", "salad", "tank"])

      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testInitializeTableWithNoFilename(self):
    with self.cached_session():
      default_value = -1
      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(
                "", dtypes.string, lookup.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup.TextFileIndex.LINE_NUMBER),
            default_value)

  def testInitializeWithVocabSize(self):
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      vocabulary_file1 = self._createVocabFile("one_column6.txt")
      table1 = lookup.HashTable(
          lookup.TextFileInitializer(
              vocabulary_file1,
              dtypes.string,
              lookup.TextFileIndex.WHOLE_LINE,
              dtypes.int64,
              lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)

      # Initialize from file.
      table1.init.run()
      self.assertEquals(vocab_size, table1.size().eval())

      vocabulary_file2 = self._createVocabFile("one_column7.txt")
      vocab_size = 5
      table2 = lookup.HashTable(
          lookup.TextFileInitializer(
              vocabulary_file2,
              dtypes.string,
              lookup.TextFileIndex.WHOLE_LINE,
              dtypes.int64,
              lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)
      with self.assertRaisesOpError("Invalid vocab_size"):
        table2.init.run()

      vocab_size = 1
      vocabulary_file3 = self._createVocabFile("one_column3.txt")
      table3 = lookup.HashTable(
          lookup.TextFileInitializer(
              vocabulary_file3,
              dtypes.string,
              lookup.TextFileIndex.WHOLE_LINE,
              dtypes.int64,
              lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)

      # Smaller vocab size reads only vocab_size records.
      table3.init.run()
      self.assertEquals(vocab_size, table3.size().eval())

  def testFeedVocabularyName(self):
    vocabulary_file = self._createVocabFile("feed_vocabulary.txt")

    with self.cached_session():
      default_value = -1
      table = lookup.HashTable(
          lookup.TextFileInitializer("old_file.txt", dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value)

      # Initialize with non existing file (old_file.txt) should fail.
      # TODO(yleon): Update message, which might change per FileSystem.
      with self.assertRaisesOpError("old_file.txt"):
        table.init.run()

      # Initialize the model feeding the vocabulary file.
      filenames = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
      table.init.run(feed_dict={filenames[0]: vocabulary_file})

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testInvalidFilenames(self):
    vocabulary_file = self._createVocabFile("filename_shape.txt")

    with self.cached_session():
      default_value = -1

      # Invalid data type
      other_type = constant_op.constant(1)
      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(
                other_type, dtypes.string, lookup.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup.TextFileIndex.LINE_NUMBER),
            default_value)

      # Non-scalar filename
      filenames = constant_op.constant([vocabulary_file, vocabulary_file])
      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(
                filenames, dtypes.string, lookup.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup.TextFileIndex.LINE_NUMBER),
            default_value)

  def testIdToStringTable(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = "UNK"
      vocab_size = 3
      table = lookup.HashTable(
          lookup.TextFileStringTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)

      table.init.run()

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)

      out = table.lookup(input_values)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], out.eval())
      self.assertEquals(vocab_size, table.size().eval())

  def testStringToIdTable(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)
      table.init.run()

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, -1], out.eval())
      self.assertEquals(vocab_size, table.size().eval())

  def testInt64ToIdTable(self):
    vocab_file = self._createVocabFile(
        "feat_to_id_3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
          default_value)
      table.init.run()

      out = table.lookup(
          constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64))
      self.assertAllEqual((0, 1, 2, -1), out.eval())
      self.assertEquals(vocab_size, table.size().eval())


class IdTableWithHashBucketsTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testStringIdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value),
          oov_buckets)

      table.init.run()

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, 3], out.eval())
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  def testInt32IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt", ("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
              default_value),
          oov_buckets,
          key_dtype=dtypes.int32)

      table.init.run()

      values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int32)

      out = table.lookup(values)
      self.assertAllEqual([0, 1, 2, 3], out.eval())
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  def testInt64IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
              default_value),
          oov_buckets)

      table.init.run()

      values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64)

      out = table.lookup(values)
      self.assertAllEqual([0, 1, 2, 3], out.eval())
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  def testStringIdTableWithOnlyHashBucket(self):
    with self.cached_session():
      oov_buckets = 5

      # Set a table that only uses hash buckets, for each input value returns
      # an id calculated by fingerprint("input") mod oov_buckets.
      table = lookup.IdTableWithHashBuckets(None, oov_buckets)
      table.init.run()

      values = constant_op.constant(("brain", "salad", "surgery"))

      out = table.lookup(values)
      self.assertAllEqual(
          [
              3,  # fingerprint("brain") mod 5.
              1,  # fingerprint("salad") mod 5.
              4  # fingerprint("surgery") mod 5
          ],
          out.eval())
      self.assertEquals(oov_buckets, table.size().eval())

  def testInt32IdTableWithOnlyHashBucket(self):
    with self.cached_session():
      oov_buckets = 5

      # Set a table that only uses hash buckets, for each input value returns
      # an id calculated by fingerprint("input") mod oov_buckets.
      table = lookup.IdTableWithHashBuckets(
          None, oov_buckets, key_dtype=dtypes.int32)
      table.init.run()

      input_string = constant_op.constant([42, 1, -1000], dtype=dtypes.int32)

      out = table.lookup(input_string)
      self.assertAllEqual(
          [
              1,  # fingerprint("42") mod 5.
              4,  # fingerprint("1") mod 5.
              2  # fingerprint("-1000") mod 5
          ],
          out.eval())
      self.assertEquals(oov_buckets, table.size().eval())

  def testFloat64IdTableWithOnlyHashBucket(self):
    with self.cached_session():
      with self.assertRaisesRegexp(TypeError, "Invalid key_dtype"):
        lookup.IdTableWithHashBuckets(
            None, num_oov_buckets=5, key_dtype=dtypes.float64)

  def testBoolIdTableWithOnlyHashBucket(self):
    with self.cached_session():
      with self.assertRaisesRegexp(TypeError, "Invalid key_dtype"):
        lookup.IdTableWithHashBuckets(
            None, num_oov_buckets=5, key_dtype=dtypes.bool)

  def testIdTableWithHashBucketsWithMultipleInitializers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    with self.cached_session() as sess:
      default_value = -1
      vocab_size = 3
      oov_buckets = 3

      vocab_table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)
      table1 = lookup.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=lookup.FastHashSpec,
          name="table1")

      table2 = lookup.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=lookup.StrongHashSpec((1, 2)),
          name="table2")

      lookup_ops.tables_initializer().run()

      input_string = constant_op.constant(
          ["fruit", "brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string)
      out2 = table2.lookup(input_string)

      out1, out2 = sess.run([out1, out2])
      self.assertAllEqual([5, 0, 1, 2, 5], out1)
      self.assertAllEqual([5, 0, 1, 2, 3], out2)
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())
      test_util.assert_ops_in_graph({
          "table1_Lookup/hash_bucket": "StringToHashBucketFast",
          "table2_Lookup/hash_bucket": "StringToHashBucketStrong",
      }, sess.graph)

  def testIdTableWithHashBucketsInitializationAcrossSessions(self):
    vocab_file = self._createVocabFile("feat_to_id_5.txt")
    shared_name = "across-sessions"
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table1 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value,
              shared_name=shared_name),
          oov_buckets)

      table1.init.run()

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string_1)

      self.assertAllEqual([0, 1, 2, 3], out1.eval())
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())

    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1

      # Underlying lookup table already initialized in previous session.
      # No need to call table2.init.run()
      table2 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value,
              shared_name=shared_name),
          oov_buckets)

      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out2 = table2.lookup(input_string_2)

      self.assertAllEqual([3, 1, 3], out2.eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())

  def testIdTableWithHashBucketsWithMultipleInitializersDifferentDefault(self):
    vocab_file = self._createVocabFile("feat_to_id_6.txt")
    with self.cached_session() as sess:
      default_value1 = -1
      vocab_size = 3
      oov_buckets = 0
      table1 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value1),
          oov_buckets)

      default_value2 = -2
      table2 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value2),
          oov_buckets)

      lookup_ops.tables_initializer().run()

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])
      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out1 = table1.lookup(input_string_1)
      out2 = table2.lookup(input_string_2)

      out1, out2 = sess.run([out1, out2])
      self.assertAllEqual([0, 1, 2, -1], out1)
      self.assertAllEqual([-2, 1, -2], out2)
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())

  def testSparseTensor(self):
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                               dtypes.string),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=3),
              -1),
          1)
      table.init.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.KeyValueTensorInitializer(
                  (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64),
              -1),
          1,
          key_dtype=dtypes.int32)
      table.init.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt64SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.KeyValueTensorInitializer(
                  (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64),
              -1),
          1,
          key_dtype=dtypes.int64)
      table.init.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  def testIdTableWithHashBucketsWithInvalidHashers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      lookup_table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)

      with self.assertRaises(TypeError):
        lookup.IdTableWithHashBuckets(
            lookup_table, oov_buckets, hasher_spec=1)

      table = lookup.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=lookup.HasherSpec("my-awesome-hash", None))

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      with self.assertRaises(ValueError):
        table.lookup(input_string)

      with self.assertRaises(ValueError):
        table = lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup.StrongHashSpec([]))

      with self.assertRaises(ValueError):
        table = lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup.StrongHashSpec([1, 2, 3]))

      with self.assertRaises(TypeError):
        table = lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup.StrongHashSpec([None, 2]))


class MutableHashTableBenchmark(test.Benchmark):

  def _create_table(self):
    return lookup.MutableHashTable(dtypes.int64, dtypes.float32, 0.0)

  def benchmark_single_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable(1.0)
    insert = table.insert(0, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) == 1

  def benchmark_many_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    c = counter.Counter().make_one_shot_iterator().get_next()
    value = variables.Variable(1.0)
    insert = table.insert(c, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) >= 10000

  def benchmark_single_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) == 32

  def benchmark_many_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    c = counter.Counter().make_one_shot_iterator().get_next()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(32 * c + list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) >= 1000*32


class MutableDenseHashTableBenchmark(MutableHashTableBenchmark):

  def _create_table(self):
    return lookup.MutableDenseHashTable(
        dtypes.int64, dtypes.float32, default_value=0.0, empty_key=-1)


class BatchGradTest(test.TestCase):

  def testSimpleBatchGrad(self):
    with self.cached_session():
      inputs_indices = [1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6,]
      batch_size = len(inputs_indices)
      ids, idx, counts = array_ops.unique_with_counts(inputs_indices)
      grad_indices = [1, 3, 5]
      grad_values = [[10., 10.], [30., 30.], [60., 60.]]
      grad = ops.IndexedSlices(grad_values, grad_indices)

      ret = gen_lookup_ops.lookup_table_batch_grad_v2(
          grad.indices, grad.values, batch_size, ids, counts)

      expected_output = [[55., 55.], [330., 330.], [220., 220.]]
      self.assertAllEqual(expected_output, ret.eval())


if __name__ == "__main__":
  test.main()
