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
"""Lookup table operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import array_ops
# pylint: disable=unused-import
from tensorflow.python.ops.lookup_ops import FastHashSpec
from tensorflow.python.ops.lookup_ops import HasherSpec
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import IdTableWithHashBuckets
from tensorflow.python.ops.lookup_ops import index_table_from_file
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow.python.ops.lookup_ops import InitializableLookupTableBase
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer
from tensorflow.python.ops.lookup_ops import LookupInterface
from tensorflow.python.ops.lookup_ops import StrongHashSpec
from tensorflow.python.ops.lookup_ops import TableInitializerBase
from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer
from tensorflow.python.ops.lookup_ops import TextFileIndex
from tensorflow.python.ops.lookup_ops import TextFileInitializer
from tensorflow.python.ops.lookup_ops import TextFileStringTableInitializer
# pylint: enable=unused-import
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util.deprecation import deprecated


@deprecated("2017-04-10", "Use `index_table_from_file`.")
def string_to_index_table_from_file(vocabulary_file=None,
                                    num_oov_buckets=0,
                                    vocab_size=None,
                                    default_value=-1,
                                    hasher_spec=FastHashSpec,
                                    name=None):
  return index_table_from_file(
      vocabulary_file, num_oov_buckets, vocab_size, default_value, hasher_spec,
      key_dtype=dtypes.string, name=name)


@deprecated("2017-04-10", "Use `index_table_from_tensor`.")
def string_to_index_table_from_tensor(mapping,
                                      num_oov_buckets=0,
                                      default_value=-1,
                                      hasher_spec=FastHashSpec,
                                      name=None):
  with ops.name_scope(name, "string_to_index") as scope:
    mapping = ops.convert_to_tensor(mapping)
  if dtypes.string != mapping.dtype.base_dtype:
    raise ValueError("string_to_index_table_from_tensor requires string.")
  return index_table_from_tensor(
      mapping, num_oov_buckets, default_value, hasher_spec, name=scope)


def index_table_from_tensor(mapping,
                            num_oov_buckets=0,
                            default_value=-1,
                            hasher_spec=FastHashSpec,
                            dtype=dtypes.string,
                            name=None):
  """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a string `mapping` 1-D tensor
  where each element is a key and corresponding index within the tensor is the
  value.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is `[mapping size, mapping size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` or `table.init.run()` once.

  Elements in `mapping` cannot have duplicates, otherwise when executing the
  table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  mapping_strings = tf.constant(["emerson", "lake", "palmer"])
  table = tf.contrib.lookup.index_table_from_tensor(
      mapping=mapping_strings, num_oov_buckets=1, default_value=-1)
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = table.lookup(features)
  ...
  tf.tables_initializer().run()

  ids.eval()  ==> [0, 1, 3, 2]
  ```

  Args:
    mapping: A 1-D `Tensor` that specifies the mapping of keys to indices. The
      type of this object must be castable to `dtype`.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignment of out-of-vocabulary buckets.
    dtype: The type of values passed to `lookup`. Only string and integers are
      supported.
    name: A name for this op (optional).

  Returns:
    The lookup table to map an input `Tensor` to index `int64` `Tensor`.

  Raises:
    ValueError: If `mapping` is invalid.
    ValueError: If `num_oov_buckets` is negative.
  """
  if mapping is None:
    raise ValueError("mapping must be specified.")
  return lookup_ops.index_table_from_tensor(
      vocabulary_list=mapping,
      num_oov_buckets=num_oov_buckets,
      default_value=default_value,
      hasher_spec=hasher_spec,
      dtype=dtype,
      name=name)


@deprecated(
    "2017-01-07", "This op will be removed after the deprecation date. "
    "Please switch to index_table_from_tensor and call the lookup "
    "method of the returned table.")
def string_to_index(tensor, mapping, default_value=-1, name=None):
  """Maps `tensor` of strings into `int64` indices based on `mapping`.

  This operation converts `tensor` of strings into `int64` indices.
  The mapping is initialized from a string `mapping` tensor where each element
  is a key and corresponding index within the tensor is the value.

  Any entry in the input which does not have a corresponding entry in 'mapping'
  (an out-of-vocabulary entry) is assigned the `default_value`

  Elements in `mapping` cannot be duplicated, otherwise the initialization
  will throw a FailedPreconditionError.

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` once.

  For example:

  ```python
  mapping_strings = tf.constant(["emerson", "lake", "palmer"])
  feats = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = tf.contrib.lookup.string_to_index(
      feats, mapping=mapping_strings, default_value=-1)
  ...
  tf.tables_initializer().run()

  ids.eval()  ==> [0, 1, -1, 2]
  ```

  Args:
    tensor: A 1-D input `Tensor` with the strings to map to indices.
    mapping: A 1-D string `Tensor` that specifies the mapping of strings to
      indices.
    default_value: The `int64` value to use for out-of-vocabulary strings.
      Defaults to -1.
    name: A name for this op (optional).

  Returns:
    The mapped indices. It has the same shape and tensor type (dense or sparse)
    as `tensor`.
  """
  table = index_table_from_tensor(
      mapping=mapping, default_value=default_value, name=name)
  return table.lookup(tensor)


def index_to_string_table_from_tensor(mapping, default_value="UNK", name=None):
  """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The mapping is initialized from a string `mapping` 1-D `Tensor` where
  each element is a value and the corresponding index within the tensor is the
  key.

  Any input which does not have a corresponding index in 'mapping'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` or `table.init.run()` once.

  Elements in `mapping` cannot have duplicates, otherwise when executing the
  table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  mapping_string = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      mapping_string, default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    mapping: A 1-D string `Tensor` that specifies the strings to map from
      indices.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `mapping` is not set.
  """

  if mapping is None:
    raise ValueError("mapping must be specified.")

  return lookup_ops.index_to_string_table_from_tensor(
      vocabulary_list=mapping, default_value=default_value, name=name)


@deprecated(
    "2017-01-07", "This op will be removed after the deprecation date. "
    "Please switch to index_to_string_table_from_tensor and call the lookup "
    "method of the returned table.")
def index_to_string(tensor, mapping, default_value="UNK", name=None):
  """Maps `tensor` of indices into string values based on `mapping`.

  This operation converts `int64` indices into string values. The mapping is
  initialized from a string `mapping` tensor where each element is a value and
  the corresponding index within the tensor is the key.

  Any input which does not have a corresponding index in 'mapping'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` once.

  For example:

  ```python
  mapping_string = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  values = tf.contrib.lookup.index_to_string(
      indices, mapping=mapping_string, default_value="UNKNOWN")
  ...
  tf.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    tensor: A `int64` `Tensor` with the indices to map to strings.
    mapping: A 1-D string `Tensor` that specifies the strings to map from
      indices.
    default_value: The string value to use for out-of-vocabulary indices.
    name: A name for this op (optional).

  Returns:
    The strings values associated to the indices. The resultant dense
    feature value tensor has the same shape as the corresponding `indices`.
  """
  table = index_to_string_table_from_tensor(
      mapping=mapping, default_value=default_value, name=name)
  return table.lookup(tensor)


def mutable_hash_table_batch_grads(grads_and_vars, id_value_map, batch_size):
  """Modify mutable hash table gradients.

  Modify mutable hash table gradietns:
         grad = (grad / #(occurrences in batch)) * batch_size
  (related paper: https://arxiv.org/pdf/1711.01761.pdf)

  Args:
    grads_and_vars: result of compute_gradients.
    id_value_map: table_name->ids in the batch.
    batch_size: batch size.

  Returns:
    Modified grads_and_vars used for apply_gradient.

  Raises:
    ValueError: If MutableHashTable not in id_value_map.
  """
  new_grads_and_vars = []
  id_counts_map = {}
  for grad, var in grads_and_vars:
    if grad is not None and isinstance(var, MutableHashTable):
      if var.name not in id_value_map:
        raise ValueError("MutableHashTable {} not in id_value_map"
                         .format(var.name))
      id_values = id_value_map[var.name]
      if not isinstance(id_values, ops.Tensor):
        raise ValueError("id_values of {} in id_value_map must be a Tensor"
                         .format(var.name))
      id_value_name = id_values.name
      if id_value_name in id_counts_map:
        ids = id_counts_map[id_value_name][0]
        counts = id_counts_map[id_value_name][1]
      else:
        ids, _, counts = array_ops.unique_with_counts(id_values,
                                                      out_idx=id_values.dtype)
        id_counts_map[id_value_name] = (ids, counts)
      g_values = gen_lookup_ops.lookup_table_batch_grad_v2(grad.indices,
                                                           grad.values,
                                                           batch_size,
                                                           ids,
                                                           counts)
      new_grads_and_vars.append((ops.IndexedSlices(g_values, grad.indices),
                                 var))
    else:
      new_grads_and_vars.append((grad, var))

  return new_grads_and_vars


class HashTableInitializer:
  def __init__(self,
               initializer_type,
               shape,
               *args):
    self._initializer_type = initializer_type
    self._shape = shape
    self._mean = None
    self._stddev = None
    self._seed = None

    if initializer_type == "truncated_normal":
      self._mean = args[0]
      self._stddev = args[1]
      if len(args) == 3:
        self._seed = args[2]
    else:
      raise ValueError("Unsupported initializer_type!")


class MutableHashTable(LookupInterface, checkpointable.CheckpointableBase):
  """A generic mutable hash table implementation.

  Data can be inserted by calling the insert method. It does not support
  initialization via the init method.

  Example usage:

  ```python
  table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                             value_dtype=tf.int64,
                                             default_value=-1)
  sess.run(table.insert(keys, values))
  out = table.lookup(query_keys)
  print(out.eval())
  ```
  """

  def __init__(self,
               key_dtype,
               value_dtype,
               default_value=None,
               initializer=None,
               hash_table_segments=1,
               tensor_cache_size=100,
               feature_clean_num=1,
               shared_name=None,
               checkpoint=True,
               trainable=True,
               variable_def=None,
               import_scope=None,
               name="MutableHashTable"):
    if variable_def is None:
      self._init_from_args(key_dtype=key_dtype,
                           value_dtype=value_dtype,
                           default_value=default_value,
                           initializer=initializer,
                           hash_table_segments=hash_table_segments,
                           tensor_cache_size=tensor_cache_size,
                           feature_clean_num=feature_clean_num,
                           shared_name=shared_name,
                           name=name,
                           checkpoint=checkpoint,
                           trainable=trainable)
    else:
      self._init_from_proto(variable_def, import_scope)

  def _init_from_proto(self, variable_def, import_scope=None):
    """Initializes from `VariableDef` proto."""
    # Note that init_from_proto is currently not supported in Eager mode.
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError("Trying to restore Variable as MutableHashTable.")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._resource_handle = g.as_graph_element(
      ops.prepend_name_scope(
        variable_def.variable_name, import_scope=import_scope))
    self._table_name = self._resource_handle.op.name.split("/")[-1]
    self._trainable = getattr(variable_def, "trainable", True)
    shape_value = self._resource_handle.op.get_attr("value_shape")
    if shape_value is None:
      self._value_shape = ()
    else:
      self._value_shape = tensor_shape.as_shape(shape_value)

    if (hasattr(variable_def, "initial_value_name") and
            variable_def.initial_value_name):
      self._default_value = g.as_graph_element(
        ops.prepend_name_scope(variable_def.initial_value_name,
                               import_scope=import_scope))

    super(MutableHashTable, self).__init__(
      self._resource_handle.op.get_attr("key_dtype"),
      self._resource_handle.op.get_attr("value_dtype"),
      self._table_name)
    if self._trainable:
      ops.add_to_collections(ops.GraphKeys.TRAINABLE_VARIABLES, self)
      self._in_graph_mode = not context.executing_eagerly()
      self._dtype = dtypes.resource

  def _init_from_args(self,
                      key_dtype,
                      value_dtype,
                      default_value=None,
                      initializer=None,
                      hash_table_segments=1,
                      tensor_cache_size=100,
                      feature_clean_num=1,
                      shared_name=None,
                      name="MutableHashTable",
                      checkpoint=True,
                      trainable=True):
    """Creates an empty `MutableHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      hash_table_segments: The num of hash_table concurrent segments.
      tensor_cache_size: Cache size of initial tensors in each segments.
      shared_name: If non-empty, this table will be shared under
        the given name across multiple sessions.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.

    Returns:
      A `MutableHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    with ops.name_scope(name, "MutabalHashTable") as name:
      if initializer == None:
        self._default_value_ref = default_value
        self._temp_default_value = ops.convert_to_tensor(default_value,
                                                         dtype=value_dtype)
        self._value_shape = self._temp_default_value.get_shape()
        self._initializer_type = ""
        self._init_mean = 0
        self._init_stddev = 0
        self._seed = None
      else:
        self._default_value_ref = np.zeros(shape=initializer._shape)
        self._temp_default_value = ops.convert_to_tensor(
                self._default_value_ref, dtype=value_dtype)
        self._value_shape = tensor_shape.TensorShape(initializer._shape)
        self._initializer_type = initializer._initializer_type
        self._init_mean = initializer._mean
        self._init_stddev = initializer._stddev
        self._seed = initializer._seed
      self._hash_table_segments = hash_table_segments
      self._tensor_cache_size = tensor_cache_size
      self._feature_clean_num = feature_clean_num
      self._checkpoint = checkpoint

      if context.executing_eagerly() and shared_name is None:
        # TODO(allenl): This will leak memory due to kernel caching by the
        # shared_name attribute value (but is better than the alternative of
        # sharing everything by default when executing eagerly; hopefully
        # creating tables in a loop is uncommon).
        shared_name = "table_%d" % (ops.uid(),)
      self._shared_name = shared_name

      # The table must be shared if checkpointing is requested for multi-worker
      # training to work correctly. Use the node name if no shared_name has been
      # explicitly specified.
      use_node_name_sharing = self._checkpoint and self._shared_name is None
      if self._value_shape.ndims == 0:
        table_ref = gen_lookup_ops.mutable_hash_table_v2(
          shared_name=self._shared_name,
          use_node_name_sharing=use_node_name_sharing,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          name=name)
      else:
        seed1, seed2 = random_seed.get_seed(self._seed)
        table_ref = gen_lookup_ops.mutable_hash_table_of_tensors_v2(
          default_value=self._temp_default_value,
          shared_name=self._shared_name,
          use_node_name_sharing=use_node_name_sharing,
          key_dtype=key_dtype,
          value_shape=self._value_shape,
          distribution=self._initializer_type,
          seed=seed1,
          seed2=seed2,
          mean=self._init_mean,
          stddev=self._init_stddev,
          hash_table_segments=self._hash_table_segments,
          tensor_cache_size=self._tensor_cache_size,
          update_ts_capacity=self._feature_clean_num,
          name=name)

      # we must ensure the default_value and the resource are placed together
      with ops.colocate_with(table_ref):
        self._default_value = ops.convert_to_tensor(self._default_value_ref,
                                                    dtype=value_dtype)
        self._default_value_ref = None

      if context.executing_eagerly():
        self._table_name = None
      else:
        self._table_name = table_ref.op.name.split("/")[-1]

      self._resource_handle = table_ref
      super(MutableHashTable, self).__init__(key_dtype, value_dtype,
                                             self._table_name)
      if checkpoint:
        saveable = MutableHashTable._Saveable(self, name)
        if not context.executing_eagerly():
          ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

      self._trainable = trainable
      if trainable:
        ops.add_to_collections(ops.GraphKeys.TRAINABLE_VARIABLES, self)
        self._in_graph_mode = not context.executing_eagerly()
        self._dtype = dtypes.resource

  @property
  def value_shape(self):
    return self._value_shape

  @property
  def key_dtype(self):
    return self._key_dtype

  @property
  def value_dtype(self):
    return self._value_dtype

  @property
  def hash_table_segments(self):
    return self._hash_table_segments

  @property
  def tensor_cache_size(self):
    return self._tensor_cache_size

  @property
  def handle(self):
    return self._resource_handle

  @property
  def graph(self):
    return self._resource_handle.graph

  @property
  def op(self):
    return self._resource_handle.op

  @property
  def name(self):
    return self._name

  @property
  def device(self):
    """The device of this hash table."""
    return self._resource_handle.device

  @property
  def dtype(self):
    return self._dtype

  @property
  def trainable(self):
    return self._trainable

  def _should_act_as_resource_variable(self):
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def to_proto(self, export_scope=None):
    """Converts a `MutableHashVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if context.executing_eagerly():
      raise RuntimeError("to_proto not supported in EAGER mode.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      var_def.is_resource = True
      var_def.trainable = self.trainable
      var_def.resource_type = ops.GraphKeys.MUTABLE_HASH_TABLE
      var_def.initial_value_name = ops.strip_name_scope(
        self._default_value.name, export_scope)
      return var_def
    else:
      return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("from_proto not supported in EAGER mode.")
    return MutableHashTable(
      key_dtype=None, value_dtype=None, default_value=None,
      variable_def=variable_def, import_scope=import_scope)

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self._name,
                        [self._resource_handle]) as name:
      with ops.colocate_with(self._resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(
          self._resource_handle, name=name)

  def remove(self, keys, name=None):
    """Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    if keys.dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    with ops.name_scope(
            name, "%s_lookup_table_remove" % self._name,
            (self._resource_handle, keys, self._default_value)) as name:
      # pylint: disable=protected-access
      op = gen_lookup_ops.lookup_table_remove_v2(
        self._resource_handle, keys, name=name)

    return op

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(name, "%s_lookup_table_find" % self._name,
                        (self._resource_handle, keys, self._default_value)) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self._resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(
            self._resource_handle, keys, self._default_value, name=name)
    return values

  def lookup_and_insert(self, keys, probability, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      probability: probability to insert element.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(
            name, "%s_lookup_table_find_and_insert" % self._name,
            (self._resource_handle, keys, self._default_value)) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self._resource_handle):
        values = gen_lookup_ops.lookup_table_find_and_insert_v2(
          self._resource_handle, keys, self._default_value,
          probability, name=name)
    return values

  def fast_lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    The keys not found in table won't be inserted, which is different from
    lookup(self, keys, name=None).

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(
            name, "%s_lookup_table_fast_find" % self._name,
            (self._resource_handle, keys, self._default_value)) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self._resource_handle):
        values = gen_lookup_ops.lookup_table_fast_find_v2(
            self._resource_handle, keys, self._default_value, name=name)
    return values

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the
        table's key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_insert" % self._name,
                        [self._resource_handle, keys, values]) as name:
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")
      with ops.colocate_with(self._resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_insert_v2(
            self._resource_handle, keys, values, name=name)
    return op

  def scatter_sub(self, keys, values, name=None):
    """Add table values by learning values.

    Args:
      keys: Keys to sub. Can be a tensor of any shape. Must match the
        table's key type.
      values: Values to be subtracted to table values. Must be a tensor of the
        same shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_scatter_sub" % self._name,
                        [self._resource_handle, keys, values]) as name:
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")
      with ops.colocate_with(self._resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_scatter_sub_v2(
            self._resource_handle, keys, values, name=name)
    return op

  def set_attr(self, keys, attr_key, attr_values, name=None):
    """Add the attributes for the keys in hash table.

    Args:
      keys: hash table keys
      attr_key: attribute's key
      attr_values: attributes values
      name: A name for the operation (optional)

    Returns:
       the created Operation
    """
    with ops.name_scope(name, "%s_lookup_table_set_attr" % self._name,
                        [self._resource_handle, keys,
                         attr_key, attr_values]) as name:
      keys_tensor = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      attr_key_tensor = ops.convert_to_tensor(attr_key, dtypes.string,
                                              name="attr_key")
      attr_values_tensor = ops.convert_to_tensor(attr_values, dtypes.float32,
                                                 name="attr_values")
      with ops.colocate_with(self._resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_set_attr(
          self._resource_handle, keys_tensor,
          attr_key_tensor, attr_values_tensor, name=name)
    return op

  def get_attr(self, keys, attr_key, default_attr_value, name=None):
    """ Get the attribute with name 'attr_key' of the keys in hash table

    Args:
      keys: keys of hash table
      attr_key: key of attribute
      default_attr_value: default value for the attribute if not found.
      name: A name for the Operation (optional)

    Return:
      A tensor contains the attribute values
    """
    with ops.name_scope(name, "%s_lookup_table_get_attr" % self._name,
                        [self._resource_handle, keys,
                         attr_key, default_attr_value]) as name:
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      attr_key = ops.convert_to_tensor(attr_key, dtypes.string, name="attr_key")
      attr_value = ops.convert_to_tensor(default_attr_value, dtypes.float32,
                                         name="default_attr_value")
      with ops.colocate_with(self._resource_handle):
        # pylint: disable=protected-access
        attribute_values = gen_lookup_ops.lookup_table_get_attr(
          self._resource_handle, keys, attr_key, attr_value, name=name)
    return attribute_values

  def import_from_file(self, filename, delimiter=" ", name=None):
    with ops.name_scope(name, "%s_lookup_table_import_from_file" % self._name,
                        [self._resource_handle, filename, delimiter]) as name:
      filename_tensor = ops.convert_to_tensor(
        filename, dtypes.string, name="filename")
      with ops.colocate_with(self._resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_import_from_file(
          self._resource_handle, filename_tensor, delimiter, name=name)
    return op

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self._name,
                        [self._resource_handle]) as name:
      with ops.colocate_with(self._resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
          self._resource_handle, self._key_dtype, self._value_dtype,
          name=name)
    return exported_keys, exported_values

  def export_values_and_attrs(self, name=None):
    """Returns tensors of all keys and attrs in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all attrs in the table.
      - keys of the table
      - values of corresponding keys
      - keys' updated timestamps
      - array of attribute keys
      - flags stand for existence of the attribute
      - attributes' values.
    """
    with ops.name_scope(name,
                        "%s_lookup_table_export_values_and_attrs" % self._name,
                        [self._resource_handle]) as name:
      with ops.colocate_with(self._resource_handle):
        exported_keys, exported_values, exported_update_ts, \
        exported_attr_keys, exported_attr_flags, exported_attr_values = \
          gen_lookup_ops.lookup_table_export_values_and_attrs_v2(
            self._resource_handle, self._key_dtype,
            self._value_dtype, name=name)
    return exported_keys, exported_values, exported_update_ts, \
           exported_attr_keys, exported_attr_flags, exported_attr_values

  def erase_by_threshold(self, threshold, name=None):
    """Erase elements whose attr less than threshold.

    Args:
      threshold: Threshold used for erase
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    """
    with ops.name_scope(name, "%s_lookup_table_erase_by_threshold" % self._name,
                        [self._resource_handle]) as name:
      with ops.colocate_with(self._resource_handle):
        op = gen_lookup_ops.lookup_table_erase_by_threshold_v2(
                        self._resource_handle, threshold, name=name)
    return op

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    return {"table": functools.partial(MutableHashTable._Saveable, table=self)}

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for MutableHashTable."""

    def __init__(self, table, name):
      tensors = table.export_values_and_attrs()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values"),
          BaseSaverBuilder.SaveSpec(tensors[2], "", name + "-update_ts"),
          BaseSaverBuilder.SaveSpec(tensors[3], "", name + "-attr_keys"),
          BaseSaverBuilder.SaveSpec(tensors[4], "", name + "-attr_flags"),
          BaseSaverBuilder.SaveSpec(tensors[5], "", name + "-attr_values")
      ]
      # pylint: disable=protected-access
      super(MutableHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.colocate_with(self.op._resource_handle):
        return gen_lookup_ops.lookup_table_import_values_and_attrs_v2(
          self.op._resource_handle, restored_tensors[0], restored_tensors[1],
          restored_tensors[2], restored_tensors[3], restored_tensors[4],
          restored_tensors[5])


ops.register_proto_function(ops.GraphKeys.MUTABLE_HASH_TABLE,
                            proto_type=variable_pb2.VariableDef,
                            to_proto=None,
                            from_proto=MutableHashTable.from_proto)


class MutableDenseHashTable(LookupInterface, checkpointable.CheckpointableBase):
  """A generic mutable hash table implementation using tensors as backing store.

  Data can be inserted by calling the insert method. It does not support
  initialization via the init method.

  It uses "open addressing" with quadratic reprobing to resolve collisions.
  Compared to `MutableHashTable` the insert and lookup operations in a
  `MutableDenseHashTable` are typically faster, but memory usage can be higher.
  However, `MutableDenseHashTable` does not require additional memory for
  temporary tensors created during checkpointing and restore operations.

  Example usage:

  ```python
  table = tf.contrib.lookup.MutableDenseHashTable(key_dtype=tf.int64,
                                                  value_dtype=tf.int64,
                                                  default_value=-1,
                                                  empty_key=0)
  sess.run(table.insert(keys, values))
  out = table.lookup(query_keys)
  print(out.eval())
  ```
  """

  # TODO(andreasst): consider extracting common code with MutableHashTable into
  # a common superclass.
  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               empty_key,
               initial_num_buckets=None,
               shared_name=None,
               name="MutableDenseHashTable",
               checkpoint=True):
    """Creates an empty `MutableDenseHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      empty_key: the key to use to represent empty buckets internally. Must not
        be used in insert or lookup operations.
      initial_num_buckets: the initial number of buckets.
      shared_name: If non-empty, this table will be shared under
        the given name across multiple sessions.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.

    Returns:
      A `MutableHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=value_dtype, name="default_value")
    self._value_shape = self._default_value.get_shape()

    # The table must be shared if checkpointing is requested for multi-worker
    # training to work correctly. Use the node name if no shared_name has been
    # explicitly specified.
    use_node_name_sharing = checkpoint and shared_name is None
    empty_key = ops.convert_to_tensor(
        empty_key, dtype=key_dtype, name="empty_key")
    executing_eagerly = context.executing_eagerly()
    if executing_eagerly and shared_name is None:
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      shared_name = "table_%d" % (ops.uid(),)
    self._table_ref = gen_lookup_ops.mutable_dense_hash_table_v2(
        empty_key=empty_key,
        shared_name=shared_name,
        use_node_name_sharing=use_node_name_sharing,
        value_dtype=value_dtype,
        value_shape=self._value_shape,
        initial_num_buckets=initial_num_buckets,
        name=name)
    if executing_eagerly:
      op_name = None
    else:
      op_name = self._table_ref.op.name.split("/")[-1]
    super(MutableDenseHashTable, self).__init__(
        key_dtype, value_dtype, op_name)

    if checkpoint:
      saveable = MutableDenseHashTable._Saveable(self, name)
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self._name,
                        [self._table_ref]) as name:
      with ops.colocate_with(self._table_ref):
        return gen_lookup_ops.lookup_table_size_v2(self._table_ref, name=name)

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(name, "%s_lookup_table_find" % self._name,
                        [self._table_ref, keys]) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self._table_ref):
        values = gen_lookup_ops.lookup_table_find_v2(
            self._table_ref, keys, self._default_value, name=name)

    return values

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the
        table's key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_insert" % self._name,
                        [self._table_ref, keys, values]) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      values = ops.convert_to_tensor(
          values, dtype=self._value_dtype, name="values")
      with ops.colocate_with(self._table_ref):
        op = gen_lookup_ops.lookup_table_insert_v2(
            self._table_ref, keys, values, name=name)
      return op

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self._name,
                        [self._table_ref]) as name:
      with ops.colocate_with(self._table_ref):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self._table_ref, self._key_dtype, self._value_dtype, name=name)

    return exported_keys, exported_values

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    return {"table": functools.partial(
        MutableDenseHashTable._Saveable, table=self)}

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for MutableDenseHashTable."""

    def __init__(self, table, name):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      # pylint: disable=protected-access
      super(MutableDenseHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.colocate_with(self.op._table_ref):
        return gen_lookup_ops.lookup_table_import_v2(
            self.op._table_ref, restored_tensors[0], restored_tensors[1])
