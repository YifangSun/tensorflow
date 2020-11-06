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
"""Benchmark for Matmul operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.framework.test_util import TensorFlowTestCase

from tensorflow.contrib import lookup
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.framework import dtypes, constant_op
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class MutableHashTableBenchmark(test.Benchmark):

  def _create_table(self, embed_size, num_segment):
    default_val = constant_op.constant([-1]*embed_size, dtypes.float32)
    return lookup.MutableHashTable(
      dtypes.int64, dtypes.float32,
      default_val, hash_table_segments=num_segment,
      tensor_cache_size=20000,
      name="t1", checkpoint=True)

  def benchmark_single_repeated_insert(self):
    embed_size = 30
    table = self._create_table(embed_size, 7)
    values = constant_op.constant([1] * embed_size, dtypes.float32)
    insert = table.insert(0, values)
    size = table.size()
    with session.Session() as sess:
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) == 1

  def benchmark_many_repeated_insert(self):
    embed_size = 20
    table = self._create_table(embed_size, 7)
    c = counter.Counter().make_one_shot_iterator().get_next()
    values = constant_op.constant([1] * embed_size, dtypes.float32)
    insert = table.insert(c, values)
    size = table.size()
    with session.Session() as sess:
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) >= 1000

  def benchmark_many_repeated_batch_32_insert(self):
    embed_size = 20
    batch_size = 1024
    table = self._create_table(embed_size, 7)
    c = counter.Counter().make_one_shot_iterator().get_next()
    values = constant_op.constant([[1] * embed_size for _ in range(batch_size)],
                                  dtypes.float32)
    insert = table.insert(batch_size * c + list(range(batch_size)), values)
    size = table.size()
    with session.Session() as sess:
      self.run_op_benchmark(sess, insert,
                            burn_iters=10, min_iters=10000)
      assert sess.run(size) >= 10000*batch_size

  def checkedThread(self, target, args=None, kwargs=None):
    ret = TensorFlowTestCase._CheckedThread(self, target, args, kwargs)
    return ret

  def benchmark_parallel_find_insert_export(self):
    batch_size = 1024
    embed_size = 20
    table = self._create_table(embed_size, 8)
    c = counter.Counter().make_one_shot_iterator().get_next()
    c_grad = counter.Counter().make_one_shot_iterator().get_next()
    grads = constant_op.constant([[.1] * embed_size for _ in range(batch_size)],
                                  dtypes.float32)

    lookup_insert_op = table.lookup_and_insert(
      batch_size * c + list(range(batch_size)), probability=1.0)
    scatter_sub_op = table.scatter_sub(
      batch_size * c_grad + list(range(batch_size)),
      grads)
    export_op = table.export_values_and_attrs()

    def lookup_insert(bench, sess):
      bench(sess, lookup_insert_op,
            burn_iters=10, min_iters=10000)

    def scatter_sub(bench, sess):
      bench(sess, scatter_sub_op,
            burn_iters=10, min_iters=5000)

    def export_func(bench, sess):
      bench(sess, export_op,
            burn_iters=0, min_iters=1)

    num_iter = 3
    with session.Session() as sess:
      insert_threads = self.checkedThread(
          target=lookup_insert, args=(self.run_op_benchmark, sess, ))
      scatter_sub_threads = self.checkedThread(
          target=scatter_sub, args=(self.run_op_benchmark, sess,))
      export_threads = [self.checkedThread(
          target=export_func, args=(self.run_op_benchmark, sess,))
        for _ in range(num_iter)]

      insert_threads.start()
      time.sleep(10)
      scatter_sub_threads.start()
      time.sleep(1)
      for i in range(num_iter):
        export_threads[i].start()
        time.sleep(5)
      insert_threads.join()
      scatter_sub_threads.join()
      for t in export_threads:
        t.join()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  test.main()
