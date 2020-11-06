/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/lookup_table_op.h"
#define EIGEN_USE_THREADS

#include <deque>
#include <set>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tbb/concurrent_hash_map.h"

namespace tensorflow {
namespace lookup {

namespace {
static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
Status GetNumLinesInTextFile(Env* env, const string& vocab_file,
                             int64* num_lines) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(vocab_file, &file));

  io::InputBuffer input_buffer(file.get(), kInputBufferSize);
  string line;
  Status s = input_buffer.ReadLine(&line);
  int64 next_id = 0;
  while (s.ok()) {
    next_id++;
    s = input_buffer.ReadLine(&line);
  }
  if (!errors::IsOutOfRange(s)) {
    return s;
  }
  *num_lines = next_id;
  return Status::OK();
}
// Iterator that reads a hash file(key\tvalues(separated with space)).
// Each iteration process one line, it parses
// the line and populates the keys and values tensors used for initialization
// with a single key and corresponding value.
//
// What information of the line to populate the key or values is specified by
// providing key_index and value_index.
class HashFileLineIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  HashFileLineIterator()
      : valid_(false),
        status_(errors::FailedPrecondition("Not initialized")) {}

  // Initialize iterator.
  //
  // Prepares the file 'filename' and sets the data types to return the keys and
  // values tensors. It requires the indices of the tokens in the line given a
  // delimiter to specify where to pick the data from.
  Status Init(const string& filename, char delimiter, DataType key_dtype,
              DataType value_dtype, TensorShape value_shape,
              Env* env) {
    filename_ = filename;
    delimiter_ = delimiter;
    key_ = Tensor(key_dtype, TensorShape({}));
    value_ = Tensor(value_dtype, value_shape);
    value_shape_ = value_shape;
    env_ = env;

    status_ = env->NewRandomAccessFile(filename_, &file_);
    if (!status_.ok()) return status_;

    input_buffer_.reset(new io::InputBuffer(file_.get(), kInputBufferSize));
    valid_ = true;
    next_id_ = 0;
    Next();
    return status_;
  }

  void Next() override {
    if (!valid_) return;

    string line;
    status_ = input_buffer_->ReadLine(&line);
    if (!status_.ok()) {
      if(errors::IsOutOfRange(status_)) {
        VLOG(4) << "End with " << status_.ToString();
        status_ = Status::OK();
      }
      valid_ = false;
      return;
    }
    if (line.empty()) {
      status_ = errors::InvalidArgument("Invalid content in ", filename_,
                                        ": empty line found at position ",
                                        input_buffer_->Tell(), ".");
      valid_ = false;
      return;
    }

    std::vector<string> tokens = str_util::Split(line, "\t");
    if (tokens.size() != 2) {
      status_ = errors::InvalidArgument(
          "Invalid number of columns in ", filename_, " line ", next_id_,
          " (", line, ") : expected ", 2,
          " got ", tokens.size());
      valid_ = false;
      return;
    }
    string key_str = tokens[0];
    std::vector<string> values = str_util::Split(tokens.at(1), delimiter_);
    if (value_shape_.dim_size(0) != values.size()) {
      status_ = errors::InvalidArgument(
          "Invalid number of values in ", filename_, " line ", next_id_,
          " (", line, ") : expected ", value_shape_.dim_size(0),
          " got ", values.size());
      valid_ = false;
      return;
    }
    status_ = SetKey(key_str, &key_);
    if (!status_.ok()) {
      valid_ = false;
      return;
    }
    status_ = SetValue(values, &value_);
    if (!status_.ok()) {
      valid_ = false;
      return;
    }

    next_id_++;
  }

  bool Valid() const override { return valid_; }

  const Tensor& keys() const override { return key_; }

  const Tensor& values() const override { return value_; }

  Status status() const override { return status_; }

  int64 total_size() const override {
    int64 new_size = 0;
    Status status = GetNumLinesInTextFile(env_, filename_, &new_size);
    if (!status.ok()) {
      LOG(WARNING) << "Unable to get line count: " << status;
      new_size = -1;
    }
    *const_cast<int64*>(&total_size_) = new_size;
    return total_size_;
  }

 private:
  Tensor key_;
  Tensor value_;
  TensorShape value_shape_;
  int64 total_size_;
  bool valid_;  // true if the iterator points to an existing range.
  Env* env_;
  int64 next_id_;
  string filename_;
  char delimiter_;
  Status status_;
  std::unique_ptr<RandomAccessFile> file_;  // must outlive input_buffer_
  std::unique_ptr<io::InputBuffer> input_buffer_;

  Status SetKey(const std::string& token, Tensor* tensor) {
    const DataType& dtype = tensor->dtype();
    switch (dtype) {
      case DT_INT32: {
        int32 value;
        if (!strings::safe_strto32(token.c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", token, " in line ", next_id_,
                                         " is not a valid int32.");
        }
        tensor->flat<int32>()(0) = value;
      } break;
      case DT_INT64: {
        int64 value;
        if (!strings::safe_strto64(token.c_str(), &value)) {
          uint64 uv;
          if (!strings::safe_strtou64(token.c_str(), &uv)) {
            valid_ = false;
            return errors::InvalidArgument(
                "Field ", token, " in line ", next_id_,
                " is not a valid int64 or uint64.");
          }
          value = static_cast<int64>(uv);
        }
        tensor->flat<int64>()(0) = value;
      } break;
      default:
        valid_ = false;
        return errors::InvalidArgument("Data type ", DataTypeString(dtype),
                                       " not supported.");
    }
    return Status::OK();
  }
  Status SetValue(const std::vector<std::string>& tokens, Tensor* tensor) {
    const DataType& dtype = tensor->dtype();
    size_t token_size = tokens.size();
    if (dtype == DT_FLOAT) {
      auto value_flat = tensor->flat<float>();
      float value;
      for (size_t i = 0; i < token_size; ++i) {
        if (!strings::safe_strtof(tokens.at(i).c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", tokens.at(i),
                                         " in line ", next_id_,
                                         " is not a valid float.");
        }
        value_flat(i) = value;
      }
    } else if (dtype == DT_DOUBLE) {
      auto value_flat = tensor->flat<double>();
      double value;
      for (size_t i = 0; i < token_size; ++i) {
        if (!strings::safe_strtod(tokens.at(i).c_str(), &value)) {
          valid_ = false;
          return errors::InvalidArgument("Field ", tokens.at(i),
                                         " in line ", next_id_,
                                         " is not a valid double.");
        }
        value_flat(i) = value;
      }
    } else {
      valid_ = false;
      return errors::InvalidArgument("Data type ", DataTypeString(dtype),
                                     " not supported.");
    }
    return Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(HashFileLineIterator);
};

}
// Lookup table that wraps an unordered_map, where the key and value data type
// is specified. Each individual value must be a scalar. If vector values are
// required, use MutableHashTableOfTensors.
//
// This table is mutable and thread safe - Insert can be called at any time.
//
// Sample use case:
//
// MutableHashTableOfScalars<int64, int64> table;  // int64 -> int64.
// // Populate the table, elements could be added in one or multiple calls.
// table.Insert(key_tensor, value_tensor); // Populate the table.
//
// table.Find(in_t, &out_t, default_t)
//
template<typename T>
class TensorCache {
 public:
  void Initialize(OpKernelContext *ctx, const Tensor &default_value,
                  const std::string &distribution, const int64 seed,
                  const int64 seed2, const float &mean, const float &stddev,
                  const int32 &cache_size, const TensorShape &value_shape,
                  const DataType &dtype) {
    default_value_ = default_value;
    distribution_ = distribution;
    cache_size_ = cache_size;
    value_shape_ = value_shape;
    dtype_ = dtype;

    generator_ptr_.reset(new GuardedPhiloxRandom());
    generator_ptr_->Init(seed, seed2);
    auto cache_shape_ = TensorShape({cache_size_, value_shape_.dim_size(0)});
    mean_tensor_ = Tensor(dtype_, cache_shape_);
    stddev_tensor_ = Tensor(dtype_, cache_shape_);
    if (distribution_ == "truncated_normal" && dtype_ == DataType::DT_FLOAT) {
      mean_tensor_.flat<float>().setConstant(mean);
      stddev_tensor_.flat<float>().setConstant(stddev);
      AllocateTensorsByInitializerFP32(ctx);
    } else {
      AllocateTensorsByDefaultValue();
    }
  }

  Tensor GetTensor(OpKernelContext *ctx) {
    mutex_lock l(mu_);
    if (TF_PREDICT_FALSE(freed_tensors_.empty())) {
      if (distribution_ == "truncated_normal" && dtype_ == DataType::DT_FLOAT) {
        AllocateTensorsByInitializerFP32(ctx);
      } else {
        AllocateTensorsByDefaultValue();
      }
    }
    Tensor result = freed_tensors_.front();
    freed_tensors_.pop_front();
    return result;
  }

 private:
  void AllocateTensorsByDefaultValue() {
    for (int i = 0; i < cache_size_; ++i) {
      Tensor new_one(dtype_, value_shape_);
      new_one.flat<T>() = default_value_.flat<T>();
      freed_tensors_.push_back(new_one);
    }
  }

  void AllocateTensorsByInitializerFP32(OpKernelContext *ctx) {
    const size_t value_size = value_shape_.dim_size(0);
    for (int i = 0; i < cache_size_; ++i) {
      Tensor new_one(dtype_, value_shape_);
      auto new_one_flat = new_one.flat<float>();
      functor::FillPhiloxRandom<Eigen::ThreadPoolDevice,
                                TruncatedNormalDistribution>()(
          ctx, ctx->eigen_device<Eigen::ThreadPoolDevice>(),
          generator_ptr_->ReserveRandomOutputs(value_size, 256),
          new_one_flat.data(), value_size, TruncatedNormalDistribution());
      new_one_flat = new_one_flat * stddev_tensor_.flat<float>() +
          mean_tensor_.flat<float>();
      freed_tensors_.push_back(new_one);
    }
  }

  Tensor default_value_;
  std::string distribution_;
  Tensor mean_tensor_;
  Tensor stddev_tensor_;
  int32 cache_size_;
  TensorShape value_shape_;
  DataType dtype_;
  mutable mutex mu_;
  std::deque<Tensor> freed_tensors_;
  std::unique_ptr<GuardedPhiloxRandom> generator_ptr_;
  typedef random::TruncatedNormalDistribution<
      random::SingleSampleAdapter<random::PhiloxRandom>, float>
      TruncatedNormalDistribution;
};

namespace {
struct HashTableValue {
  Tensor value;
  uint32 update_ts_capacity;
  int32 offset;
  std::deque<uint64> timestamps;

  HashTableValue()
      : value(),
        update_ts_capacity(0),
        offset(0),
        timestamps() {}
  HashTableValue(const Tensor &value, const uint32 update_ts_capacity)
      : value(value),
        update_ts_capacity(update_ts_capacity),
        offset(update_ts_capacity - 1),
        timestamps(update_ts_capacity, 0) {
    CHECK_GE(update_ts_capacity, 1);
  }
  HashTableValue(const Tensor &value,
                 const uint32 update_ts_capacity,
                 uint64 cur_ts)
      : value(value),
        update_ts_capacity(update_ts_capacity),
        offset(0),
        timestamps(update_ts_capacity, cur_ts) {
    CHECK_GE(update_ts_capacity, 1);
  }
  HashTableValue(const Tensor &value, const uint32 update_ts_capacity,
                 std::deque<uint64> ts)
      : value(value),
        update_ts_capacity(update_ts_capacity),
        offset(0),
        timestamps(std::move(ts)) {
    CHECK_GE(update_ts_capacity, 1);
    while (offset < update_ts_capacity - 1 && timestamps[offset] == 0) {
      ++offset;
    }
  }
  HashTableValue(const HashTableValue &other)
      : value(other.value),
        update_ts_capacity(other.update_ts_capacity),
        offset(other.offset),
        timestamps(other.timestamps) {}
  HashTableValue(HashTableValue &&other)
      : value(std::move(other.value)),
        update_ts_capacity(other.update_ts_capacity),
        offset(other.offset),
        timestamps(std::move(other.timestamps)) {}

  HashTableValue &operator=(const HashTableValue &other) {
    if (&other != this) {
      value = other.value;
      update_ts_capacity = other.update_ts_capacity;
      offset = other.offset;
      timestamps = other.timestamps;
    }
    return *this;
  }
  HashTableValue &operator=(HashTableValue &&other) {
    if (&other != this) {
      value = std::move(other.value);
      update_ts_capacity = other.update_ts_capacity;
      offset = other.offset;
      timestamps = std::move(other.timestamps);
    }
    return *this;
  }

  void SetTS(const uint64 ts) {
    timestamps.pop_front();
    timestamps.push_back(ts);
    offset = offset > 0 ? offset - 1 : 0;
  }
  uint64 GetEarliestTS() const { return timestamps[offset]; }
  uint64 GetTS(const uint32 i) const {
    CHECK_LT(i, update_ts_capacity);
    return timestamps[i];
  }
};

bool FloatEqual(float v1, float v2) {
  static const float epsilon = 1.0e-9;
  return ((v2 - epsilon) <= v1 && v1 <= (v2 + epsilon));
}
}  // namespace

// Conventions
// 1. Every key in HashTable has same attributes.
template<class K, class V>
class TensorHashTable {
  typedef tbb::concurrent_hash_map<K, HashTableValue> CHashTable;
  typedef tbb::concurrent_hash_map<K,
  std::unordered_map<std::string, Tensor>> AttrHashTable;
 public:
  void Initialize(OpKernelContext *ctx, const Tensor &default_value,
                  const std::string &distribution, const int64 seed,
                  const int64 seed2, const float &mean, const float &stddev,
                  const int32 &tensor_cache_size, const TensorShape &vshape,
                  const DataType &key_dtype, const DataType &value_dtype,
                  const uint32 update_ts_capacity = 1) {
    default_value_ = default_value;
    value_shape_ = vshape;
    key_dtype_ = key_dtype;
    value_dtype_ = value_dtype;
    tensor_cache_size_ = tensor_cache_size;
    tensor_bytes_ = vshape.num_elements() * sizeof(V);
    attr_tensor_bytes_ = vshape.num_elements() * sizeof(float);
    update_ts_capacity_ = update_ts_capacity;
    table_.rehash(tensor_cache_size);
    cache_.Initialize(ctx, default_value, distribution, seed, seed2, mean,
                      stddev, tensor_cache_size, vshape, value_dtype);

    std::random_device rd;
    generator_ptr_.reset(new std::default_random_engine(rd()));
  }

  size_t Size() const {
    tf_shared_lock l(mu_);
    return table_.size();
  }

  void Clear() {
    {
      mutex_lock l(mu_);
      table_.clear();
    }
    {
      mutex_lock l(attr_mu_);
      table_attr_.clear();
    }
  }

  // Though there can be at most one occurrence of a given key in the map,
  // there may be other key-value pairs in flight with the same key.
  //  These arise from the semantics of the insert and erase methods.
  // The insert methods can create and destroy a temporary key-value pair
  // that is not inserted into a map. The erase methods remove a key-value pair
  // from the map before destroying it, thus permitting another thread to
  // construct a similar key before the old one is destroyed.
  // https://software.intel.com/en-us/node/506194
  void Find(OpKernelContext *ctx, const std::vector<K> &keys,
            const std::vector<int> &value_index,
            typename TTypes<V, 2>::Tensor *value, const float probability) {
    tf_shared_lock l(mu_);
    const uint64_t ts = Env::Default()->NowSeconds();
    for (int i = 0; i < keys.size(); ++i) {
      typename CHashTable::const_accessor found_acc;
      if (table_.find(found_acc, keys[i])) {
        const auto result_flat = found_acc->second.value.template flat<V>();
        value->template chip<0>(value_index[i]) = result_flat;
      } else {
        if (FloatEqual(probability, 1.0) ||
            std::generate_canonical<float, 10>(*generator_ptr_) <
                probability) {
          typename CHashTable::accessor acc;
          if (table_.insert(acc, keys[i])) {
            Tensor res = cache_.GetTensor(ctx);
            acc->second = HashTableValue(res, update_ts_capacity_, ts);
          }
          const auto result_flat = acc->second.value.template flat<V>();
          memcpy(&(*value)(value_index[i], 0), &result_flat(0), tensor_bytes_);
        } else {
          const auto result_flat = default_value_.flat<V>();
          memcpy(&(*value)(value_index[i], 0), &result_flat(0), tensor_bytes_);
        }
      }
    }
  }

  void FastFind(const std::vector<K> &keys, const std::vector<int> &value_index,
                typename TTypes<V, 2>::Tensor *value) {
    tf_shared_lock l(mu_);
    const size_t keys_size = keys.size();
    for (size_t i = 0; i < keys_size; ++i) {
      typename CHashTable::const_accessor found_acc;
      if (table_.find(found_acc, keys[i])) {
        const auto found_flat = found_acc->second.value.template flat<V>();
        memcpy(&(*value)(value_index[i], 0), &found_flat(0), tensor_bytes_);
      } else {
        memset(&(*value)(value_index[i], 0), 0, tensor_bytes_);
      }
    }
  }

  void ScatterSub(OpKernelContext *ctx, const std::vector<K> &keys,
                  const std::vector<int> &value_index,
                  const typename TTypes<V, 2>::ConstTensor &value) {
    tf_shared_lock l(mu_);
    const uint64_t ts = Env::Default()->NowSeconds();
    for (int i = 0; i < value_index.size(); ++i) {
      typename CHashTable::accessor found_acc;
      if (table_.find(found_acc, keys[i])) {
        auto flat_val = (found_acc->second.value).template flat<V>();
        flat_val = flat_val - value.template chip<0>(value_index[i]);
        found_acc->second.SetTS(ts);
      }
    }
  }

  void Insert(OpKernelContext *ctx, const std::vector<K> &keys,
              const std::vector<int> &value_index,
              const typename TTypes<V, 2>::ConstTensor *value) {
    tf_shared_lock l(mu_);
    const uint64_t ts = Env::Default()->NowSeconds();
    for (int i = 0; i < value_index.size(); ++i) {
      typename CHashTable::accessor acc;
      if (table_.insert(acc, keys[i])) {
        Tensor t = cache_.GetTensor(ctx);
        acc->second = HashTableValue(t, update_ts_capacity_, ts);
      }
      auto result_flat = acc->second.value.template flat<V>();
      memcpy(&result_flat(0), &(*value)(value_index[i], 0), tensor_bytes_);
    }
  }

  void SetAttr(const std::vector<K> &keys,
               const std::vector<int> &value_index,
               const std::string &attr_key,
               const typename TTypes<float, 2>::ConstTensor &value) {
    tf_shared_lock l(attr_mu_);
    const size_t keys_size = keys.size();
    for (size_t i = 0; i < keys_size; ++i) {
      typename AttrHashTable::accessor acc;
      table_attr_.insert(acc, keys[i]);
      Tensor t(DT_FLOAT, value_shape_);
      auto flat_val = t.flat<float>();
      memcpy(&(flat_val(0)), &(value(value_index[i], 0)), attr_tensor_bytes_);
      acc->second[attr_key] = std::move(t);
    }
  }

  void GetAttr(const std::vector<K> &keys,
               const std::vector<int> &value_index,
               const std::string &attr_key,
               const typename TTypes<float>::ConstFlat &default_attr_value,
               typename TTypes<float, 2>::Tensor *value) {
    tf_shared_lock l(attr_mu_);
    const size_t keys_size = keys.size();
    for (size_t i = 0; i < keys_size; ++i) {
      typename AttrHashTable::const_accessor found_acc;
      if (table_attr_.find(found_acc, keys[i])
          && (found_acc->second.count(attr_key) == 1)) {
        auto flat_val = found_acc->second.at(attr_key).template flat<float>();
        memcpy(&(*value)(value_index[i], 0), &(flat_val(0)),
               attr_tensor_bytes_);
      } else {
        memcpy(&(*value)(value_index[i], 0), &(default_attr_value(0)),
               attr_tensor_bytes_);
      }
    }
  }

  void ImportValuesAndAttrs(
      OpKernelContext *ctx, const std::vector<K> &keys,
      const std::vector<int> &value_index,
      const typename TTypes<V, 2>::ConstTensor *value,
      const typename TTypes<uint64, 2>::ConstTensor &update_ts,
      const std::vector<std::string> &attr_keys,
      const typename TTypes<int8, 2>::ConstTensor &attr_flags,
      const typename TTypes<float, 2>::ConstTensor &attr_values) {
    {
      tf_shared_lock l(mu_);
      for (int i = 0; i < value_index.size(); ++i) {
        typename CHashTable::accessor acc;
        if (table_.insert(acc, keys[i])) {
          Tensor t = cache_.GetTensor(ctx);
          acc->second = HashTableValue(
              t, update_ts_capacity_,
              std::deque<uint64>(&update_ts(value_index[i], 0),
                                 &update_ts(value_index[i], 0) +
                                     update_ts_capacity_));
        }
        auto result_flat = acc->second.value.template flat<V>();
        memcpy(&(result_flat(0)), &(*value)(value_index[i], 0),
               tensor_bytes_);
      }
    }
    {
      size_t attr_keys_len = attr_keys.size();
      tf_shared_lock l(attr_mu_);
      for (int i = 0; i < keys.size(); ++i) {
        typename AttrHashTable::accessor acc;
        if (table_attr_.insert(acc, keys[i])) {
          for (size_t j = 0; j < attr_keys_len; ++j) {
            if (attr_flags(value_index[i], j) == 1) {
              Tensor t(DT_FLOAT, value_shape_);
              auto flat_val = t.flat<float>();
              memcpy(&flat_val(0),
                     &(attr_values(value_index[i], j * attr_keys_len)),
                     attr_tensor_bytes_);
              acc->second.insert(std::make_pair(attr_keys[j], std::move(t)));
            }
          }
        }
      }
    }
  }

  void Remove(const std::vector<K> &keys) {
    {
      tf_shared_lock l(mu_);
      for (const auto &k : keys) {
        typename CHashTable::accessor acc;
        if (table_.find(acc, SubtleMustCopyIfIntegral(k))) {
          table_.erase(acc);
        }
      }
    }

    {
      tf_shared_lock l(attr_mu_);
      for (const auto &k : keys) {
        typename AttrHashTable::accessor acc;
        if (table_attr_.find(acc, SubtleMustCopyIfIntegral(k))) {
          table_attr_.erase(acc);
        }
      }
    }
  }

  int64 ExportValues(std::vector<Tensor> *keys, std::vector<Tensor> *values,
                     const size_t segment_id) {
    mutex_lock l(mu_);
    int64 table_size = table_.size();
    int64 value_dims = value_shape_.dim_size(0);
    (*keys)[segment_id] = Tensor(key_dtype_, TensorShape({table_size}));
    (*values)[segment_id] =
        Tensor(value_dtype_, TensorShape({table_size, value_dims}));
    auto keys_data = (*keys)[segment_id].flat<K>();
    auto values_data = (*values)[segment_id].flat_inner_dims<V, 2>();
    size_t i = 0;

    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      K key = it->first;
      Tensor value = it->second.value;
      keys_data(i) = key;
      auto value_flat = value.flat<V>();
      memcpy(&(values_data(i, 0)), &(value_flat(0)),
             tensor_bytes_);
    }
    return table_size;
  }

  int64 ExportValuesAndAttrs(std::vector<Tensor> *keys,
                             std::vector<Tensor> *values,
                             std::vector<Tensor> *update_ts,
                             const size_t segment_id) {
    mutex_lock l(mu_);
    int64 table_size = table_.size();
    int64 value_dims = value_shape_.dim_size(0);
    (*keys)[segment_id] = Tensor(key_dtype_, TensorShape({table_size}));
    (*values)[segment_id] =
        Tensor(value_dtype_, TensorShape({table_size, value_dims}));
    (*update_ts)[segment_id] =
        Tensor(DT_UINT64, TensorShape({table_size, update_ts_capacity_}));
    auto keys_data = (*keys)[segment_id].flat<K>();
    auto values_data = (*values)[segment_id].flat_inner_dims<V, 2>();
    auto update_ts_data = (*update_ts)[segment_id].matrix<uint64>();
    size_t i = 0;

    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      K key = it->first;
      Tensor value = it->second.value;

      keys_data(i) = key;
      auto value_flat = value.flat<V>();
      memcpy(&(values_data(i, 0)), &(value_flat(0)),
             tensor_bytes_);
      for (size_t j = 0; j < update_ts_capacity_; ++j) {
        update_ts_data(i, j) = it->second.timestamps[j];
      }
    }
    return table_size;
  }

  int64 ExportValuesAndAttrs(std::vector<Tensor> *keys,
                             std::vector<Tensor> *values,
                             std::vector<Tensor> *update_ts,
                             std::vector<Tensor> *attr_flags,
                             std::vector<Tensor> *attr_values,
                             const std::vector<std::string> &attr_keys,
                             const size_t segment_id) {
    mutex_lock l(mu_);
    mutex_lock attr_l(attr_mu_);
    int64 table_size = table_.size();
    int64 value_dims = value_shape_.dim_size(0);
    int64 attr_value_len = value_dims;
    int64 attr_key_len = attr_keys.size();
    (*keys)[segment_id] = Tensor(key_dtype_, TensorShape({table_size}));
    (*values)[segment_id] =
        Tensor(value_dtype_, TensorShape({table_size, value_dims}));
    (*update_ts)[segment_id] =
        Tensor(DT_UINT64, TensorShape({table_size, update_ts_capacity_}));
    (*attr_flags)[segment_id] = Tensor(
        DT_INT8, TensorShape({table_size, attr_key_len}));
    (*attr_values)[segment_id] =
        Tensor(DT_FLOAT, TensorShape({table_size, attr_key_len * value_dims}));
    auto keys_data = (*keys)[segment_id].flat<K>();
    auto values_data = (*values)[segment_id].flat_inner_dims<V, 2>();
    auto update_ts_data = (*update_ts)[segment_id].matrix<uint64>();
    auto attr_flags_data = (*attr_flags)[segment_id].flat_inner_dims<int8, 2>();
    auto attr_values_data =
        (*attr_values)[segment_id].flat_inner_dims<float, 2>();
    size_t i = 0;

    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      K key = it->first;
      Tensor value = it->second.value;

      keys_data(i) = key;
      auto value_flat = value.flat<V>();
      memcpy(&(values_data(i, 0)), &(value_flat(0)),
             tensor_bytes_);
      for (size_t j = 0; j < update_ts_capacity_; ++j) {
        update_ts_data(i, j) = it->second.timestamps[j];
      }
      for (int64 attr_idx = 0; attr_idx < attr_key_len; ++attr_idx) {
        typename AttrHashTable::const_accessor found_acc;
        if (table_attr_.find(found_acc, key)
            && found_acc->second.count(attr_keys[attr_idx]) == 1) {
          attr_flags_data(i, attr_idx) = 1;
          auto flat_val =
              found_acc->second.at(attr_keys[attr_idx]).template flat<float>();
          memcpy(&(attr_values_data(i, attr_idx * attr_value_len)),
                 &flat_val(0),
                 attr_tensor_bytes_);
        } else {
          attr_flags_data(i, attr_idx) = 0;
          memset(&(attr_values_data(i, attr_idx * attr_value_len)),
                 0,
                 attr_tensor_bytes_);
        }
      }
    }
    return table_size;
  }

  Status EraseByAttrThreshold(const uint64 &threshold) {
    std::set<K> keys_to_erase;
    {
      mutex_lock l(mu_);
      if (!table_.empty()) {
        for (auto it = table_.begin(); it != table_.end(); ++it) {
          if (it->second.GetEarliestTS() < threshold) {
            keys_to_erase.insert(it->first);
          }
        }

        for (auto k : keys_to_erase) {
          typename CHashTable::accessor acc;
          if (table_.find(acc, SubtleMustCopyIfIntegral(k))) {
            table_.erase(acc);
          }
        }
      }
    }
    {
      tf_shared_lock l(attr_mu_);
      if (!table_attr_.empty()) {
        for (auto k : keys_to_erase) {
          typename AttrHashTable::accessor acc;
          if (table_attr_.find(acc, SubtleMustCopyIfIntegral(k))) {
            table_attr_.erase(acc);
          }
        }
      }
    }

    return Status::OK();
  }

  int64 MemoryUsed() const {
    tf_shared_lock l(mu_);
    tf_shared_lock attr_l(attr_mu_);
    int64 value_mem = table_.size() * default_value_.TotalBytes();
    int64 attr_mem = table_attr_.size() * attr_tensor_bytes_;
    return sizeof(CHashTable) + value_mem + sizeof(AttrHashTable) + attr_mem;
  }

 private:
  Tensor default_value_;
  TensorShape value_shape_;
  DataType key_dtype_;
  DataType value_dtype_;
  int32 tensor_cache_size_;
  size_t tensor_bytes_;
  size_t attr_tensor_bytes_;
  uint32 update_ts_capacity_;
  mutable mutex mu_;
  mutable mutex attr_mu_;
  TensorCache<V> cache_;
  CHashTable table_ GUARDED_BY(mu_);
  AttrHashTable table_attr_ GUARDED_BY(attr_mu_);
  std::unique_ptr<std::default_random_engine> generator_ptr_;
};

// Lookup table that wraps an unordered_map. Behaves identical to
// MutableHashTableOfScalars except that each value must be a vector.
template<class K, class V>
class TrainableHashTableOfTensors : public LookupInterface {
 public:
  TrainableHashTableOfTensors(OpKernelContext *ctx, OpKernel *kernel) {
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "hash_table_segments",
                                    &hash_table_segments_));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "tensor_cache_size",
                                    &tensor_cache_size_));

    int64 seed, seed2;
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "seed", &seed));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "seed2", &seed2));

    std::string distribution;
    float mean;
    float stddev;
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "distribution", &distribution));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "mean", &mean));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "stddev", &stddev));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));

    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "update_ts_capacity",
                                    &update_ts_capacity_));

    tables_.resize(hash_table_segments_);
    for (int i = 0; i < hash_table_segments_; ++i) {
      tables_[i].Initialize(ctx, ctx->input(0), distribution, seed, seed2, mean,
                            stddev, tensor_cache_size_, value_shape_,
                            key_dtype(), value_dtype(), update_ts_capacity_);
    }
  }

  size_t size() const override {
    size_t result = 0;
    for (int i = 0; i < hash_table_segments_; ++i) {
      result += tables_[i].Size();
    }
    return result;
  }

  void ShardKeys(const typename TTypes<K>::ConstFlat &key_values,
                 std::vector<std::vector<K>> *sharded_keys,
                 std::vector<std::vector<int>> *sharded_indexes) {
    for (int i = 0; i < key_values.size(); ++i) {
      size_t segment_id = std::hash<K>{}(key_values(i)) % hash_table_segments_;
      (*sharded_keys)[segment_id].push_back(key_values(i));
      if (sharded_indexes) {
        (*sharded_indexes)[segment_id].push_back(i);
      }
    }
  }

  void ScheduleSegments(
      OpKernelContext *ctx,
      std::function<void(const size_t &segment_id)> per_segment_work) {
    int num_threads =
        ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
    num_threads =
        num_threads > hash_table_segments_ ? hash_table_segments_ : num_threads;
    thread::ThreadPool *thread_pool =
        ctx->device()->tensorflow_cpu_worker_threads()->workers;

    BlockingCounter counter(num_threads);
    auto per_thread_work = [&, this](int thread_id) {
      for (int i = thread_id; i < hash_table_segments_; i += num_threads) {
        per_segment_work(i);
      }
      counter.DecrementCount();
    };

    for (int i = 0; i < num_threads - 1; ++i) {
      thread_pool->Schedule([&, i]() { per_thread_work(i); });
    }
    per_thread_work(num_threads - 1);
    counter.Wait();
  }

  Status DoFind(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
                const Tensor &default_value, const float probability) {
    const auto key_values = keys.flat<K>();
    auto value_values = values->flat_inner_dims<V, 2>();

    std::vector<std::vector<K>> sharded_keys(hash_table_segments_);
    std::vector<std::vector<int>> sharded_indexes(hash_table_segments_);

    ShardKeys(key_values, &sharded_keys, &sharded_indexes);

    auto per_segment_work = [&, this](const size_t &segment_id) {
      if (sharded_keys[segment_id].empty()) return;
      tables_[segment_id].Find(ctx, sharded_keys[segment_id],
                               sharded_indexes[segment_id], &value_values,
                               probability);
    };
    ScheduleSegments(ctx, per_segment_work);

    return Status::OK();
  }

  Status Find(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
              const Tensor &default_value) override {
    return DoFind(ctx, keys, values, default_value, 0.0);
  }

  Status FindAndInsert(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
                       const Tensor &default_value,
                       const float probability) override {
    return DoFind(ctx, keys, values, default_value, probability);
  }

  Status FastFind(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
                  const Tensor &default_value) override {
    const auto key_values = keys.flat<K>();
    auto value_values = values->flat_inner_dims<V, 2>();

    std::vector<std::vector<K>> sharded_keys(hash_table_segments_);
    std::vector<std::vector<int>> sharded_indexes(hash_table_segments_);

    ShardKeys(key_values, &sharded_keys, &sharded_indexes);

    auto per_segment_work = [&, this](const size_t &segment_id) {
      if (sharded_keys[segment_id].empty()) return;
      tables_[segment_id].FastFind(sharded_keys[segment_id],
                                   sharded_indexes[segment_id], &value_values);
    };
    ScheduleSegments(ctx, per_segment_work);

    return Status::OK();
  }

  Status ScatterSub(OpKernelContext *ctx, const Tensor &keys,
                    const Tensor &values) override {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat_inner_dims<V, 2>();

    std::vector<std::vector<K>> sharded_keys(this->hash_table_segments_);
    std::vector<std::vector<int>> sharded_indexes(this->hash_table_segments_);

    this->ShardKeys(key_values, &sharded_keys, &sharded_indexes);

    auto per_segment_work = [&, this](size_t segment_id) {
      if (sharded_keys[segment_id].empty()) return;
      this->tables_[segment_id].ScatterSub(ctx, sharded_keys[segment_id],
                                           sharded_indexes[segment_id],
                                           value_values);
    };
    this->ScheduleSegments(ctx, per_segment_work);

    return Status::OK();
  }

  Status DoInsert(bool clear, OpKernelContext *ctx, const Tensor &keys,
                  const Tensor &values) {
    const auto key_values = keys.flat<K>();
    auto value_values = values.flat_inner_dims<V, 2>();

    std::vector<std::vector<K>> sharded_keys(hash_table_segments_);
    std::vector<std::vector<int>> sharded_indexes(hash_table_segments_);

    ShardKeys(key_values, &sharded_keys, &sharded_indexes);

    auto per_segment_work = [&, this](size_t segment_id) {
      if (clear) tables_[segment_id].Clear();
      if (sharded_keys[segment_id].empty()) return;
      tables_[segment_id].Insert(ctx, sharded_keys[segment_id],
                                 sharded_indexes[segment_id], &value_values);
    };
    ScheduleSegments(ctx, per_segment_work);
    return Status::OK();
  }

  Status Insert(OpKernelContext *ctx, const Tensor &keys,
                const Tensor &values) override {
    return DoInsert(false, ctx, keys, values);
  }

  Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
    const auto key_values = keys.flat<K>();

    std::vector<std::vector<K>> sharded_keys(hash_table_segments_);

    ShardKeys(key_values, &sharded_keys, nullptr);

    auto per_segment_work = [&, this](const size_t &segment_id) {
      if (sharded_keys[segment_id].empty()) return;
      tables_[segment_id].Remove(sharded_keys[segment_id]);
    };
    ScheduleSegments(ctx, per_segment_work);
    return Status::OK();
  }

  Status GetAttr(OpKernelContext *ctx, const Tensor &keys,
                 const std::string &attr_key,
                 const Tensor &default_attr_value,
                 Tensor *out_attr) {
    const auto key_values = keys.flat<K>();
    const auto default_attr_value_values = default_attr_value.flat<float>();
    auto attr_values = out_attr->flat_inner_dims<float, 2>();

    std::vector<std::vector<K>> sharded_keys(hash_table_segments_);
    std::vector<std::vector<int>> sharded_indexes(hash_table_segments_);

    ShardKeys(key_values, &sharded_keys, &sharded_indexes);

    auto per_segment_work = [&, this](const size_t &segment_id) {
      if (sharded_keys[segment_id].empty()) return;
      tables_[segment_id].GetAttr(sharded_keys[segment_id],
                                  sharded_indexes[segment_id],
                                  attr_key,
                                  default_attr_value_values,
                                  &attr_values);
    };
    ScheduleSegments(ctx, per_segment_work);
    return Status::OK();
  }

  Status SetAttr(OpKernelContext *ctx, const Tensor &keys,
                 const std::string &attr_key,
                 const Tensor &attr_values_tensor) {
    // record the attribute keys
    if (attr_keys_set_.count(attr_key) == 0) {
      attr_keys_set_.insert(attr_key);
      attr_keys_.push_back(attr_key);
    }
    const auto key_values = keys.flat<K>();
    const auto attr_values = attr_values_tensor.flat_inner_dims<float, 2>();

    std::vector<std::vector<K>> sharded_keys(hash_table_segments_);
    std::vector<std::vector<int>> sharded_indexes(hash_table_segments_);

    ShardKeys(key_values, &sharded_keys, &sharded_indexes);

    auto per_segment_work = [&, this](const size_t &segment_id) {
      if (sharded_keys[segment_id].empty()) return;
      tables_[segment_id].SetAttr(sharded_keys[segment_id],
                                  sharded_indexes[segment_id],
                                  attr_key,
                                  attr_values);
    };
    ScheduleSegments(ctx, per_segment_work);
    return Status::OK();
  }

  Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                      const Tensor &values) override {
    DoInsert(true, ctx, keys, values);
    return Status::OK();
  }

  Status ImportValuesAndAttrs(OpKernelContext *ctx,
                              const Tensor &keys,
                              const Tensor &values,
                              const Tensor &update_ts,
                              const Tensor &attr_keys,
                              const Tensor &attr_flags,
                              const Tensor &attr_values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat_inner_dims<V, 2>();
    const auto update_ts_values = update_ts.flat_inner_dims<uint64, 2>();
    const auto attr_keys_values = attr_keys.flat<string>();
    const auto attr_flags_values = attr_flags.flat_inner_dims<int8, 2>();
    const auto attr_values_values = attr_values.flat_inner_dims<float, 2>();

    int64 attr_key_len = attr_keys_values.size();
    for (int64 i = 0; i < attr_key_len; ++i) {
      // Exporting will save a empty placeholder.
      if (!attr_keys_values(i).empty()) {
        attr_keys_.push_back(attr_keys_values(i));
        attr_keys_set_.insert(attr_keys_values(i));
      }
    }
    std::vector<std::vector<K>> sharded_keys(hash_table_segments_);
    std::vector<std::vector<int>> sharded_indexes(hash_table_segments_);

    ShardKeys(key_values, &sharded_keys, &sharded_indexes);

    auto per_segment_work = [&, this](size_t segment_id) {
      tables_[segment_id].Clear();
      if (sharded_keys[segment_id].empty()) return;
      tables_[segment_id].ImportValuesAndAttrs(ctx, sharded_keys[segment_id],
                                               sharded_indexes[segment_id],
                                               &value_values,
                                               update_ts_values,
                                               attr_keys_,
                                               attr_flags_values,
                                               attr_values_values);
    };
    ScheduleSegments(ctx, per_segment_work);
    return Status::OK();
  }

  Status ImportFromFile(OpKernelContext *ctx, const std::string& filename,
                        char delimiter) {
    lookup::HashFileLineIterator iter;
    Status status = iter.Init(filename, delimiter, key_dtype(),
                              value_dtype(), value_shape_, ctx->env());
    if (status.code() == error::NOT_FOUND) {
      LOG(WARNING) << "Import file " << filename
                   << " not found for lookup table";
      return Status::OK();
    }
    if (!iter.Valid()) {
      return iter.status();
    }
    TF_RETURN_IF_ERROR(
        CheckKeyAndValueTensorsForInsert(iter.keys(), iter.values()));

    while (iter.Valid()) {
      TF_RETURN_IF_ERROR(DoInsert(false, ctx, iter.keys(), iter.values()));
      iter.Next();
    }
    if (!errors::IsOutOfRange(iter.status())) {
      return iter.status();
    }

    return Status::OK();
  }

  Status ExportValues(OpKernelContext *ctx) override {
    std::vector<Tensor> keys_segments;
    std::vector<Tensor> values_segments;
    std::vector<size_t> table_sizes;

    keys_segments.resize(hash_table_segments_);
    values_segments.resize(hash_table_segments_);
    table_sizes.resize(hash_table_segments_);

    int64 total_size = 0;
    for (size_t segment_id = 0; segment_id < hash_table_segments_;
         ++segment_id) {
      table_sizes[segment_id] = tables_[segment_id].ExportValues(
          &keys_segments, &values_segments, segment_id);
      total_size += table_sizes[segment_id];
    }

    Tensor *keys;
    Tensor *values;
    int64 value_dim = value_shape_.dim_size(0);
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({total_size}), &keys));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({total_size, value_dim}), &values));

    auto keys_data = keys->flat<K>();
    auto values_data = values->flat_inner_dims<V, 2>();
    int64 export_id = 0;
    size_t tensor_bytes = value_dim * sizeof(V);
    for (size_t segment_id = 0; segment_id < hash_table_segments_;
         ++segment_id) {
      auto keys_flat = keys_segments[segment_id].flat<K>();
      auto values_flat = values_segments[segment_id].flat_inner_dims<V, 2>();
      for (size_t i = 0; i < table_sizes[segment_id]; ++i, ++export_id) {
        keys_data(export_id) = keys_flat(i);
        memcpy(&(values_data(export_id, 0)), &(values_flat(i, 0)),
               tensor_bytes);
      }
    }

    return Status::OK();
  }

  Status ExportValuesAndAttrs(OpKernelContext *ctx) override {
    std::vector<Tensor> keys_segments;
    std::vector<Tensor> values_segments;
    std::vector<Tensor> update_ts_segments;
    std::vector<size_t> table_sizes;

    Tensor *keys;
    Tensor *values;
    Tensor *update_ts;
    Tensor *attr_keys;
    Tensor *attr_flags;
    Tensor *attr_values;
    int64 value_dim = value_shape_.dim_size(0);

    keys_segments.resize(hash_table_segments_);
    values_segments.resize(hash_table_segments_);
    update_ts_segments.resize(hash_table_segments_);
    table_sizes.resize(hash_table_segments_);

    int64 attr_key_len = attr_keys_.size();
    int64 total_size = 0;
    if (attr_key_len == 0) {
      for (size_t segment_id = 0; segment_id < hash_table_segments_;
           ++segment_id) {
        table_sizes[segment_id] = tables_[segment_id].ExportValuesAndAttrs(
            &keys_segments, &values_segments, &update_ts_segments, segment_id);
        total_size += table_sizes[segment_id];
      }

      TF_RETURN_IF_ERROR(
          ctx->allocate_output("keys", TensorShape({total_size}), &keys));
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "values", TensorShape({total_size, value_dim}), &values));
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "update_timestamp", TensorShape({total_size, update_ts_capacity_}),
          &update_ts));
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "attr_keys", TensorShape(), &attr_keys));
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "attr_flags", TensorShape(), &attr_flags));
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "attr_values", TensorShape(), &attr_values));

      auto keys_data = keys->flat<K>();
      auto values_data = values->flat_inner_dims<V, 2>();
      auto update_ts_data = update_ts->flat_inner_dims<uint64, 2>();
      int64 export_id = 0;
      size_t tensor_bytes = value_dim * sizeof(V);
      size_t update_ts_bytes = update_ts_capacity_ * sizeof(uint64);
      for (size_t segment_id = 0; segment_id < hash_table_segments_;
           ++segment_id) {
        auto keys_flat = keys_segments[segment_id].flat<K>();
        auto values_flat = values_segments[segment_id].flat_inner_dims<V, 2>();
        auto update_ts_flat =
            update_ts_segments[segment_id].flat_inner_dims<uint64, 2>();
        for (size_t i = 0; i < table_sizes[segment_id]; ++i, ++export_id) {
          keys_data(export_id) = keys_flat(i);
          memcpy(&(values_data(export_id, 0)), &(values_flat(i, 0)),
                 tensor_bytes);
          memcpy(&(update_ts_data(export_id, 0)), &(update_ts_flat(i, 0)),
                 update_ts_bytes);
        }
      }
    } else {
      std::vector<Tensor> attr_flags_segments;
      std::vector<Tensor> attr_values_segments;
      attr_flags_segments.resize(hash_table_segments_);
      attr_values_segments.resize(hash_table_segments_);
      table_sizes.resize(hash_table_segments_);

      for (size_t segment_id = 0; segment_id < hash_table_segments_;
           ++segment_id) {
        table_sizes[segment_id] = tables_[segment_id].ExportValuesAndAttrs(
            &keys_segments, &values_segments, &update_ts_segments,
            &attr_flags_segments, &attr_values_segments,
            attr_keys_, segment_id);
        total_size += table_sizes[segment_id];
      }
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("keys", TensorShape({total_size}), &keys));
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "values", TensorShape({total_size, value_dim}), &values));
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("update_timestamp",
                               TensorShape({total_size, update_ts_capacity_}),
                               &update_ts));
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("attr_keys",
                               TensorShape({attr_key_len}), &attr_keys));
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("attr_flags",
                               TensorShape({total_size, attr_key_len}),
                               &attr_flags));
      TF_RETURN_IF_ERROR(
          ctx->allocate_output(
              "attr_values",
              TensorShape({total_size,
                           attr_key_len * value_dim}),
              &attr_values));

      auto keys_data = keys->flat<K>();
      auto values_data = values->flat_inner_dims<V, 2>();
      auto update_ts_data = update_ts->flat_inner_dims<uint64, 2>();
      auto attr_flags_data = attr_flags->flat_inner_dims<int8, 2>();
      auto attr_values_data = attr_values->flat_inner_dims<float, 2>();
      int64 export_id = 0;
      size_t tensor_bytes = value_dim * sizeof(V);
      size_t update_ts_bytes = update_ts_capacity_ * sizeof(uint64);
      size_t attr_values_bytes = value_dim * attr_key_len * sizeof(float);
      for (size_t segment_id = 0; segment_id < hash_table_segments_;
           ++segment_id) {
        auto keys_flat = keys_segments[segment_id].flat<K>();
        auto values_flat = values_segments[segment_id].flat_inner_dims<V, 2>();
        auto update_ts_flat =
            update_ts_segments[segment_id].flat_inner_dims<uint64, 2>();
        auto attr_flags_flat =
            attr_flags_segments[segment_id].flat_inner_dims<int8, 2>();
        auto attr_values_flat =
            attr_values_segments[segment_id].flat_inner_dims<float, 2>();
        for (size_t i = 0; i < table_sizes[segment_id]; ++i, ++export_id) {
          keys_data(export_id) = keys_flat(i);
          memcpy(&(values_data(export_id, 0)), &(values_flat(i, 0)),
                 tensor_bytes);
          memcpy(&(update_ts_data(export_id, 0)), &(update_ts_flat(i, 0)),
                 update_ts_bytes);
          attr_flags_data.template chip<0>(export_id) =
              attr_flags_flat.template chip<0>(i);
          memcpy(&(attr_values_data(export_id, 0)), &(attr_values_flat(i, 0)),
                 attr_values_bytes);
        }
      }
      auto attr_keys_flat = attr_keys->flat<string>();
      for (int64 i = 0; i < attr_key_len; ++i) {
        attr_keys_flat(i) = attr_keys_[i];
      }
    }

    return Status::OK();
  }

  Status EraseByAttrThreshold(const uint64 &threshold) {
    for (size_t segment_id = 0; segment_id < hash_table_segments_;
         ++segment_id) {
      const Status err_code =
          tables_[segment_id].EraseByAttrThreshold(threshold);
      if (err_code != Status::OK()) {
        return err_code;
      }
    }
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return value_shape_; }

  int64 MemoryUsed() const override {
    int64 ret = 0;
    for (size_t segment_id = 0; segment_id < hash_table_segments_;
         ++segment_id) {
      ret += tables_[segment_id].MemoryUsed();
    }
    return sizeof(TrainableHashTableOfTensors) + ret;
  }

 protected:
  int32 hash_table_segments_;
  int32 tensor_cache_size_;
  TensorShape value_shape_;
  std::vector<std::string> attr_keys_;
  std::set<std::string> attr_keys_set_;
  std::vector<TensorHashTable<K, V>> tables_;
  int32 update_ts_capacity_;
};

}  // namespace lookup

// Table lookup and insert op.
// Perform the lookup operation on the given table,
// insert it with probability if not exist.
class LookupTableFindAndInsertOp : public OpKernel {
 public:
  explicit LookupTableFindAndInsertOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype(), DT_FLOAT};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);
    const Tensor &probability = ctx->input(3);
    OP_REQUIRES_OK(ctx, table->CheckFindArguments(key, default_value));

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, table->FindAndInsert(ctx, key, out, default_value,
                                             probability.scalar<float>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableFindAndInsert").Device(DEVICE_CPU),
                        LookupTableFindAndInsertOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableFindAndInsertV2").Device(DEVICE_CPU),
                        LookupTableFindAndInsertOp);

// Table lookup op. Perform the lookup operation on the given table.
// The ones not found won't inserted.
class LookupTableFastFindOp : public OpKernel {
 public:
  explicit LookupTableFastFindOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckFindArguments(key, default_value));

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, table->FastFind(ctx, key, out, default_value));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableFastFind").Device(DEVICE_CPU),
                        LookupTableFastFindOp);
REGISTER_KERNEL_BUILDER(Name("LookupTableFastFindV2").Device(DEVICE_CPU),
                        LookupTableFastFindOp);

class LookupTableScatterSubOp : public OpKernel {
 public:
  explicit LookupTableScatterSubOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, lookup::GetLookupTable("table_handle", ctx, &table));

    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ScatterSub(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
          memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableScatterSubV2").Device(DEVICE_CPU),
                        LookupTableScatterSubOp);

// Op that get specified attribute of the keys in table.
class LookupTableGetAttrOp : public OpKernel {
 public:
  explicit LookupTableGetAttrOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      DT_STRING, DT_FLOAT};
    DataTypeVector expected_outputs = {DT_FLOAT};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor &key_tensor = ctx->input(1);
    const Tensor &attr_key_tensor = ctx->input(2);
    const Tensor &default_attr_value_tensor = ctx->input(3);

    TensorShape output_shape = key_tensor.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("attrs", output_shape, &out));

    const string &attr_key = attr_key_tensor.scalar<string>()();
    OP_REQUIRES_OK(ctx, table->GetAttr(ctx,
                                       key_tensor,
                                       attr_key,
                                       default_attr_value_tensor,
                                       out));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableGetAttr").Device(DEVICE_CPU),
                        LookupTableGetAttrOp);

// Table insert op.
class LookupTableSetAttrOp : public OpKernel {
 public:
  explicit LookupTableSetAttrOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      DT_STRING, DT_FLOAT};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &key_tensor = ctx->input(1);
    const Tensor &attr_key_tensor = ctx->input(2);
    const Tensor &attr_values_tensor = ctx->input(3);

    const string &attr_key = attr_key_tensor.scalar<string>()();
    OP_REQUIRES_OK(ctx, table->SetAttr(ctx, key_tensor, attr_key,
                                       attr_values_tensor));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableSetAttr").Device(DEVICE_CPU),
                        LookupTableSetAttrOp);

template<class K, class V>
class LookupTableBatchGradOp : public OpKernel {
 public:
  explicit LookupTableBatchGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor &indices = ctx->input(0);
    const Tensor &grads = ctx->input(1);
    const Tensor &batch = ctx->input(2);
    const Tensor &ids = ctx->input(3);
    const Tensor &counts = ctx->input(4);

    const int64 indices_num = indices.NumElements();
    const int64 grads_num = grads.dim_size(0);
    const int32 batch_size = batch.scalar<int32>()();
    const int64 ids_num = ids.NumElements();
    const int64 counts_num = counts.NumElements();
    OP_REQUIRES(ctx, indices_num == grads_num,
                errors::InvalidArgument(
                    "indices should be the same size as dimension 0 of grads"));
    OP_REQUIRES(
        ctx, ids_num == counts_num,
        errors::InvalidArgument("ids should be the same size as counts. got: ",
                                ids_num, ",", counts_num, ",", indices_num));

    TensorShape output_shape = grads.shape();
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    const auto indices_values = indices.flat<K>();
    const auto grads_values = grads.flat_inner_dims<V, 2>();
    const auto ids_values = ids.flat<K>();
    const auto counts_values = counts.flat<K>();

    std::unordered_map<K, int64> id_index_map(ids_num);
    for (int64 i = 0; i < ids_num; ++i) {
      id_index_map.insert({ids_values(i), i});
    }

    auto out_values = out->flat_inner_dims<V, 2>();
    const int64 grad_len = grads.dim_size(1);
    for (int64 i = 0; i < indices_num; ++i) {
      const K grad_count = counts_values(id_index_map[indices_values(i)]);
      for (int64 j = 0; j < grad_len; ++j) {
        out_values(i, j) = grads_values(i, j) * batch_size / grad_count;
      }
    }
  }
};

#define REGISTER_KERNEL(indice_dtype, grad_dtype)                           \
  REGISTER_KERNEL_BUILDER(Name("LookupTableBatchGradV2")                    \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<indice_dtype>("indice_dtype") \
                              .TypeConstraint<grad_dtype>("grad_dtype"),    \
                          LookupTableBatchGradOp<indice_dtype, grad_dtype>);

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);

#undef REGISTER_KERNEL

// Op that outputs tensors of all keys and all attrs.
class LookupTableExportValuesAndAttrsOp : public OpKernel {
 public:
  explicit LookupTableExportValuesAndAttrsOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValuesAndAttrs(ctx));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("LookupTableExportValuesAndAttrs").Device(DEVICE_CPU),
    LookupTableExportValuesAndAttrsOp);

REGISTER_KERNEL_BUILDER(
    Name("LookupTableExportValuesAndAttrsV2").Device(DEVICE_CPU),
    LookupTableExportValuesAndAttrsOp);

// Op that erase elements with attr value less than threshold.
class LookupTableEraseByThresholdOp : public OpKernel {
 public:
  explicit LookupTableEraseByThresholdOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, DT_UINT64};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &threshold = ctx->input(1);
    OP_REQUIRES_OK(ctx,
                   table->EraseByAttrThreshold(threshold.scalar<uint64>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableEraseByThreshold").Device(DEVICE_CPU),
                        LookupTableEraseByThresholdOp);
REGISTER_KERNEL_BUILDER(
    Name("LookupTableEraseByThresholdV2").Device(DEVICE_CPU),
    LookupTableEraseByThresholdOp);

class LookupTableImportValuesAndAttrsOp : public OpKernel {
 public:
  explicit LookupTableImportValuesAndAttrsOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype(), DT_UINT64,
                                      DT_STRING, DT_INT8, DT_FLOAT};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));
    const Tensor &update_ts = ctx->input(3);
    CHECK_EQ(update_ts.dim_size(0), keys.dim_size(0));
    const Tensor &attr_keys = ctx->input(4);
    const Tensor &attr_flags = ctx->input(5);
    const Tensor &attr_values = ctx->input(6);

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ImportValuesAndAttrs(
        ctx, keys, values, update_ts, attr_keys, attr_flags, attr_values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
          memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("LookupTableImportValuesAndAttrs").Device(DEVICE_CPU),
    LookupTableImportValuesAndAttrsOp);
REGISTER_KERNEL_BUILDER(
    Name("LookupTableImportValuesAndAttrsV2").Device(DEVICE_CPU),
    LookupTableImportValuesAndAttrsOp);

class LookupTableImportFromFileOp: public OpKernel {
 public:
  explicit LookupTableImportFromFileOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
    string delimiter;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delimiter", &delimiter));
    OP_REQUIRES(ctx, delimiter.size() == 1,
                errors::InvalidArgument("delimiter should be only 1 char"));
    delimiter_ = delimiter[0];
  }

  void Compute(OpKernelContext *ctx) override {
    lookup::LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0,
                                      DT_STRING};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& filename_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(filename_tensor.shape()),
        errors::InvalidArgument("filename should be a single string, but got ",
                                filename_tensor.shape().DebugString()));

    string filename = filename_tensor.scalar<string>()();
    OP_REQUIRES(ctx, !filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    OP_REQUIRES_OK(ctx,
                   table->ImportFromFile(ctx, filename, delimiter_));
  }
 private:
  char delimiter_;
};

REGISTER_KERNEL_BUILDER(
    Name("LookupTableImportFromFile").Device(DEVICE_CPU),
    LookupTableImportFromFileOp);

// Register the MutableHashTableOfTensors op for trainable value.
#define REGISTER_TRAINABLE_KERNEL(key_dtype, value_dtype)              \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MutableHashTableOfTensors")                                \
          .Device(DEVICE_CPU)                                          \
          .TypeConstraint<key_dtype>("key_dtype")                      \
          .TypeConstraint<value_dtype>("value_dtype"),                 \
      LookupTableOp<                                                   \
          lookup::TrainableHashTableOfTensors<key_dtype, value_dtype>, \
          key_dtype, value_dtype>)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MutableHashTableOfTensorsV2")                              \
          .Device(DEVICE_CPU)                                          \
          .TypeConstraint<key_dtype>("key_dtype")                      \
          .TypeConstraint<value_dtype>("value_dtype"),                 \
      LookupTableOp<                                                   \
          lookup::TrainableHashTableOfTensors<key_dtype, value_dtype>, \
          key_dtype, value_dtype>)

REGISTER_TRAINABLE_KERNEL(int32, double);
REGISTER_TRAINABLE_KERNEL(int32, float);
REGISTER_TRAINABLE_KERNEL(int32, int32);
REGISTER_TRAINABLE_KERNEL(int64, double);
REGISTER_TRAINABLE_KERNEL(int64, float);
REGISTER_TRAINABLE_KERNEL(int64, int32);
REGISTER_TRAINABLE_KERNEL(int64, int64);
REGISTER_TRAINABLE_KERNEL(string, double);
REGISTER_TRAINABLE_KERNEL(string, float);
REGISTER_TRAINABLE_KERNEL(string, int32);
REGISTER_TRAINABLE_KERNEL(string, int64);

#undef REGISTER_TRAINABLE_KERNEL

}  // namespace tensorflow
