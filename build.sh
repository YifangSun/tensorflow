#!/bin/bash
bazel build --incompatible_remove_native_http_archive=false  --verbose_failures -c opt --define=with_seastar_support=true //tensorflow:libtensorflow_cc.so
bazel build --incompatible_remove_native_http_archive=false  --verbose_failures -c opt --define=with_seastar_support=true //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /home/work/temp/tf_dbg_pkg
