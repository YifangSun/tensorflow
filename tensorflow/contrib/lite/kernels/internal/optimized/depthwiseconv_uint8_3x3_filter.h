/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_

#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Enable for arm64 except for the Nvidia Linux 4 Tegra (L4T) running on
// Jetson TX-2. This compiler does not support the offsetof() macro.
#if defined(__aarch64__) && !defined(GOOGLE_L4T)
#include <stddef.h>
// clang-format gets confused with this file and ends up formatting lines to
// be larger than 80 characters. Turn off here and back on at the end of the
// file.

// clang-format off

#define DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE 10 * 10 * 64

// Encapsulates constant parameters used in DepthwiseConv.
// 64-bit is used for types that will be added to 64-bit addresses in asm.
struct DepthwiseConvParams {
  int64_t input_depth;
  int64_t input_row_size;
  int64_t output_depth;
  int64_t output_row_size;
  int64_t filter_row_size;
  int32 input_offset;
  int32 output_offset;
  int32 filter_offset;
  int32 output_multiplier;
  int32 output_activation_min;
  int32 output_activation_max;
  int32 output_right_shift;
  int32 input_width;
  int32 input_height;
  int32 stride_width;
  int32 stride_height;
  int32 output_width;
  int32 output_height;
};

#define STR(s) STR_UNEXPANDED(s)
#define STR_UNEXPANDED(s) #s

// Represents the number of bytes offset from the start of the
// DepthwiseConvParams struct. This is used in the asm to load parameters.
// Keep these values in sync with the static_asserts below.
#define OFFSET_INPUT_DEPTH 0
#define OFFSET_INPUT_ROW_SIZE 8
#define OFFSET_OUTPUT_DEPTH 16
#define OFFSET_OUTPUT_ROW_SIZE 24
#define OFFSET_FILTER_ROW_SIZE 32
#define OFFSET_INPUT_OFFSET 40
#define OFFSET_OUTPUT_OFFSET 44
#define OFFSET_FILTER_OFFSET 48
#define OFFSET_OUTPUT_MULTIPLIER 52
#define OFFSET_OUTPUT_ACTIVATION_MIN 56
#define OFFSET_OUTPUT_ACTIVATION_MAX 60
#define OFFSET_OUTPUT_RIGHT_SHIFT 64
#define OFFSET_INPUT_WIDTH 68
#define OFFSET_INPUT_HEIGHT 72
#define OFFSET_STRIDE_WIDTH 76
#define OFFSET_STRIDE_HEIGHT 80
#define OFFSET_OUTPUT_WIDTH 84
#define OFFSET_OUTPUT_HEIGHT 88

static_assert(offsetof(DepthwiseConvParams, input_depth) ==
                  OFFSET_INPUT_DEPTH, "");
static_assert(offsetof(DepthwiseConvParams, input_row_size) ==
                  OFFSET_INPUT_ROW_SIZE, "");
static_assert(offsetof(DepthwiseConvParams, output_depth) ==
                  OFFSET_OUTPUT_DEPTH, "");
static_assert(offsetof(DepthwiseConvParams, output_row_size) ==
                  OFFSET_OUTPUT_ROW_SIZE, "");
static_assert(offsetof(DepthwiseConvParams, filter_row_size) ==
                  OFFSET_FILTER_ROW_SIZE, "");
static_assert(offsetof(DepthwiseConvParams, input_offset) ==
                  OFFSET_INPUT_OFFSET, "");
static_assert(offsetof(DepthwiseConvParams, output_offset) ==
                  OFFSET_OUTPUT_OFFSET, "");
static_assert(offsetof(DepthwiseConvParams, filter_offset) ==
                  OFFSET_FILTER_OFFSET, "");
static_assert(offsetof(DepthwiseConvParams, output_multiplier) ==
                  OFFSET_OUTPUT_MULTIPLIER, "");
static_assert(offsetof(DepthwiseConvParams, output_activation_min) ==
                  OFFSET_OUTPUT_ACTIVATION_MIN, "");
static_assert(offsetof(DepthwiseConvParams, output_activation_max) ==
                  OFFSET_OUTPUT_ACTIVATION_MAX, "");
static_assert(offsetof(DepthwiseConvParams, output_right_shift) ==
                  OFFSET_OUTPUT_RIGHT_SHIFT, "");
static_assert(offsetof(DepthwiseConvParams, input_width) ==
                  OFFSET_INPUT_WIDTH, "");
static_assert(offsetof(DepthwiseConvParams, input_height) ==
                  OFFSET_INPUT_HEIGHT, "");
static_assert(offsetof(DepthwiseConvParams, stride_width) ==
                  OFFSET_STRIDE_WIDTH, "");
static_assert(offsetof(DepthwiseConvParams, stride_height) ==
                  OFFSET_STRIDE_HEIGHT, "");
static_assert(offsetof(DepthwiseConvParams, output_width) ==
                  OFFSET_OUTPUT_WIDTH, "");
static_assert(offsetof(DepthwiseConvParams, output_height) ==
                  OFFSET_OUTPUT_HEIGHT, "");

template <int32 kDepth, int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvWindow {};

template <>
struct DepthwiseConvWindow<8, 1, 1> {
 public:
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                  const int32* bias_ptr, uint8* output_ptr, int64_t input_depth,
                  int64_t input_row_size, int32 output_window_height,
                  int32 output_window_width,
                  const DepthwiseConvParams* params_ptr) {
    const int64_t input_width_increment = 2 * input_depth;
    const int64_t input_height_increment = 2 * input_row_size;
    const int64_t output_height_increment = 2 * params_ptr->output_row_size;

#define DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "1"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "2"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "3"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "4"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "5"
#define DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "6"
#define DEPTHWISECONV_LABEL_HEIGHT_1 "7"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "8"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "9"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "10"
#define DEPTHWISECONV_LABEL_HEIGHT_1_END "11"

    asm volatile(
        // Performs depthwise convolutions for a window specified by
        // |output_window_height| and |output_window_width|. The inner-most loop
        // processes 2x2 outputs, and any leftovers at the end.
        //
        // Algorithm works as follows:
        //
        //   1. Load filters of 8 depth (8x3x3). Registers v0--v8 hold filter
        //      values.
        //   2. For 2 output heights at a time:
        //        i.  For 2 output widths at a time, load inputs for a 2x1 (2
        //            height, 1 width) output window (4x3 input window).
        //            Registers v9--v20 hold input values. Mul-add with
        //            accumulators v21--v24. Then run activation, downquantize
        //            and store. Repeat for the next 2x1 output window,
        //            leveraging overlapping inputs.
        //        ii. Handle single leftover width if exists.
        //   3. Handle single leftover height if exists.
        //        i.  For 2 output widths at a time, load inputs for a 1x2 (1
        //            height, 2 width) output window (3x4 input window).
        //            Registers v9--v20 hold input values. Mul-add with
        //            accumulators v21--v24. Then run activation, downquantize
        //            and store. Repeat for the next 1x2 output window,
        //            leveraging overlapping inputs.
        //        ii. Handle single leftover width if exists.
        //
        // Loads are placed as soon as the register is no longer needed and
        // interleaved with arithmetic operations to take advantage of
        // dual-issue pipelines. We also add input offsets as far from the loads
        // as possible to give loads enough cycles to fetch data from memory.

        // Set "constant" registers. These registers may be replaced with temp
        // values from time to time when there are not enough NEON registers.
        // We use x9--x15 general purpose registers as they are caller-saved
        // temporary registers (see http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf).  // NOLINT
        "ldr w9, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr x3, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "cmp %w[output_window_height], #2\n"
        "dup v26.8h, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "ldr w2, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v29.4s, w2\n"
        "ldr w4, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "dup v30.4s, w4\n"
        "ldr w0, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v31.4s, w0\n"
        "neg w9, w9\n"
        "dup v28.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "add x10, %[bias_ptr], #16\n"
        "ldr x1, [%[params_ptr], #" STR(OFFSET_OUTPUT_ROW_SIZE) "]\n"
        "dup v9.8h, w9\n"

        // Load filters and add offsets.
        "ld1 {v0.8b}, [%[filter_ptr]], x3\n"
        "ld1 {v1.8b}, [%[filter_ptr]], x3\n"
        "uaddw v0.8h, v9.8h, v0.8b\n"
        "ld1 {v2.8b}, [%[filter_ptr]], x3\n"
        "uaddw v1.8h, v9.8h, v1.8b\n"
        "ld1 {v3.8b}, [%[filter_ptr]], x3\n"
        "uaddw v2.8h, v9.8h, v2.8b\n"
        "ld1 {v4.8b}, [%[filter_ptr]], x3\n"
        "uaddw v3.8h, v9.8h, v3.8b\n"
        "ld1 {v5.8b}, [%[filter_ptr]], x3\n"
        "uaddw v4.8h, v9.8h, v4.8b\n"
        "ld1 {v6.8b}, [%[filter_ptr]], x3\n"
        "uaddw v5.8h, v9.8h, v5.8b\n"
        "ld1 {v7.8b}, [%[filter_ptr]], x3\n"
        "uaddw v6.8h, v9.8h, v6.8b\n"
        "ld1 {v8.8b}, [%[filter_ptr]], x3\n"
        "uaddw v7.8h, v9.8h, v7.8b\n"
        "uaddw v8.8h, v9.8h, v8.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_2_LOOP ":\n"
          // This loop processes 2x2 outputs. To avoid register exhaustion,
          // inputs for the left 2 outputs are loaded first, then the right
          // two outputs.
          "mov x11, %[input_ptr]\n"
          "mov x12, x11\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "add x13, x11, %[input_row_size]\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "add x14, x13, %[input_row_size]\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "add x15, x14, %[input_row_size]\n"
          "ld1 {v12.8b}, [x13], %[input_depth]\n"
          "mov w5, %w[output_window_width]\n"
          "ld1 {v13.8b}, [x13], %[input_depth]\n"
          "mov x6, %[output_ptr]\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "add x7, %[output_ptr], x1\n"
          "ld1 {v15.8b}, [x14], %[input_depth]\n"
          // The height 2 / width 2 loop loads an extra 2x1 outputs (2 height,
          // 1 width) in anticipation for the next iteration. Make sure
          // |output_window_width| is large enough to handle the additional
          // loads, otherwise jump to specific the appropriate label to handle
          // smaller widths.
          "cmp w5, #2\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "ld1 {v16.8b}, [x14], %[input_depth]\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "ld1 {v17.8b}, [x14], %[input_depth]\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "ld1 {v18.8b}, [x15], %[input_depth]\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "ld1 {v19.8b}, [x15], %[input_depth]\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"
          "ld1 {v20.8b}, [x15], %[input_depth]\n"
          "uaddw v14.8h, v26.8h, v14.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "ld1 {v22.4s}, [x10]\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v17.8h, v26.8h, v17.8b\n"
          "ld1 {v24.4s}, [x10]\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "uaddw v19.8h, v26.8h, v19.8b\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"

          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "f\n"
          "cmp w5, #1\n"
          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          //"loop_%=:\n"
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP ":\n"
            // Mul-add left outputs.
            "smlal v21.4s, v0.4h, v9.4h\n"
            "subs w5, w5, #2\n"
            "smlal2 v22.4s, v0.8h, v9.8h\n"
            "cmp w5, #3\n"
            "smlal v23.4s, v0.4h, v12.4h\n"
            "ld1 {v9.8b}, [x12]\n"
            "smlal2 v24.4s, v0.8h, v12.8h\n"
            "smlal v21.4s, v1.4h, v10.4h\n"
            "smlal2 v22.4s, v1.8h, v10.8h\n"
            "smlal v23.4s, v1.4h, v13.4h\n"
            "smlal2 v24.4s, v1.8h, v13.8h\n"
            "smlal v21.4s, v2.4h, v11.4h\n"
            "smlal2 v22.4s, v2.8h, v11.8h\n"
            "smlal v23.4s, v2.4h, v14.4h\n"
            "smlal2 v24.4s, v2.8h, v14.8h\n"
            "smlal v21.4s, v3.4h, v12.4h\n"
            "smlal2 v22.4s, v3.8h, v12.8h\n"
            "ld1 {v12.8b}, [x13]\n"
            "smlal v23.4s, v3.4h, v15.4h\n"
            "smlal2 v24.4s, v3.8h, v15.8h\n"
            "smlal v21.4s, v4.4h, v13.4h\n"
            "smlal2 v22.4s, v4.8h, v13.8h\n"
            "smlal v23.4s, v4.4h, v16.4h\n"
            "smlal2 v24.4s, v4.8h, v16.8h\n"
            "smlal v21.4s, v5.4h, v14.4h\n"
            "smlal2 v22.4s, v5.8h, v14.8h\n"
            "smlal v23.4s, v5.4h, v17.4h\n"
            "smlal2 v24.4s, v5.8h, v17.8h\n"
            "smlal v21.4s, v6.4h, v15.4h\n"
            "smlal2 v22.4s, v6.8h, v15.8h\n"
            "ld1 {v15.8b}, [x14]\n"
            "smlal v23.4s, v6.4h, v18.4h\n"
            "smlal2 v24.4s, v6.8h, v18.8h\n"
            "ld1 {v18.8b}, [x15]\n"
            "smlal v21.4s, v7.4h, v16.4h\n"
            "smlal2 v22.4s, v7.8h, v16.8h\n"
            "smlal v23.4s, v7.4h, v19.4h\n"
            "smlal2 v24.4s, v7.8h, v19.8h\n"
            "smlal v21.4s, v8.4h, v17.4h\n"
            "smlal2 v22.4s, v8.8h, v17.8h\n"
            "smlal v23.4s, v8.4h, v20.4h\n"
            "smlal2 v24.4s, v8.8h, v20.8h\n"

            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v25.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v25.4s, v25.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v25.4s\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, w4\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, w0\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x10]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x10]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v26.8h, v9.8b\n"
            "st1 {v21.8b}, [x6], x3\n"
            "uaddw v12.8h, v26.8h, v12.8b\n"
            "st1 {v23.8b}, [x7], x3\n"
            "uaddw v15.8h, v26.8h, v15.8b\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "uaddw v18.8h, v26.8h, v18.8b\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"

            // Mul-add right outputs.
            "smlal v21.4s, v0.4h, v10.4h\n"
            "add x11, x11, %[input_width_increment]\n"
            "smlal2 v22.4s, v0.8h, v10.8h\n"
            "mov x12, x11\n"
            "smlal v23.4s, v0.4h, v13.4h\n"
            "add x13, x11, %[input_row_size]\n"
            "smlal2 v24.4s, v0.8h, v13.8h\n"
            "add x14, x13, %[input_row_size]\n"
            "smlal v21.4s, v1.4h, v11.4h\n"
            "add x15, x14, %[input_row_size]\n"
            "smlal2 v22.4s, v1.8h, v11.8h\n"
            "smlal v23.4s, v1.4h, v14.4h\n"
            "smlal2 v24.4s, v1.8h, v14.8h\n"
            "smlal v21.4s, v2.4h, v9.4h\n"
            "smlal2 v22.4s, v2.8h, v9.8h\n"
            "ld1 {v9.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v2.4h, v12.4h\n"
            "ld1 {v10.8b}, [x12], %[input_depth]\n"
            "smlal2 v24.4s, v2.8h, v12.8h\n"
            "ld1 {v11.8b}, [x12], %[input_depth]\n"
            "smlal v21.4s, v3.4h, v13.4h\n"
            "smlal2 v22.4s, v3.8h, v13.8h\n"
            "smlal v23.4s, v3.4h, v16.4h\n"
            "smlal2 v24.4s, v3.8h, v16.8h\n"
            "smlal v21.4s, v4.4h, v14.4h\n"
            "smlal2 v22.4s, v4.8h, v14.8h\n"
            "smlal v23.4s, v4.4h, v17.4h\n"
            "smlal2 v24.4s, v4.8h, v17.8h\n"
            "smlal v21.4s, v5.4h, v12.4h\n"
            "smlal2 v22.4s, v5.8h, v12.8h\n"
            "ld1 {v12.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v5.4h, v15.4h\n"
            "ld1 {v13.8b}, [x13], %[input_depth]\n"
            "smlal2 v24.4s, v5.8h, v15.8h\n"
            "ld1 {v14.8b}, [x13], %[input_depth]\n"
            "smlal v21.4s, v6.4h, v16.4h\n"
            "smlal2 v22.4s, v6.8h, v16.8h\n"
            "smlal v23.4s, v6.4h, v19.4h\n"
            "smlal2 v24.4s, v6.8h, v19.8h\n"
            "smlal v21.4s, v7.4h, v17.4h\n"
            "smlal2 v22.4s, v7.8h, v17.8h\n"
            "smlal v23.4s, v7.4h, v20.4h\n"
            "smlal2 v24.4s, v7.8h, v20.8h\n"
            "smlal v21.4s, v8.4h, v15.4h\n"
            "smlal2 v22.4s, v8.8h, v15.8h\n"
            "ld1 {v15.8b}, [x14], %[input_depth]\n"
            "smlal v23.4s, v8.4h, v18.4h\n"
            "ld1 {v16.8b}, [x14], %[input_depth]\n"
            "smlal2 v24.4s, v8.8h, v18.8h\n"
            "ld1 {v17.8b}, [x14], %[input_depth]\n"

            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "ld1 {v18.8b}, [x15], %[input_depth]\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "ld1 {v19.8b}, [x15], %[input_depth]\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "ld1 {v20.8b}, [x15], %[input_depth]\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v25.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v25.4s, v25.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v25.4s\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, w4\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, w0\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x10]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x10]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v26.8h, v9.8b\n"
            "st1 {v21.8b}, [x6], x3\n"
            "uaddw v10.8h, v26.8h, v10.8b\n"
            "st1 {v23.8b}, [x7], x3\n"
            "uaddw v11.8h, v26.8h, v11.8b\n"
            "uaddw v12.8h, v26.8h, v12.8b\n"
            "uaddw v13.8h, v26.8h, v13.8b\n"
            "uaddw v14.8h, v26.8h, v14.8b\n"
            "uaddw v15.8h, v26.8h, v15.8b\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "uaddw v16.8h, v26.8h, v16.8b\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"
            "uaddw v17.8h, v26.8h, v17.8b\n"
            "uaddw v18.8h, v26.8h, v18.8b\n"
            "uaddw v19.8h, v26.8h, v19.8b\n"
            "uaddw v20.8h, v26.8h, v20.8b\n"

            "bge " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "b\n"

          // At this point, there will be one of 2 width or 1 width leftover,
          // not both.
          "cmp w5, #2\n"
          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          // Handle last 2 columns if exists.
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER ":\n"
          // Mul-add left outputs.
          "smlal v21.4s, v0.4h, v9.4h\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "ld1 {v9.8b}, [x12]\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "smlal v23.4s, v2.4h, v14.4h\n"
          "smlal2 v24.4s, v2.8h, v14.8h\n"
          "smlal v21.4s, v3.4h, v12.4h\n"
          "smlal2 v22.4s, v3.8h, v12.8h\n"
          "ld1 {v12.8b}, [x13]\n"
          "smlal v23.4s, v3.4h, v15.4h\n"
          "smlal2 v24.4s, v3.8h, v15.8h\n"
          "smlal v21.4s, v4.4h, v13.4h\n"
          "smlal2 v22.4s, v4.8h, v13.8h\n"
          "smlal v23.4s, v4.4h, v16.4h\n"
          "smlal2 v24.4s, v4.8h, v16.8h\n"
          "smlal v21.4s, v5.4h, v14.4h\n"
          "smlal2 v22.4s, v5.8h, v14.8h\n"
          "smlal v23.4s, v5.4h, v17.4h\n"
          "smlal2 v24.4s, v5.8h, v17.8h\n"
          "smlal v21.4s, v6.4h, v15.4h\n"
          "smlal2 v22.4s, v6.8h, v15.8h\n"
          "ld1 {v15.8b}, [x14]\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "ld1 {v18.8b}, [x15]\n"
          "smlal v21.4s, v7.4h, v16.4h\n"
          "smlal2 v22.4s, v7.8h, v16.8h\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v25.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v25.4s, v25.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v25.4s\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w4\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w0\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "ld1 {v22.4s}, [x10]\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "ld1 {v24.4s}, [x10]\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "st1 {v21.8b}, [x6], x3\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "st1 {v23.8b}, [x7], x3\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"

          // Mul-add right outputs.
          "smlal v21.4s, v0.4h, v10.4h\n"
          "smlal2 v22.4s, v0.8h, v10.8h\n"
          "smlal v23.4s, v0.4h, v13.4h\n"
          "smlal2 v24.4s, v0.8h, v13.8h\n"
          "smlal v21.4s, v1.4h, v11.4h\n"
          "smlal2 v22.4s, v1.8h, v11.8h\n"
          "smlal v23.4s, v1.4h, v14.4h\n"
          "smlal2 v24.4s, v1.8h, v14.8h\n"
          "smlal v21.4s, v2.4h, v9.4h\n"
          "smlal2 v22.4s, v2.8h, v9.8h\n"
          "smlal v23.4s, v2.4h, v12.4h\n"
          "smlal2 v24.4s, v2.8h, v12.8h\n"
          "smlal v21.4s, v3.4h, v13.4h\n"
          "smlal2 v22.4s, v3.8h, v13.8h\n"
          "smlal v23.4s, v3.4h, v16.4h\n"
          "smlal2 v24.4s, v3.8h, v16.8h\n"
          "smlal v21.4s, v4.4h, v14.4h\n"
          "smlal2 v22.4s, v4.8h, v14.8h\n"
          "smlal v23.4s, v4.4h, v17.4h\n"
          "smlal2 v24.4s, v4.8h, v17.8h\n"
          "smlal v21.4s, v5.4h, v12.4h\n"
          "smlal2 v22.4s, v5.8h, v12.8h\n"
          "smlal v23.4s, v5.4h, v15.4h\n"
          "smlal2 v24.4s, v5.8h, v15.8h\n"
          "smlal v21.4s, v6.4h, v16.4h\n"
          "smlal2 v22.4s, v6.8h, v16.8h\n"
          "smlal v23.4s, v6.4h, v19.4h\n"
          "smlal2 v24.4s, v6.8h, v19.8h\n"
          "smlal v21.4s, v7.4h, v17.4h\n"
          "smlal2 v22.4s, v7.8h, v17.8h\n"
          "smlal v23.4s, v7.4h, v20.4h\n"
          "smlal2 v24.4s, v7.8h, v20.8h\n"
          "smlal v21.4s, v8.4h, v15.4h\n"
          "smlal2 v22.4s, v8.8h, v15.8h\n"
          "smlal v23.4s, v8.4h, v18.4h\n"
          "smlal2 v24.4s, v8.8h, v18.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v25.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v25.4s, v25.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v25.4s\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w4\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w0\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v21.8b}, [x6], x3\n"
          "st1 {v23.8b}, [x7], x3\n"
          "b " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "f\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER ":\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "smlal v23.4s, v2.4h, v14.4h\n"
          "smlal2 v24.4s, v2.8h, v14.8h\n"
          "smlal v21.4s, v3.4h, v12.4h\n"
          "smlal2 v22.4s, v3.8h, v12.8h\n"
          "smlal v23.4s, v3.4h, v15.4h\n"
          "smlal2 v24.4s, v3.8h, v15.8h\n"
          "smlal v21.4s, v4.4h, v13.4h\n"
          "smlal2 v22.4s, v4.8h, v13.8h\n"
          "smlal v23.4s, v4.4h, v16.4h\n"
          "smlal2 v24.4s, v4.8h, v16.8h\n"
          "smlal v21.4s, v5.4h, v14.4h\n"
          "smlal2 v22.4s, v5.8h, v14.8h\n"
          "smlal v23.4s, v5.4h, v17.4h\n"
          "smlal2 v24.4s, v5.8h, v17.8h\n"
          "smlal v21.4s, v6.4h, v15.4h\n"
          "smlal2 v22.4s, v6.8h, v15.8h\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "smlal v21.4s, v7.4h, v16.4h\n"
          "smlal2 v22.4s, v7.8h, v16.8h\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v9.16b, v21.16b, v28.16b\n"
          "and v12.16b, v22.16b, v28.16b\n"
          "and v15.16b, v23.16b, v28.16b\n"
          "and v18.16b, v24.16b, v28.16b\n"
          "sshr v9.4s, v9.4s, #31\n"
          "sshr v12.4s, v12.4s, #31\n"
          "sshr v15.4s, v15.4s, #31\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sqadd v21.4s, v21.4s, v9.4s\n"
          "sqadd v22.4s, v22.4s, v12.4s\n"
          "sqadd v23.4s, v23.4s, v15.4s\n"
          "sqadd v24.4s, v24.4s, v18.4s\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v21.8b}, [x6], x3\n"
          "st1 {v23.8b}, [x7], x3\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP ":\n"
          "subs %w[output_window_height], %w[output_window_height], #2\n"
          "add %[input_ptr], %[input_ptr], %[input_height_increment]\n"
          "cmp %w[output_window_height], #2\n"
          "add %[output_ptr], %[output_ptr], %[output_height_increment]\n"
          "bge " DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "b\n"

        DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP ":\n"
        "cmp %w[output_window_height], #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        DEPTHWISECONV_LABEL_HEIGHT_1 ":\n"
        "mov x12, %[input_ptr]\n"
        "ld1 {v9.8b}, [x12], %[input_depth]\n"
        "add x13, %[input_ptr], %[input_row_size]\n"
        "ld1 {v10.8b}, [x12], %[input_depth]\n"
        "add x14, x13, %[input_row_size]\n"
        "ld1 {v11.8b}, [x12], %[input_depth]\n"
        "add x15, x14, %[input_row_size]\n"
        "mov w5, %w[output_window_width]\n"
        "ld1 {v13.8b}, [x13], %[input_depth]\n"
        "mov x6, %[output_ptr]\n"
        "ld1 {v14.8b}, [x13], %[input_depth]\n"
        "add x7, %[output_ptr], x1\n"
        "ld1 {v15.8b}, [x13], %[input_depth]\n"
        // The height 1 / width 2 loop loads an extra 1x1 output in anticipation
        // for the next iteration. Make sure |output_window_width| is large
        // enough to handle the additional load, otherwise jump to the
        // appropriate label to handle smaller widths.
        "cmp w5, #2\n"
        "ld1 {v17.8b}, [x14], %[input_depth]\n"
        "ld1 {v18.8b}, [x14], %[input_depth]\n"
        "ld1 {v19.8b}, [x14], %[input_depth]\n"
        "ld1 {v21.4s}, [%[bias_ptr]]\n"
        "ld1 {v22.4s}, [x10]\n"
        "ld1 {v23.4s}, [%[bias_ptr]]\n"
        "ld1 {v24.4s}, [x10]\n"

        "uaddw v9.8h, v26.8h, v9.8b\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"
        "uaddw v13.8h, v26.8h, v13.8b\n"
        "uaddw v14.8h, v26.8h, v14.8b\n"
        "uaddw v15.8h, v26.8h, v15.8b\n"
        "uaddw v17.8h, v26.8h, v17.8b\n"
        "uaddw v18.8h, v26.8h, v18.8b\n"
        "uaddw v19.8h, v26.8h, v19.8b\n"

        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "f\n"
        "cmp w5, #1\n"
        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP ":\n"
          // Load inputs for 3x4 input window which corresponds to a 1x2 output
          // window.
          "smlal v21.4s, v0.4h, v9.4h\n"
          "ld1 {v12.8b}, [x12]\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "ld1 {v16.8b}, [x13]\n"
          "smlal v23.4s, v0.4h, v10.4h\n"
          "ld1 {v20.8b}, [x14]\n"
          "smlal2 v24.4s, v0.8h, v10.8h\n"
          "subs w5, w5, #2\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "cmp w5, #3\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "add %[input_ptr], %[input_ptr], %[input_width_increment]\n"
          "smlal v23.4s, v1.4h, v11.4h\n"
          "mov x12, %[input_ptr]\n"
          "smlal2 v24.4s, v1.8h, v11.8h\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "add x13, %[input_ptr], %[input_row_size]\n"
          "smlal v23.4s, v2.4h, v12.4h\n"
          "add x14, x13, %[input_row_size]\n"
          "smlal2 v24.4s, v2.8h, v12.8h\n"
          "smlal v21.4s, v3.4h, v13.4h\n"
          "add x15, x14, %[input_row_size]\n"
          "smlal2 v22.4s, v3.8h, v13.8h\n"
          "ld1 {v13.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v3.4h, v14.4h\n"
          "smlal2 v24.4s, v3.8h, v14.8h\n"
          "smlal v21.4s, v4.4h, v14.4h\n"
          "smlal2 v22.4s, v4.8h, v14.8h\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v4.4h, v15.4h\n"
          "smlal2 v24.4s, v4.8h, v15.8h\n"
          "smlal v21.4s, v5.4h, v15.4h\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "smlal2 v22.4s, v5.8h, v15.8h\n"
          "ld1 {v15.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v5.4h, v16.4h\n"
          "smlal2 v24.4s, v5.8h, v16.8h\n"
          "smlal v21.4s, v6.4h, v17.4h\n"
          "smlal2 v22.4s, v6.8h, v17.8h\n"
          "ld1 {v17.8b}, [x14], %[input_depth]\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "smlal v21.4s, v7.4h, v18.4h\n"
          "smlal2 v22.4s, v7.8h, v18.8h\n"
          "ld1 {v18.8b}, [x14], %[input_depth]\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v19.4h\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"
          "smlal2 v22.4s, v8.8h, v19.8h\n"
          "ld1 {v19.8b}, [x14], %[input_depth]\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v25.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v25.4s, v25.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v25.4s\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w4\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w0\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "ld1 {v22.4s}, [x10]\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "ld1 {v24.4s}, [x10]\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "st1 {v21.8b}, [%[output_ptr]], x3\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "st1 {v23.8b}, [%[output_ptr]], x3\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"
          "uaddw v14.8h, v26.8h, v14.8b\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v17.8h, v26.8h, v17.8b\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "uaddw v19.8h, v26.8h, v19.8b\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"

          "bge " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "b\n"

        // At this point, there will be one of 2 width or 1 width leftover,
        // not both.
        "cmp w5, #2\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        // Handle last two horizontal outputs if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER ":\n"
        "smlal v21.4s, v0.4h, v9.4h\n"
        "ld1 {v12.8b}, [x12], %[input_depth]\n"
        "smlal2 v22.4s, v0.8h, v9.8h\n"
        "ld1 {v16.8b}, [x13], %[input_depth]\n"
        "smlal v23.4s, v0.4h, v10.4h\n"
        "ld1 {v20.8b}, [x14], %[input_depth]\n"
        "smlal2 v24.4s, v0.8h, v10.8h\n"
        "smlal v21.4s, v1.4h, v10.4h\n"
        "smlal2 v22.4s, v1.8h, v10.8h\n"
        "smlal v23.4s, v1.4h, v11.4h\n"
        "smlal2 v24.4s, v1.8h, v11.8h\n"
        "smlal v21.4s, v2.4h, v11.4h\n"
        "uaddw v12.8h, v26.8h, v12.8b\n"
        "smlal2 v22.4s, v2.8h, v11.8h\n"
        "smlal v23.4s, v2.4h, v12.4h\n"
        "smlal2 v24.4s, v2.8h, v12.8h\n"
        "smlal v21.4s, v3.4h, v13.4h\n"
        "smlal2 v22.4s, v3.8h, v13.8h\n"
        "smlal v23.4s, v3.4h, v14.4h\n"
        "smlal2 v24.4s, v3.8h, v14.8h\n"
        "smlal v21.4s, v4.4h, v14.4h\n"
        "smlal2 v22.4s, v4.8h, v14.8h\n"
        "smlal v23.4s, v4.4h, v15.4h\n"
        "smlal2 v24.4s, v4.8h, v15.8h\n"
        "smlal v21.4s, v5.4h, v15.4h\n"
        "uaddw v16.8h, v26.8h, v16.8b\n"
        "smlal2 v22.4s, v5.8h, v15.8h\n"
        "smlal v23.4s, v5.4h, v16.4h\n"
        "smlal2 v24.4s, v5.8h, v16.8h\n"
        "smlal v21.4s, v6.4h, v17.4h\n"
        "smlal2 v22.4s, v6.8h, v17.8h\n"
        "smlal v23.4s, v6.4h, v18.4h\n"
        "smlal2 v24.4s, v6.8h, v18.8h\n"
        "smlal v21.4s, v7.4h, v18.4h\n"
        "smlal2 v22.4s, v7.8h, v18.8h\n"
        "smlal v23.4s, v7.4h, v19.4h\n"
        "smlal2 v24.4s, v7.8h, v19.8h\n"
        "smlal v21.4s, v8.4h, v19.4h\n"
        "uaddw v20.8h, v26.8h, v20.8b\n"
        "smlal2 v22.4s, v8.8h, v19.8h\n"
        "smlal v23.4s, v8.4h, v20.4h\n"
        "smlal2 v24.4s, v8.8h, v20.8h\n"

        "sqrdmulh v21.4s, v21.4s, v27.4s\n"
        "sqrdmulh v22.4s, v22.4s, v27.4s\n"
        "sqrdmulh v23.4s, v23.4s, v27.4s\n"
        "sqrdmulh v24.4s, v24.4s, v27.4s\n"
        "and v25.16b, v21.16b, v28.16b\n"
        "and v29.16b, v22.16b, v28.16b\n"
        "and v30.16b, v23.16b, v28.16b\n"
        "and v31.16b, v24.16b, v28.16b\n"
        "sshr v25.4s, v25.4s, #31\n"
        "sshr v29.4s, v29.4s, #31\n"
        "sshr v30.4s, v30.4s, #31\n"
        "sshr v31.4s, v31.4s, #31\n"
        "sqadd v21.4s, v21.4s, v25.4s\n"
        "sqadd v22.4s, v22.4s, v29.4s\n"
        "dup v29.4s, w2\n"
        "sqadd v23.4s, v23.4s, v30.4s\n"
        "dup v30.4s, w4\n"
        "sqadd v24.4s, v24.4s, v31.4s\n"
        "dup v31.4s, w0\n"
        "srshl v21.4s, v21.4s, v28.4s\n"
        "srshl v22.4s, v22.4s, v28.4s\n"
        "srshl v23.4s, v23.4s, v28.4s\n"
        "srshl v24.4s, v24.4s, v28.4s\n"
        "add v21.4s, v21.4s, v29.4s\n"
        "add v22.4s, v22.4s, v29.4s\n"
        "add v23.4s, v23.4s, v29.4s\n"
        "add v24.4s, v24.4s, v29.4s\n"
        "smax v21.4s, v21.4s, v30.4s\n"
        "smax v22.4s, v22.4s, v30.4s\n"
        "smax v23.4s, v23.4s, v30.4s\n"
        "smax v24.4s, v24.4s, v30.4s\n"
        "smin v21.4s, v21.4s, v31.4s\n"
        "smin v22.4s, v22.4s, v31.4s\n"
        "smin v23.4s, v23.4s, v31.4s\n"
        "smin v24.4s, v24.4s, v31.4s\n"
        "sqxtn v21.4h, v21.4s\n"
        "sqxtn v23.4h, v23.4s\n"
        "sqxtn2 v21.8h, v22.4s\n"
        "sqxtn2 v23.8h, v24.4s\n"
        "sqxtun v21.8b, v21.8h\n"
        "sqxtun v23.8b, v23.8h\n"
        "st1 {v21.8b}, [%[output_ptr]], x3\n"
        "st1 {v23.8b}, [%[output_ptr]], x3\n"
        "b " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        // Handle bottom right output if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER ":\n"
        "smlal v21.4s, v0.4h, v9.4h\n"
        "smlal2 v22.4s, v0.8h, v9.8h\n"
        "smlal v21.4s, v1.4h, v10.4h\n"
        "smlal2 v22.4s, v1.8h, v10.8h\n"
        "smlal v21.4s, v2.4h, v11.4h\n"
        "smlal2 v22.4s, v2.8h, v11.8h\n"
        "smlal v21.4s, v3.4h, v13.4h\n"
        "smlal2 v22.4s, v3.8h, v13.8h\n"
        "smlal v21.4s, v4.4h, v14.4h\n"
        "smlal2 v22.4s, v4.8h, v14.8h\n"
        "smlal v21.4s, v5.4h, v15.4h\n"
        "smlal2 v22.4s, v5.8h, v15.8h\n"
        "smlal v21.4s, v6.4h, v17.4h\n"
        "smlal2 v22.4s, v6.8h, v17.8h\n"
        "smlal v21.4s, v7.4h, v18.4h\n"
        "smlal2 v22.4s, v7.8h, v18.8h\n"
        "smlal v21.4s, v8.4h, v19.4h\n"
        "smlal2 v22.4s, v8.8h, v19.8h\n"

        "sqrdmulh v21.4s, v21.4s, v27.4s\n"
        "sqrdmulh v22.4s, v22.4s, v27.4s\n"
        "and v9.16b, v21.16b, v28.16b\n"
        "and v12.16b, v22.16b, v28.16b\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v12.4s, v12.4s, #31\n"
        "sqadd v21.4s, v21.4s, v9.4s\n"
        "sqadd v22.4s, v22.4s, v12.4s\n"
        "srshl v21.4s, v21.4s, v28.4s\n"
        "srshl v22.4s, v22.4s, v28.4s\n"
        "add v21.4s, v21.4s, v29.4s\n"
        "add v22.4s, v22.4s, v29.4s\n"
        "smax v21.4s, v21.4s, v30.4s\n"
        "smax v22.4s, v22.4s, v30.4s\n"
        "smin v21.4s, v21.4s, v31.4s\n"
        "smin v22.4s, v22.4s, v31.4s\n"
        "sqxtn v21.4h, v21.4s\n"
        "sqxtn2 v21.8h, v22.4s\n"
        "sqxtun v21.8b, v21.8h\n"
        "st1 {v21.8b}, [%[output_ptr]]\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_END ":\n"
    :
    // Outputs.
    [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
    [output_ptr] "+r"(output_ptr),
    [output_window_height] "+r"(output_window_height)
    :
    // Inputs.
    [bias_ptr] "r"(bias_ptr), [input_row_size] "r"(input_row_size),
    [input_depth] "r"(input_depth),
    [output_window_width] "r"(output_window_width),
    [input_width_increment] "r"(input_width_increment),
    [input_height_increment] "r"(input_height_increment),
    [output_height_increment] "r"(output_height_increment),
    [params_ptr] "r"(params_ptr)
    :
    // Clobbers.
    "cc", "memory",
    // We use these NEON registers.
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
    "v30", "v31",
    // We use these general-purpose registers.
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_HEIGHT_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_END
  }
};

template <>
struct DepthwiseConvWindow<8, 2, 2> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                  const int32* bias_ptr, uint8* output_ptr, int64_t input_depth,
                  int64_t input_row_size, int32 output_window_height,
                  int32 output_window_width,
                  const DepthwiseConvParams* params_ptr) {
    const int64_t input_width_increment = 4 * input_depth;
    const int64_t input_height_increment = 4 * input_row_size;
    const int64_t output_height_increment = 2 * params_ptr->output_row_size;

#define DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "1"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "2"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "3"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "4"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "5"
#define DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "6"
#define DEPTHWISECONV_LABEL_HEIGHT_1 "7"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "8"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "9"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "10"
#define DEPTHWISECONV_LABEL_HEIGHT_1_END "11"

    asm volatile(
        // Performs depthwise convolutions for a window specified by
        // |output_window_height| and |output_window_width|. The inner-most loop
        // processes 2x2 outputs, and any leftovers at the end.
        //
        // Algorithm works as follows:
        //
        //   1. Load filters of 8 depth (8x3x3). Registers v0--v8 hold filter
        //      values.
        //   2. For 2 output heights at a time:
        //        i.  For 2 output widths at a time at stride 2, a 5x5 input
        //            window is required. To avoid register exhaustion, we load
        //            the first 2 rows of the 5x5 input window into registers
        //            v9--v18, and use the same registers to load the next 2
        //            rows, and finally v9--v13 to load the last row.
        //            Accumulators for all 2x2 outputs are reserved by registers
        //            v21-v22 (top left output), v23-v24 (top right output),
        //            v19-v20 (bottom left output), v25-v26 (bottom right
        //            output).
        //        ii. Handle single leftover width if exists.
        //   3. Handle single leftover height if exists.
        //        i.  For 2 output widths at a time at stride 2, load inputs for
        //            a 1x2 (1 height, 2 width) output window (3x5 input
        //            window). Registers v9--v24 hold input values. Mul-add with
        //            accumulators v24--v27.
        //        ii. Handle single leftover width if exists.
        //
        // Loads are placed as soon as the register is no longer needed and
        // interleaved with arithmetic operations to take advantage of
        // dual-issue pipelines. We also add input offsets as far from the loads
        // as possible to give loads enough cycles to fetch data from memory.

        // Set "constant" registers. These registers may be replaced with temp
        // values from time to time when there are not enough NEON registers.
        // We use x9--x15 general purpose registers as they are caller-saved
        // temporary registers (see http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf).  // NOLINT
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "ldr w0, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "cmp %w[output_window_height], #2\n"
        "dup v28.8h, w0\n"
        "neg w9, w9\n"
        "ldr w1, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.4s, w9\n"
        "ldr w2, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w1\n"
        "ldr w3, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "dup v29.4s, w2\n"
        "ldr w4, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w3\n"
        "ldr x5, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "dup v31.4s, w4\n"
        "ldr x19, [%[params_ptr], #" STR(OFFSET_OUTPUT_ROW_SIZE) "]\n"
        "ldr w20, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"

        // Load filters and add offsets.
        "add x10, %[bias_ptr], #16\n"
        "ld1 {v0.8b}, [%[filter_ptr]], x5\n"
        "dup v9.8h, w20\n"
        "ld1 {v1.8b}, [%[filter_ptr]], x5\n"
        "uaddw v0.8h, v9.8h, v0.8b\n"
        "ld1 {v2.8b}, [%[filter_ptr]], x5\n"
        "uaddw v1.8h, v9.8h, v1.8b\n"
        "ld1 {v3.8b}, [%[filter_ptr]], x5\n"
        "uaddw v2.8h, v9.8h, v2.8b\n"
        "ld1 {v4.8b}, [%[filter_ptr]], x5\n"
        "uaddw v3.8h, v9.8h, v3.8b\n"
        "ld1 {v5.8b}, [%[filter_ptr]], x5\n"
        "uaddw v4.8h, v9.8h, v4.8b\n"
        "ld1 {v6.8b}, [%[filter_ptr]], x5\n"
        "uaddw v5.8h, v9.8h, v5.8b\n"
        "ld1 {v7.8b}, [%[filter_ptr]], x5\n"
        "uaddw v6.8h, v9.8h, v6.8b\n"
        "ld1 {v8.8b}, [%[filter_ptr]]\n"
        "uaddw v7.8h, v9.8h, v7.8b\n"
        "uaddw v8.8h, v9.8h, v8.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_2_LOOP ":\n"
          // Load the first two rows of the 5x5 input window, then reuse the
          // same registers to load subsequent rows as they become available.
          "mov x11, %[input_ptr]\n"
          "mov x12, x11\n"
          "add x13, x12, %[input_row_size]\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "mov w14, %w[output_window_width]\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          // The height 2 / width 2 loop loads an extra 1 output horizontally in
          // anticipation for the next iteration. Make sure
          // |output_window_width| is large enough to handle the additional
          // load, otherwise jump to the appropriate label to handle smaller
          // widths.
          "cmp w14, #2\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "add x15, x13, %[input_row_size]\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "mov x6, %[output_ptr]\n"
          "ld1 {v15.8b}, [x13], %[input_depth]\n"
          "add x7, %[output_ptr], x19\n"
          "ld1 {v16.8b}, [x13], %[input_depth]\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "ld1 {v22.4s}, [x10]\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "ld1 {v24.4s}, [x10]\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "ld1 {v19.4s}, [%[bias_ptr]]\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "ld1 {v20.4s}, [x10]\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "ld1 {v25.4s}, [%[bias_ptr]]\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "ld1 {v26.4s}, [x10]\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"

          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER "f\n"
          "cmp w14, #1\n"
          "beq " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          //"loop_%=:\n"
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP ":\n"
            "smlal v21.4s, v0.4h, v9.4h\n"
            "ld1 {v12.8b}, [x12], %[input_depth]\n"
            "smlal2 v22.4s, v0.8h, v9.8h\n"
            "ld1 {v13.8b}, [x12]\n"
            "add x12, x15, %[input_row_size]\n"
            "smlal v23.4s, v0.4h, v11.4h\n"
            "ld1 {v17.8b}, [x13], %[input_depth]\n"
            "smlal2 v24.4s, v0.8h, v11.8h\n"
            "ld1 {v18.8b}, [x13]\n"
            "add x13, x12, %[input_row_size]\n"
            "smlal v21.4s, v1.4h, v10.4h\n"
            "ld1 {v9.8b}, [x15], %[input_depth]\n"
            "smlal2 v22.4s, v1.8h, v10.8h\n"
            "ld1 {v10.8b}, [x15], %[input_depth]\n"
            "smlal v21.4s, v2.4h, v11.4h\n"
            "smlal2 v22.4s, v2.8h, v11.8h\n"
            "ld1 {v11.8b}, [x15], %[input_depth]\n"
            "smlal v21.4s, v3.4h, v14.4h\n"
            "smlal2 v22.4s, v3.8h, v14.8h\n"
            "ld1 {v14.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v3.4h, v16.4h\n"
            "subs w14, w14, #2\n"
            "smlal2 v24.4s, v3.8h, v16.8h\n"
            "cmp w14, #3\n"
            "smlal v21.4s, v4.4h, v15.4h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal2 v22.4s, v4.8h, v15.8h\n"
            "ld1 {v15.8b}, [x12], %[input_depth]\n"
            "smlal v21.4s, v5.4h, v16.4h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal2 v22.4s, v5.8h, v16.8h\n"
            "ld1 {v16.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v1.4h, v12.4h\n"
            "uaddw v17.8h, v28.8h, v17.8b\n"
            "smlal2 v24.4s, v1.8h, v12.8h\n"
            "ld1 {v12.8b}, [x15], %[input_depth]\n"
            "smlal v23.4s, v2.4h, v13.4h\n"
            "uaddw v18.8h, v28.8h, v18.8b\n"
            "smlal2 v24.4s, v2.8h, v13.8h\n"
            "ld1 {v13.8b}, [x15]\n"
            "smlal v23.4s, v4.4h, v17.4h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "smlal2 v24.4s, v4.8h, v17.8h\n"
            "ld1 {v17.8b}, [x12], %[input_depth]\n"
            "smlal v23.4s, v5.4h, v18.4h\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "smlal2 v24.4s, v5.8h, v18.8h\n"
            "ld1 {v18.8b}, [x12]\n"

            "smlal v21.4s, v6.4h, v9.4h\n"
            "smlal2 v22.4s, v6.8h, v9.8h\n"
            "smlal v19.4s, v0.4h, v9.4h\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"
            "smlal2 v20.4s, v0.8h, v9.8h\n"
            "ld1 {v9.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v6.4h, v11.4h\n"
            "smlal2 v24.4s, v6.8h, v11.8h\n"
            "smlal v21.4s, v7.4h, v10.4h\n"
            "smlal2 v22.4s, v7.8h, v10.8h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal v19.4s, v1.4h, v10.4h\n"
            "smlal2 v20.4s, v1.8h, v10.8h\n"
            "ld1 {v10.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v7.4h, v12.4h\n"
            "smlal2 v24.4s, v7.8h, v12.8h\n"
            "smlal v25.4s, v1.4h, v12.4h\n"
            "smlal2 v26.4s, v1.8h, v12.8h\n"
            "smlal v21.4s, v8.4h, v11.4h\n"
            "smlal2 v22.4s, v8.8h, v11.8h\n"
            "add x11, x11, %[input_width_increment]\n"
            "smlal v19.4s, v2.4h, v11.4h\n"
            "mov x12, x11\n"
            "smlal2 v20.4s, v2.8h, v11.8h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal v25.4s, v0.4h, v11.4h\n"
            "smlal2 v26.4s, v0.8h, v11.8h\n"
            "ld1 {v11.8b}, [x13], %[input_depth]\n"
            "smlal v23.4s, v8.4h, v13.4h\n"
            "ld1 {v12.8b}, [x13], %[input_depth]\n"
            "smlal2 v24.4s, v8.8h, v13.8h\n"
            "smlal v25.4s, v2.4h, v13.4h\n"
            "smlal2 v26.4s, v2.8h, v13.8h\n"
            "ld1 {v13.8b}, [x13]\n"
            "add x13, x12, %[input_row_size]\n"
            "add x15, x13, %[input_row_size]\n"

            "dup v28.4s, w9\n"
            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v27.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v27.4s, v27.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v27.4s\n"
            "dup v27.4s, w1\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, w3\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, w4\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "dup v28.8h, w0\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x10]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x10]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "st1 {v21.8b}, [x6], x5\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "st1 {v23.8b}, [x6], x5\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"

            "smlal v19.4s, v6.4h, v9.4h\n"
            "smlal2 v20.4s, v6.8h, v9.8h\n"
            "ld1 {v9.8b}, [x12], %[input_depth]\n"
            "smlal v25.4s, v6.4h, v11.4h\n"
            "smlal2 v26.4s, v6.8h, v11.8h\n"
            "smlal v19.4s, v7.4h, v10.4h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal2 v20.4s, v7.8h, v10.8h\n"
            "ld1 {v10.8b}, [x12], %[input_depth]\n"
            "smlal v25.4s, v7.4h, v12.4h\n"
            "smlal2 v26.4s, v7.8h, v12.8h\n"
            "smlal v19.4s, v8.4h, v11.4h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal2 v20.4s, v8.8h, v11.8h\n"
            "ld1 {v11.8b}, [x12], %[input_depth]\n"
            "smlal v25.4s, v8.4h, v13.4h\n"
            "uaddw v14.8h, v28.8h, v14.8b\n"
            "smlal2 v26.4s, v8.8h, v13.8h\n"
            "uaddw v16.8h, v28.8h, v16.8b\n"
            "smlal v19.4s, v3.4h, v14.4h\n"
            "uaddw v15.8h, v28.8h, v15.8b\n"
            "smlal2 v20.4s, v3.8h, v14.8h\n"
            "ld1 {v14.8b}, [x13], %[input_depth]\n"
            "smlal v25.4s, v3.4h, v16.4h\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "smlal2 v26.4s, v3.8h, v16.8h\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"
            "smlal v19.4s, v4.4h, v15.4h\n"
            "uaddw v17.8h, v28.8h, v17.8b\n"
            "smlal2 v20.4s, v4.8h, v15.8h\n"
            "ld1 {v15.8b}, [x13], %[input_depth]\n"
            "smlal v25.4s, v4.4h, v17.4h\n"
            "smlal2 v26.4s, v4.8h, v17.8h\n"
            "smlal v19.4s, v5.4h, v16.4h\n"
            "uaddw v18.8h, v28.8h, v18.8b\n"
            "smlal2 v20.4s, v5.8h, v16.8h\n"
            "ld1 {v16.8b}, [x13], %[input_depth]\n"
            "smlal v25.4s, v5.4h, v18.4h\n"
            "smlal2 v26.4s, v5.8h, v18.8h\n"

            "dup v28.4s, w9\n"
            "sqrdmulh v19.4s, v19.4s, v27.4s\n"
            "sqrdmulh v20.4s, v20.4s, v27.4s\n"
            "sqrdmulh v25.4s, v25.4s, v27.4s\n"
            "sqrdmulh v26.4s, v26.4s, v27.4s\n"
            "and v27.16b, v19.16b, v28.16b\n"
            "and v29.16b, v20.16b, v28.16b\n"
            "and v30.16b, v25.16b, v28.16b\n"
            "and v31.16b, v26.16b, v28.16b\n"
            "sshr v27.4s, v27.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v19.4s, v19.4s, v27.4s\n"
            "dup v27.4s, w1\n"
            "sqadd v20.4s, v20.4s, v29.4s\n"
            "dup v29.4s, w2\n"
            "sqadd v25.4s, v25.4s, v30.4s\n"
            "dup v30.4s, w3\n"
            "sqadd v26.4s, v26.4s, v31.4s\n"
            "dup v31.4s, w4\n"
            "srshl v19.4s, v19.4s, v28.4s\n"
            "srshl v20.4s, v20.4s, v28.4s\n"
            "srshl v25.4s, v25.4s, v28.4s\n"
            "srshl v26.4s, v26.4s, v28.4s\n"
            "dup v28.8h, w0\n"
            "add v19.4s, v19.4s, v29.4s\n"
            "add v20.4s, v20.4s, v29.4s\n"
            "add v25.4s, v25.4s, v29.4s\n"
            "add v26.4s, v26.4s, v29.4s\n"
            "smax v19.4s, v19.4s, v30.4s\n"
            "smax v20.4s, v20.4s, v30.4s\n"
            "smax v25.4s, v25.4s, v30.4s\n"
            "smax v26.4s, v26.4s, v30.4s\n"
            "smin v19.4s, v19.4s, v31.4s\n"
            "smin v20.4s, v20.4s, v31.4s\n"
            "smin v25.4s, v25.4s, v31.4s\n"
            "smin v26.4s, v26.4s, v31.4s\n"
            "sqxtn v19.4h, v19.4s\n"
            "sqxtn v25.4h, v25.4s\n"
            "sqxtn2 v19.8h, v20.4s\n"
            "ld1 {v20.4s}, [x10]\n"
            "sqxtn2 v25.8h, v26.4s\n"
            "ld1 {v26.4s}, [x10]\n"
            "sqxtun v19.8b, v19.8h\n"
            "sqxtun v25.8b, v25.8h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "st1 {v19.8b}, [x7], x5\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "st1 {v25.8b}, [x7], x5\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"
            "ld1 {v19.4s}, [%[bias_ptr]]\n"
            "uaddw v14.8h, v28.8h, v14.8b\n"
            "ld1 {v25.4s}, [%[bias_ptr]]\n"
            "uaddw v15.8h, v28.8h, v15.8b\n"
            "uaddw v16.8h, v28.8h, v16.8b\n"

            "bge " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "b\n"

          // At this point, there will be one of 2 width or 1 width leftover,
          // not both.
          "cmp w14, #2\n"
          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER "f\n"

          // Handle last 2 columns if exists.
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER ":\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "ld1 {v12.8b}, [x12], %[input_depth]\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "ld1 {v13.8b}, [x12]\n"
          "add x12, x15, %[input_row_size]\n"
          "smlal v23.4s, v0.4h, v11.4h\n"
          "ld1 {v17.8b}, [x13], %[input_depth]\n"
          "smlal2 v24.4s, v0.8h, v11.8h\n"
          "ld1 {v18.8b}, [x13]\n"
          "add x13, x12, %[input_row_size]\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "ld1 {v9.8b}, [x15], %[input_depth]\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x15], %[input_depth]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x15], %[input_depth]\n"
          "smlal v21.4s, v3.4h, v14.4h\n"
          "smlal2 v22.4s, v3.8h, v14.8h\n"
          "ld1 {v14.8b}, [x12], %[input_depth]\n"
          "smlal v23.4s, v3.4h, v16.4h\n"
          "smlal2 v24.4s, v3.8h, v16.8h\n"
          "smlal v21.4s, v4.4h, v15.4h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal2 v22.4s, v4.8h, v15.8h\n"
          "ld1 {v15.8b}, [x12], %[input_depth]\n"
          "smlal v21.4s, v5.4h, v16.4h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "smlal2 v22.4s, v5.8h, v16.8h\n"
          "ld1 {v16.8b}, [x12], %[input_depth]\n"
          "smlal v23.4s, v1.4h, v12.4h\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "smlal2 v24.4s, v1.8h, v12.8h\n"
          "ld1 {v12.8b}, [x15], %[input_depth]\n"
          "smlal v23.4s, v2.4h, v13.4h\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "smlal2 v24.4s, v2.8h, v13.8h\n"
          "ld1 {v13.8b}, [x15]\n"
          "smlal v23.4s, v4.4h, v17.4h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "smlal2 v24.4s, v4.8h, v17.8h\n"
          "ld1 {v17.8b}, [x12], %[input_depth]\n"
          "smlal v23.4s, v5.4h, v18.4h\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "smlal2 v24.4s, v5.8h, v18.8h\n"
          "ld1 {v18.8b}, [x12]\n"

          "smlal v21.4s, v6.4h, v9.4h\n"
          "smlal2 v22.4s, v6.8h, v9.8h\n"
          "smlal v19.4s, v0.4h, v9.4h\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "smlal2 v20.4s, v0.8h, v9.8h\n"
          "ld1 {v9.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v6.4h, v11.4h\n"
          "smlal2 v24.4s, v6.8h, v11.8h\n"
          "smlal v21.4s, v7.4h, v10.4h\n"
          "smlal2 v22.4s, v7.8h, v10.8h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal v19.4s, v1.4h, v10.4h\n"
          "smlal2 v20.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v7.4h, v12.4h\n"
          "smlal2 v24.4s, v7.8h, v12.8h\n"
          "smlal v25.4s, v1.4h, v12.4h\n"
          "smlal2 v26.4s, v1.8h, v12.8h\n"
          "smlal v21.4s, v8.4h, v11.4h\n"
          "smlal2 v22.4s, v8.8h, v11.8h\n"
          "smlal v19.4s, v2.4h, v11.4h\n"
          "smlal2 v20.4s, v2.8h, v11.8h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "smlal v25.4s, v0.4h, v11.4h\n"
          "smlal2 v26.4s, v0.8h, v11.8h\n"
          "ld1 {v11.8b}, [x13], %[input_depth]\n"
          "smlal v23.4s, v8.4h, v13.4h\n"
          "ld1 {v12.8b}, [x13], %[input_depth]\n"
          "smlal2 v24.4s, v8.8h, v13.8h\n"
          "smlal v25.4s, v2.4h, v13.4h\n"
          "smlal2 v26.4s, v2.8h, v13.8h\n"
          "ld1 {v13.8b}, [x13]\n"

          "dup v28.4s, w9\n"
          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v27.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v27.4s, v27.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v27.4s\n"
          "dup v27.4s, w1\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, w3\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, w4\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "dup v28.8h, w0\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "ld1 {v22.4s}, [x10]\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "ld1 {v24.4s}, [x10]\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v21.8b}, [x6], x5\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "st1 {v23.8b}, [x6]\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"

          "smlal v19.4s, v6.4h, v9.4h\n"
          "smlal2 v20.4s, v6.8h, v9.8h\n"
          "smlal v25.4s, v6.4h, v11.4h\n"
          "smlal2 v26.4s, v6.8h, v11.8h\n"
          "smlal v19.4s, v7.4h, v10.4h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal2 v20.4s, v7.8h, v10.8h\n"
          "smlal v25.4s, v7.4h, v12.4h\n"
          "smlal2 v26.4s, v7.8h, v12.8h\n"
          "smlal v19.4s, v8.4h, v11.4h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "smlal2 v20.4s, v8.8h, v11.8h\n"
          "smlal v25.4s, v8.4h, v13.4h\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "smlal2 v26.4s, v8.8h, v13.8h\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "smlal v19.4s, v3.4h, v14.4h\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "smlal2 v20.4s, v3.8h, v14.8h\n"
          "smlal v25.4s, v3.4h, v16.4h\n"
          "smlal2 v26.4s, v3.8h, v16.8h\n"
          "smlal v19.4s, v4.4h, v15.4h\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "smlal2 v20.4s, v4.8h, v15.8h\n"
          "smlal v25.4s, v4.4h, v17.4h\n"
          "smlal2 v26.4s, v4.8h, v17.8h\n"
          "smlal v19.4s, v5.4h, v16.4h\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "smlal2 v20.4s, v5.8h, v16.8h\n"
          "smlal v25.4s, v5.4h, v18.4h\n"
          "smlal2 v26.4s, v5.8h, v18.8h\n"

          "dup v28.4s, w9\n"
          "sqrdmulh v19.4s, v19.4s, v27.4s\n"
          "sqrdmulh v20.4s, v20.4s, v27.4s\n"
          "sqrdmulh v25.4s, v25.4s, v27.4s\n"
          "sqrdmulh v26.4s, v26.4s, v27.4s\n"
          "and v27.16b, v19.16b, v28.16b\n"
          "and v29.16b, v20.16b, v28.16b\n"
          "and v30.16b, v25.16b, v28.16b\n"
          "and v31.16b, v26.16b, v28.16b\n"
          "sshr v27.4s, v27.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v19.4s, v19.4s, v27.4s\n"
          "dup v27.4s, w1\n"
          "sqadd v20.4s, v20.4s, v29.4s\n"
          "dup v29.4s, w2\n"
          "sqadd v25.4s, v25.4s, v30.4s\n"
          "dup v30.4s, w3\n"
          "sqadd v26.4s, v26.4s, v31.4s\n"
          "dup v31.4s, w4\n"
          "srshl v19.4s, v19.4s, v28.4s\n"
          "srshl v20.4s, v20.4s, v28.4s\n"
          "srshl v25.4s, v25.4s, v28.4s\n"
          "srshl v26.4s, v26.4s, v28.4s\n"
          "dup v28.8h, w0\n"
          "add v19.4s, v19.4s, v29.4s\n"
          "add v20.4s, v20.4s, v29.4s\n"
          "add v25.4s, v25.4s, v29.4s\n"
          "add v26.4s, v26.4s, v29.4s\n"
          "smax v19.4s, v19.4s, v30.4s\n"
          "smax v20.4s, v20.4s, v30.4s\n"
          "smax v25.4s, v25.4s, v30.4s\n"
          "smax v26.4s, v26.4s, v30.4s\n"
          "smin v19.4s, v19.4s, v31.4s\n"
          "smin v20.4s, v20.4s, v31.4s\n"
          "smin v25.4s, v25.4s, v31.4s\n"
          "smin v26.4s, v26.4s, v31.4s\n"
          "sqxtn v19.4h, v19.4s\n"
          "sqxtn v25.4h, v25.4s\n"
          "sqxtn2 v19.8h, v20.4s\n"
          "sqxtn2 v25.8h, v26.4s\n"
          "sqxtun v19.8b, v19.8h\n"
          "sqxtun v25.8b, v25.8h\n"
          "st1 {v19.8b}, [x7], x5\n"
          "st1 {v25.8b}, [x7]\n"
          "b " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "f\n"

          // Handle last column if exists.
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER ":\n"
          // Registers v9, v10, v11, v14, v15, and v16 have already been loaded
          // with the correct values at this point. This corresponds to the
          // first two input rows of the top left output. Now load the last
          // input row for this output. Once these inputs are no longer needed,
          // load the input rows for the bottom left output.
          "add x12, x15, %[input_row_size]\n"
          "add x13, x12, %[input_row_size]\n"

          "ld1 {v12.8b}, [x15], %[input_depth]\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "ld1 {v13.8b}, [x15], %[input_depth]\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "ld1 {v17.8b}, [x15]\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x12]\n"
          "smlal v21.4s, v3.4h, v14.4h\n"
          "smlal2 v22.4s, v3.8h, v14.8h\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "smlal v21.4s, v4.4h, v15.4h\n"
          "smlal2 v22.4s, v4.8h, v15.8h\n"
          "ld1 {v15.8b}, [x13], %[input_depth]\n"
          "smlal v21.4s, v5.4h, v16.4h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal2 v22.4s, v5.8h, v16.8h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "ld1 {v16.8b}, [x13]\n"

          "smlal v21.4s, v6.4h, v12.4h\n"
          "smlal2 v22.4s, v6.8h, v12.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v7.4h, v13.4h\n"
          "smlal2 v22.4s, v7.8h, v13.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v2.4h, v17.4h\n"
          "smlal2 v24.4s, v2.8h, v17.8h\n"

          "dup v26.4s, w9\n"
          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "and v18.16b, v21.16b, v26.16b\n"
          "and v19.16b, v22.16b, v26.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v21.4s, v21.4s, v18.4s\n"
          "sqadd v22.4s, v22.4s, v19.4s\n"
          "srshl v21.4s, v21.4s, v26.4s\n"
          "srshl v22.4s, v22.4s, v26.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v21.8b}, [x6]\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"

          "smlal v23.4s, v3.4h, v9.4h\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "smlal2 v24.4s, v3.8h, v9.8h\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "smlal v23.4s, v4.4h, v10.4h\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "smlal2 v24.4s, v4.8h, v10.8h\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "smlal v23.4s, v5.4h, v11.4h\n"
          "smlal2 v24.4s, v5.8h, v11.8h\n"

          "smlal v23.4s, v6.4h, v14.4h\n"
          "smlal2 v24.4s, v6.8h, v14.8h\n"
          "smlal v23.4s, v7.4h, v15.4h\n"
          "smlal2 v24.4s, v7.8h, v15.8h\n"
          "smlal v23.4s, v8.4h, v16.4h\n"
          "smlal2 v24.4s, v8.8h, v16.8h\n"

          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v18.16b, v23.16b, v26.16b\n"
          "and v19.16b, v24.16b, v26.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v23.4s, v23.4s, v18.4s\n"
          "sqadd v24.4s, v24.4s, v19.4s\n"
          "srshl v23.4s, v23.4s, v26.4s\n"
          "srshl v24.4s, v24.4s, v26.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v23.8b}, [x7]\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP ":\n"
          "subs %w[output_window_height], %w[output_window_height], #2\n"
          "add %[input_ptr], %[input_ptr], %[input_height_increment]\n"
          "cmp %w[output_window_height], #2\n"
          "add %[output_ptr], %[output_ptr], %[output_height_increment]\n"
          "bge " DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "b\n"

        DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP ":\n"
        "cmp %w[output_window_height], #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        DEPTHWISECONV_LABEL_HEIGHT_1 ":\n"
        "mov x11, %[input_ptr]\n"
        "mov x12, x11\n"
        "add x13, x12, %[input_row_size]\n"
        "ld1 {v9.8b}, [x12], %[input_depth]\n"
        "add x15, x13, %[input_row_size]\n"
        "ld1 {v10.8b}, [x12], %[input_depth]\n"
        "mov x6, %[output_ptr]\n"
        "ld1 {v11.8b}, [x12], %[input_depth]\n"
        "mov w14, %w[output_window_width]\n"
        // The height 1 / width 2 loop loads an extra 1x1 output in anticipation
        // for the next iteration. Make sure |output_window_width| is large
        // enough to handle the additional load, otherwise jump to the
        // appropriate label to handle smaller widths.
        "cmp w14, #2\n"
        "ld1 {v12.8b}, [x13], %[input_depth]\n"
        "ld1 {v13.8b}, [x13], %[input_depth]\n"
        "ld1 {v14.8b}, [x13], %[input_depth]\n"
        "ld1 {v15.8b}, [x15], %[input_depth]\n"
        "ld1 {v16.8b}, [x15], %[input_depth]\n"
        "ld1 {v17.8b}, [x15], %[input_depth]\n"

        "uaddw v9.8h, v28.8h, v9.8b\n"
        "ld1 {v24.4s}, [%[bias_ptr]]\n"
        "uaddw v10.8h, v28.8h, v10.8b\n"
        "ld1 {v25.4s}, [x10]\n"
        "uaddw v11.8h, v28.8h, v11.8b\n"
        "ld1 {v26.4s}, [%[bias_ptr]]\n"
        "ld1 {v27.4s}, [x10]\n"
        "uaddw v12.8h, v28.8h, v12.8b\n"
        "uaddw v13.8h, v28.8h, v13.8b\n"
        "uaddw v14.8h, v28.8h, v14.8b\n"
        "uaddw v15.8h, v28.8h, v15.8b\n"
        "uaddw v16.8h, v28.8h, v16.8b\n"
        "uaddw v17.8h, v28.8h, v17.8b\n"

        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER "f\n"
        "cmp w14, #1\n"
        "beq " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP ":\n"
          "smlal v24.4s, v0.4h, v9.4h\n"
          "ld1 {v18.8b}, [x12], %[input_depth]\n"
          "smlal2 v25.4s, v0.8h, v9.8h\n"
          "ld1 {v19.8b}, [x12]\n"
          "smlal v26.4s, v0.4h, v11.4h\n"
          "ld1 {v20.8b}, [x13], %[input_depth]\n"
          "smlal2 v27.4s, v0.8h, v11.8h\n"
          "ld1 {v21.8b}, [x13]\n"
          "smlal v24.4s, v1.4h, v10.4h\n"
          "ld1 {v22.8b}, [x15], %[input_depth]\n"
          "smlal2 v25.4s, v1.8h, v10.8h\n"
          "ld1 {v23.8b}, [x15]\n"
          "smlal v24.4s, v2.4h, v11.4h\n"
          "subs w14, w14, #2\n"
          "smlal2 v25.4s, v2.8h, v11.8h\n"
          "cmp w14, #3\n"
          "smlal v24.4s, v3.4h, v12.4h\n"
          "add x11, x11, %[input_width_increment]\n"
          "smlal2 v25.4s, v3.8h, v12.8h\n"
          "mov x12, x11\n"
          "smlal v26.4s, v3.4h, v14.4h\n"
          "add x13, x12, %[input_row_size]\n"
          "smlal2 v27.4s, v3.8h, v14.8h\n"
          "add x15, x13, %[input_row_size]\n"
          "smlal v24.4s, v4.4h, v13.4h\n"
          "ld1 {v9.8b}, [x12], %[input_depth]\n"
          "smlal2 v25.4s, v4.8h, v13.8h\n"
          "ld1 {v10.8b}, [x12], %[input_depth]\n"
          "smlal v24.4s, v5.4h, v14.4h\n"
          "ld1 {v11.8b}, [x12], %[input_depth]\n"
          "smlal2 v25.4s, v5.8h, v14.8h\n"
          "ld1 {v12.8b}, [x13], %[input_depth]\n"
          "smlal v24.4s, v6.4h, v15.4h\n"
          "ld1 {v13.8b}, [x13], %[input_depth]\n"
          "smlal2 v25.4s, v6.8h, v15.8h\n"
          "ld1 {v14.8b}, [x13], %[input_depth]\n"
          "smlal v26.4s, v6.4h, v17.4h\n"
          "ld1 {v15.8b}, [x15], %[input_depth]\n"
          "smlal2 v27.4s, v6.8h, v17.8h\n"
          "smlal v24.4s, v7.4h, v16.4h\n"
          "smlal2 v25.4s, v7.8h, v16.8h\n"
          "ld1 {v16.8b}, [x15], %[input_depth]\n"
          "smlal v24.4s, v8.4h, v17.4h\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "smlal2 v25.4s, v8.8h, v17.8h\n"
          "ld1 {v17.8b}, [x15], %[input_depth]\n"
          "uaddw v19.8h, v28.8h, v19.8b\n"

          "smlal v26.4s, v1.4h, v18.4h\n"
          "uaddw v20.8h, v28.8h, v20.8b\n"
          "smlal2 v27.4s, v1.8h, v18.8h\n"
          "smlal v26.4s, v2.4h, v19.4h\n"
          "uaddw v21.8h, v28.8h, v21.8b\n"
          "smlal2 v27.4s, v2.8h, v19.8h\n"
          "smlal v26.4s, v4.4h, v20.4h\n"
          "smlal v26.4s, v5.4h, v21.4h\n"
          "smlal2 v27.4s, v4.8h, v20.8h\n"
          "uaddw v22.8h, v28.8h, v22.8b\n"
          "smlal2 v27.4s, v5.8h, v21.8h\n"
          "uaddw v23.8h, v28.8h, v23.8b\n"
          "smlal v26.4s, v7.4h, v22.4h\n"
          "smlal2 v27.4s, v7.8h, v22.8h\n"
          "smlal v26.4s, v8.4h, v23.4h\n"
          "smlal2 v27.4s, v8.8h, v23.8h\n"

          "dup v28.4s, w1\n"
          "dup v29.4s, w9\n"
          "sqrdmulh v24.4s, v24.4s, v28.4s\n"
          "sqrdmulh v25.4s, v25.4s, v28.4s\n"
          "sqrdmulh v26.4s, v26.4s, v28.4s\n"
          "sqrdmulh v27.4s, v27.4s, v28.4s\n"
          "dup v28.4s, w2\n"
          "and v30.16b, v24.16b, v29.16b\n"
          "and v31.16b, v25.16b, v29.16b\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v24.4s, v24.4s, v30.4s\n"
          "sqadd v25.4s, v25.4s, v31.4s\n"
          "and v30.16b, v26.16b, v29.16b\n"
          "and v31.16b, v27.16b, v29.16b\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v26.4s, v26.4s, v30.4s\n"
          "dup v30.4s, w3\n"
          "sqadd v27.4s, v27.4s, v31.4s\n"
          "dup v31.4s, w4\n"
          "srshl v24.4s, v24.4s, v29.4s\n"
          "srshl v25.4s, v25.4s, v29.4s\n"
          "srshl v26.4s, v26.4s, v29.4s\n"
          "srshl v27.4s, v27.4s, v29.4s\n"
          "add v24.4s, v24.4s, v28.4s\n"
          "add v25.4s, v25.4s, v28.4s\n"
          "add v26.4s, v26.4s, v28.4s\n"
          "add v27.4s, v27.4s, v28.4s\n"
          "dup v28.8h, w0\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smax v25.4s, v25.4s, v30.4s\n"
          "smax v26.4s, v26.4s, v30.4s\n"
          "smax v27.4s, v27.4s, v30.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "smin v25.4s, v25.4s, v31.4s\n"
          "smin v26.4s, v26.4s, v31.4s\n"
          "smin v27.4s, v27.4s, v31.4s\n"
          "sqxtn v24.4h, v24.4s\n"
          "sqxtn v26.4h, v26.4s\n"
          "sqxtn2 v24.8h, v25.4s\n"
          "ld1 {v25.4s}, [x10]\n"
          "sqxtn2 v26.8h, v27.4s\n"
          "ld1 {v27.4s}, [x10]\n"
          "sqxtun v24.8b, v24.8h\n"
          "sqxtun v26.8b, v26.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v24.8b}, [x6], x5\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "st1 {v26.8b}, [x6], x5\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "ld1 {v24.4s}, [%[bias_ptr]]\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "ld1 {v26.4s}, [%[bias_ptr]]\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"

          "bge " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "b\n"

        // At this point, there will be one of 2 width or 1 width leftover,
        // not both.
        "cmp w14, #2\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER "f\n"

        // Handle last two horizontal outputs if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER ":\n"
        "smlal v24.4s, v0.4h, v9.4h\n"
        "ld1 {v18.8b}, [x12], %[input_depth]\n"
        "smlal2 v25.4s, v0.8h, v9.8h\n"
        "ld1 {v19.8b}, [x12]\n"
        "smlal v26.4s, v0.4h, v11.4h\n"
        "ld1 {v20.8b}, [x13], %[input_depth]\n"
        "smlal2 v27.4s, v0.8h, v11.8h\n"
        "ld1 {v21.8b}, [x13]\n"
        "smlal v24.4s, v1.4h, v10.4h\n"
        "ld1 {v22.8b}, [x15], %[input_depth]\n"
        "smlal2 v25.4s, v1.8h, v10.8h\n"
        "ld1 {v23.8b}, [x15]\n"
        "smlal v24.4s, v2.4h, v11.4h\n"
        "smlal2 v25.4s, v2.8h, v11.8h\n"
        "smlal v24.4s, v3.4h, v12.4h\n"
        "smlal2 v25.4s, v3.8h, v12.8h\n"
        "smlal v26.4s, v3.4h, v14.4h\n"
        "smlal2 v27.4s, v3.8h, v14.8h\n"
        "smlal v24.4s, v4.4h, v13.4h\n"
        "smlal2 v25.4s, v4.8h, v13.8h\n"
        "smlal v24.4s, v5.4h, v14.4h\n"
        "smlal2 v25.4s, v5.8h, v14.8h\n"
        "smlal v24.4s, v6.4h, v15.4h\n"
        "smlal2 v25.4s, v6.8h, v15.8h\n"
        "smlal v26.4s, v6.4h, v17.4h\n"
        "smlal2 v27.4s, v6.8h, v17.8h\n"
        "smlal v24.4s, v7.4h, v16.4h\n"
        "smlal2 v25.4s, v7.8h, v16.8h\n"
        "smlal v24.4s, v8.4h, v17.4h\n"
        "uaddw v18.8h, v28.8h, v18.8b\n"
        "smlal2 v25.4s, v8.8h, v17.8h\n"
        "uaddw v19.8h, v28.8h, v19.8b\n"

        "smlal v26.4s, v1.4h, v18.4h\n"
        "uaddw v20.8h, v28.8h, v20.8b\n"
        "smlal2 v27.4s, v1.8h, v18.8h\n"
        "smlal v26.4s, v2.4h, v19.4h\n"
        "uaddw v21.8h, v28.8h, v21.8b\n"
        "smlal2 v27.4s, v2.8h, v19.8h\n"
        "smlal v26.4s, v4.4h, v20.4h\n"
        "smlal v26.4s, v5.4h, v21.4h\n"
        "smlal2 v27.4s, v4.8h, v20.8h\n"
        "uaddw v22.8h, v28.8h, v22.8b\n"
        "smlal2 v27.4s, v5.8h, v21.8h\n"
        "uaddw v23.8h, v28.8h, v23.8b\n"
        "smlal v26.4s, v7.4h, v22.4h\n"
        "smlal2 v27.4s, v7.8h, v22.8h\n"
        "smlal v26.4s, v8.4h, v23.4h\n"
        "smlal2 v27.4s, v8.8h, v23.8h\n"

        "dup v28.4s, w1\n"
        "dup v29.4s, w9\n"
        "sqrdmulh v24.4s, v24.4s, v28.4s\n"
        "sqrdmulh v25.4s, v25.4s, v28.4s\n"
        "sqrdmulh v26.4s, v26.4s, v28.4s\n"
        "sqrdmulh v27.4s, v27.4s, v28.4s\n"
        "dup v28.4s, w2\n"
        "and v30.16b, v24.16b, v29.16b\n"
        "and v31.16b, v25.16b, v29.16b\n"
        "sshr v30.4s, v30.4s, #31\n"
        "sshr v31.4s, v31.4s, #31\n"
        "sqadd v24.4s, v24.4s, v30.4s\n"
        "sqadd v25.4s, v25.4s, v31.4s\n"
        "and v30.16b, v26.16b, v29.16b\n"
        "and v31.16b, v27.16b, v29.16b\n"
        "sshr v30.4s, v30.4s, #31\n"
        "sshr v31.4s, v31.4s, #31\n"
        "sqadd v26.4s, v26.4s, v30.4s\n"
        "dup v30.4s, w3\n"
        "sqadd v27.4s, v27.4s, v31.4s\n"
        "dup v31.4s, w4\n"
        "srshl v24.4s, v24.4s, v29.4s\n"
        "srshl v25.4s, v25.4s, v29.4s\n"
        "srshl v26.4s, v26.4s, v29.4s\n"
        "srshl v27.4s, v27.4s, v29.4s\n"
        "add v24.4s, v24.4s, v28.4s\n"
        "add v25.4s, v25.4s, v28.4s\n"
        "add v26.4s, v26.4s, v28.4s\n"
        "add v27.4s, v27.4s, v28.4s\n"
        "dup v28.8h, w0\n"
        "smax v24.4s, v24.4s, v30.4s\n"
        "smax v25.4s, v25.4s, v30.4s\n"
        "smax v26.4s, v26.4s, v30.4s\n"
        "smax v27.4s, v27.4s, v30.4s\n"
        "smin v24.4s, v24.4s, v31.4s\n"
        "smin v25.4s, v25.4s, v31.4s\n"
        "smin v26.4s, v26.4s, v31.4s\n"
        "smin v27.4s, v27.4s, v31.4s\n"
        "sqxtn v24.4h, v24.4s\n"
        "sqxtn v26.4h, v26.4s\n"
        "sqxtn2 v24.8h, v25.4s\n"
        "sqxtn2 v26.8h, v27.4s\n"
        "sqxtun v24.8b, v24.8h\n"
        "sqxtun v26.8b, v26.8h\n"
        "st1 {v24.8b}, [x6], x5\n"
        "st1 {v26.8b}, [x6]\n"
        "b " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        // Handle bottom right output if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER ":\n"
        "dup v26.4s, w9\n"
        "dup v27.4s, w1\n"
        "dup v29.4s, w2\n"

        "smlal v24.4s, v0.4h, v9.4h\n"
        "smlal2 v25.4s, v0.8h, v9.8h\n"
        "smlal v24.4s, v1.4h, v10.4h\n"
        "smlal2 v25.4s, v1.8h, v10.8h\n"
        "smlal v24.4s, v2.4h, v11.4h\n"
        "smlal2 v25.4s, v2.8h, v11.8h\n"
        "smlal v24.4s, v3.4h, v12.4h\n"
        "smlal2 v25.4s, v3.8h, v12.8h\n"
        "smlal v24.4s, v4.4h, v13.4h\n"
        "smlal2 v25.4s, v4.8h, v13.8h\n"
        "smlal v24.4s, v5.4h, v14.4h\n"
        "smlal2 v25.4s, v5.8h, v14.8h\n"
        "smlal v24.4s, v6.4h, v15.4h\n"
        "smlal2 v25.4s, v6.8h, v15.8h\n"
        "smlal v24.4s, v7.4h, v16.4h\n"
        "smlal2 v25.4s, v7.8h, v16.8h\n"
        "smlal v24.4s, v8.4h, v17.4h\n"
        "smlal2 v25.4s, v8.8h, v17.8h\n"

        "sqrdmulh v24.4s, v24.4s, v27.4s\n"
        "sqrdmulh v25.4s, v25.4s, v27.4s\n"
        "and v18.16b, v24.16b, v26.16b\n"
        "and v19.16b, v25.16b, v26.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v24.4s, v24.4s, v18.4s\n"
        "sqadd v25.4s, v25.4s, v19.4s\n"
        "srshl v24.4s, v24.4s, v26.4s\n"
        "srshl v25.4s, v25.4s, v26.4s\n"
        "add v24.4s, v24.4s, v29.4s\n"
        "add v25.4s, v25.4s, v29.4s\n"
        "smax v24.4s, v24.4s, v30.4s\n"
        "smax v25.4s, v25.4s, v30.4s\n"
        "smin v24.4s, v24.4s, v31.4s\n"
        "smin v25.4s, v25.4s, v31.4s\n"
        "sqxtn v24.4h, v24.4s\n"
        "sqxtn2 v24.8h, v25.4s\n"
        "sqxtun v24.8b, v24.8h\n"
        "st1 {v24.8b}, [x6]\n"

        DEPTHWISECONV_LABEL_HEIGHT_1_END ":\n"
    :
    // Outputs.
    [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
    [output_ptr] "+r"(output_ptr),
    [output_window_height] "+r"(output_window_height)
    :
    // Inputs.
    [bias_ptr] "r"(bias_ptr), [input_row_size] "r"(input_row_size),
    [input_depth] "r"(input_depth),
    [output_window_width] "r"(output_window_width),
    [input_width_increment] "r"(input_width_increment),
    [input_height_increment] "r"(input_height_increment),
    [output_height_increment] "r"(output_height_increment),
    [params_ptr] "r"(params_ptr)
    :
    // Clobbers.
    "cc", "memory",
    // We use these NEON registers.
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
    "v30", "v31",
    // We use these general-purpose registers.
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x9", "x10", "x11", "x12", "x13", "x14", "x15",
    "x19", "x20");
#undef DEPTHWISECONV_LABEL_HEIGHT_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LEFTOVER
#undef DEPTHWISECONV_LABEL_HEIGHT_1_END
  }
};

enum class EdgeType { kCorner, kHorizontal, kVertical, kCenter };

template <EdgeType kEdgeType, int kPadWidth, int kPadHeight>
struct DepthwiseConvPartial {};

template <>
struct DepthwiseConvPartial<EdgeType::kCenter, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 1x1 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the 1x1 input and filter values.
        "ld1 {v8.8b}, [%[input_ptr]], #8\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr x11, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "ldr w10, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w10\n"
        "ld1 {v0.8b}, [%[filter_ptr]], #8\n"
        "cmp x11, #16\n"
        "ldr w10, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w10, w10\n"
        "dup v29.4s, w10\n"
        "ldr w10, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w9\n"
        "ldr w9, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w10\n"
        "dup v25.8h, w9\n"

        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v0.8h, v25.8h, v0.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "smlal v16.4s, v0.4h, v8.4h\n"
          "subs x11, x11, #8\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [%[input_ptr]], #8\n"
          "cmp x11, #16\n"
          "ld1 {v0.8b}, [%[filter_ptr]], #8\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v0.8h, v25.8h, v0.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"

        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v8", "v16", "v17", "v18", "v19", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31",
        // We use these general-purpose registers.
        "x9", "x10", "x11");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

template <>
struct DepthwiseConvPartial<EdgeType::kCorner, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 2x2 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the beginning of the 2x2 input and
        // filter values.

        // Load input and filter values.
        "ldr x15, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "ldr x9, [%[params_ptr], #" STR(OFFSET_INPUT_ROW_SIZE) "]\n"
        "cmp x15, #16\n"
        "add x12, %[input_ptr], x15\n"
        "add x13, %[input_ptr], x9\n"
        "ld1 {v8.8b}, [%[input_ptr]], #8\n"
        "add x14, x13, x15\n"
        "ld1 {v9.8b}, [x12], #8\n"
        "ldr x6, [%[params_ptr], #" STR(OFFSET_FILTER_ROW_SIZE) "]\n"

        "add x9, %[filter_ptr], x15\n"
        "ld1 {v10.8b}, [x13], #8\n"
        "add x10, %[filter_ptr], x6\n"
        "ld1 {v11.8b}, [x14], #8\n"
        "ld1 {v0.8b}, [%[filter_ptr]], #8\n"
        "add x11, x10, x15\n"
        "ld1 {v1.8b}, [x9], #8\n"
        "ld1 {v2.8b}, [x10], #8\n"
        "ld1 {v3.8b}, [x11], #8\n"

        // Load constants.
        "ldr w6, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr w7, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w6\n"
        "ldr w6, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w7\n"
        "ldr w7, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w6\n"
        "ldr w6, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w7, w7\n"
        "dup v29.4s, w7\n"
        "ldr w7, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w6\n"
        "ldr w6, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w7\n"
        "dup v25.8h, w6\n"

        // Add input and filter offsets.
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v9.8h, v26.8h, v9.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"

        "uaddw v0.8h, v25.8h, v0.8b\n"
        "uaddw v1.8h, v25.8h, v1.8b\n"
        "uaddw v2.8h, v25.8h, v2.8b\n"
        "uaddw v3.8h, v25.8h, v3.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "smlal v16.4s, v0.4h, v8.4h\n"
          "subs x15, x15, #8\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [%[input_ptr]], #8\n"
          "cmp x15, #16\n"
          "ld1 {v0.8b}, [%[filter_ptr]], #8\n"
          "smlal v16.4s, v1.4h, v9.4h\n"
          "smlal2 v17.4s, v1.8h, v9.8h\n"
          "ld1 {v9.8b}, [x12], #8\n"
          "smlal v16.4s, v2.4h, v10.4h\n"
          "ld1 {v1.8b}, [x9], #8\n"
          "smlal2 v17.4s, v2.8h, v10.8h\n"
          "ld1 {v10.8b}, [x13], #8\n"
          "smlal v16.4s, v3.4h, v11.4h\n"
          "ld1 {v2.8b}, [x10], #8\n"
          "smlal2 v17.4s, v3.8h, v11.8h\n"
          "ld1 {v11.8b}, [x14], #8\n"
          "ld1 {v3.8b}, [x11], #8\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v0.8h, v25.8h, v0.8b\n"
          "uaddw v1.8h, v25.8h, v1.8b\n"
          "uaddw v2.8h, v25.8h, v2.8b\n"
          "uaddw v3.8h, v25.8h, v3.8b\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"
        "smlal v16.4s, v1.4h, v9.4h\n"
        "smlal2 v17.4s, v1.8h, v9.8h\n"
        "smlal v16.4s, v2.4h, v10.4h\n"
        "smlal2 v17.4s, v2.8h, v10.8h\n"
        "smlal v16.4s, v3.4h, v11.4h\n"
        "smlal2 v17.4s, v3.8h, v11.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"

        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18",
        "v19", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        // We use these general-purpose registers.
        "x6", "x7", "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

template <>
struct DepthwiseConvPartial<EdgeType::kHorizontal, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 2x3 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the beginning of the 2x3 input and
        // filter values.

        // Load input and filter values.
        "ldr x7, [%[params_ptr], #" STR(OFFSET_INPUT_DEPTH) "]\n"
        "mov x12, %[input_ptr]\n"
        "ldr x11, [%[params_ptr], #" STR(OFFSET_INPUT_ROW_SIZE) "]\n"
        "mov x9, %[filter_ptr]\n"
        "ldr x14, [%[params_ptr], #" STR(OFFSET_FILTER_ROW_SIZE) "]\n"
        "add x13, x12, x11\n"
        "ldr x15, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"

        "ld1 {v8.8b}, [x12], x7\n"
        "add x10, x9, x14\n"
        "ld1 {v9.8b}, [x12], x7\n"
        "cmp x15, #16\n"
        "ld1 {v10.8b}, [x12]\n"
        "add %[input_ptr], %[input_ptr], #8\n"
        "ld1 {v11.8b}, [x13], x7\n"
        "add %[filter_ptr], %[filter_ptr], #8\n"
        "ld1 {v12.8b}, [x13], x7\n"
        "ld1 {v13.8b}, [x13]\n"

        "ld1 {v0.8b}, [x9], x7\n"
        "ld1 {v1.8b}, [x9], x7\n"
        "ld1 {v2.8b}, [x9]\n"
        "ld1 {v3.8b}, [x10], x7\n"
        "ld1 {v4.8b}, [x10], x7\n"
        "ld1 {v5.8b}, [x10]\n"

        // Load constants.
        "ldr w12, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w13, w13\n"
        "dup v29.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w13\n"
        "dup v25.8h, w12\n"

        // Add input and filter offsets.
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v9.8h, v26.8h, v9.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"
        "uaddw v12.8h, v26.8h, v12.8b\n"
        "uaddw v13.8h, v26.8h, v13.8b\n"

        "uaddw v0.8h, v25.8h, v0.8b\n"
        "uaddw v1.8h, v25.8h, v1.8b\n"
        "uaddw v2.8h, v25.8h, v2.8b\n"
        "uaddw v3.8h, v25.8h, v3.8b\n"
        "uaddw v4.8h, v25.8h, v4.8b\n"
        "uaddw v5.8h, v25.8h, v5.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "mov x12, %[input_ptr]\n"
          "subs x15, x15, #8\n"
          "add x13, x12, x11\n"
          "cmp x15, #16\n"
          "add %[input_ptr], %[input_ptr], #8\n"

          "smlal v16.4s, v0.4h, v8.4h\n"
          "mov x9, %[filter_ptr]\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [x12], x7\n"
          "smlal v16.4s, v1.4h, v9.4h\n"
          "add x10, x9, x14\n"
          "smlal2 v17.4s, v1.8h, v9.8h\n"
          "ld1 {v9.8b}, [x12], x7\n"
          "smlal v16.4s, v2.4h, v10.4h\n"
          "add %[filter_ptr], %[filter_ptr], #8\n"
          "smlal2 v17.4s, v2.8h, v10.8h\n"
          "ld1 {v10.8b}, [x12]\n"
          "smlal v16.4s, v3.4h, v11.4h\n"
          "ld1 {v0.8b}, [x9], x7\n"
          "smlal2 v17.4s, v3.8h, v11.8h\n"
          "ld1 {v11.8b}, [x13], x7\n"
          "smlal v16.4s, v4.4h, v12.4h\n"
          "ld1 {v1.8b}, [x9], x7\n"
          "smlal2 v17.4s, v4.8h, v12.8h\n"
          "ld1 {v12.8b}, [x13], x7\n"
          "smlal v16.4s, v5.4h, v13.4h\n"
          "ld1 {v2.8b}, [x9]\n"
          "smlal2 v17.4s, v5.8h, v13.8h\n"
          "ld1 {v13.8b}, [x13]\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "ld1 {v3.8b}, [x10], x7\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "ld1 {v4.8b}, [x10], x7\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "ld1 {v5.8b}, [x10]\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"

          "uaddw v0.8h, v25.8h, v0.8b\n"
          "uaddw v1.8h, v25.8h, v1.8b\n"
          "uaddw v2.8h, v25.8h, v2.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v3.8h, v25.8h, v3.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
          "uaddw v4.8h, v25.8h, v4.8b\n"
          "uaddw v5.8h, v25.8h, v5.8b\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"
        "smlal v16.4s, v1.4h, v9.4h\n"
        "smlal2 v17.4s, v1.8h, v9.8h\n"
        "smlal v16.4s, v2.4h, v10.4h\n"
        "smlal2 v17.4s, v2.8h, v10.8h\n"
        "smlal v16.4s, v3.4h, v11.4h\n"
        "smlal2 v17.4s, v3.8h, v11.8h\n"
        "smlal v16.4s, v4.4h, v12.4h\n"
        "smlal2 v17.4s, v4.8h, v12.8h\n"
        "smlal v16.4s, v5.4h, v13.4h\n"
        "smlal2 v17.4s, v5.8h, v13.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"
        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
        "v13", "v16", "v17", "v18", "v19", "v25", "v26", "v27", "v28", "v29",
        "v30", "v31",
        // We use these general-purpose registers.
        "x7", "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

template <>
struct DepthwiseConvPartial<EdgeType::kVertical, 1, 1> {
  static inline void Run(const uint8* input_ptr, const uint8* filter_ptr,
                         const int32* bias_ptr, uint8* output_ptr,
                         const DepthwiseConvParams* params_ptr) {
#define DEPTHWISECONV_LABEL_DEPTH_8_LOOP "1"
#define DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "2"
    asm volatile(
        // Performs depthwise convolutions for an input window of size 3x2 and
        // padding of 1 across the full depth. Expects |input_ptr| and
        // |filter_ptr| to be pointing to the beginning of the 3x2 input and
        // filter values.

        // Load input and filter values.
        "ldr x6, [%[params_ptr], #" STR(OFFSET_INPUT_DEPTH) "]\n"
        "mov x12, %[input_ptr]\n"
        "ldr x11, [%[params_ptr], #" STR(OFFSET_INPUT_ROW_SIZE) "]\n"
        "mov x7, %[filter_ptr]\n"
        "ldr x5, [%[params_ptr], #" STR(OFFSET_FILTER_ROW_SIZE) "]\n"
        "add x13, x12, x11\n"
        "ldr x15, [%[params_ptr], #" STR(OFFSET_OUTPUT_DEPTH) "]\n"
        "add x14, x13, x11\n"

        "ld1 {v8.8b}, [x12], x6\n"
        "add x9, x7, x5\n"
        "ld1 {v9.8b}, [x12]\n"
        "cmp x15, #16\n"
        "add x10, x9, x5\n"
        "ld1 {v10.8b}, [x13], x6\n"
        "add %[input_ptr], %[input_ptr], #8\n"
        "ld1 {v11.8b}, [x13]\n"
        "add %[filter_ptr], %[filter_ptr], #8\n"
        "ld1 {v12.8b}, [x14], x6\n"
        "ld1 {v13.8b}, [x14]\n"

        "ld1 {v0.8b}, [x7], x6\n"
        "ld1 {v1.8b}, [x7]\n"
        "ld1 {v2.8b}, [x9], x6\n"
        "ld1 {v3.8b}, [x9]\n"
        "ld1 {v4.8b}, [x10], x6\n"
        "ld1 {v5.8b}, [x10]\n"

        // Load constants.
        "ldr w12, [%[params_ptr], #" STR(OFFSET_INPUT_OFFSET) "]\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_MULTIPLIER) "]\n"
        "dup v26.8h, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_OFFSET) "]\n"
        "dup v27.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_RIGHT_SHIFT) "]\n"
        "dup v28.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MIN) "]\n"
        "neg w13, w13\n"
        "dup v29.4s, w13\n"
        "ldr w13, [%[params_ptr], #" STR(OFFSET_OUTPUT_ACTIVATION_MAX) "]\n"
        "dup v30.4s, w12\n"
        "ldr w12, [%[params_ptr], #" STR(OFFSET_FILTER_OFFSET) "]\n"
        "dup v31.4s, w13\n"
        "dup v25.8h, w12\n"

        // Add input and filter offsets.
        "uaddw v8.8h, v26.8h, v8.8b\n"
        "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
        "uaddw v9.8h, v26.8h, v9.8b\n"
        "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"
        "uaddw v12.8h, v26.8h, v12.8b\n"
        "uaddw v13.8h, v26.8h, v13.8b\n"

        "uaddw v0.8h, v25.8h, v0.8b\n"
        "uaddw v1.8h, v25.8h, v1.8b\n"
        "uaddw v2.8h, v25.8h, v2.8b\n"
        "uaddw v3.8h, v25.8h, v3.8b\n"
        "uaddw v4.8h, v25.8h, v4.8b\n"
        "uaddw v5.8h, v25.8h, v5.8b\n"

        "blt " DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_DEPTH_8_LOOP ":\n"
          "mov x12, %[input_ptr]\n"
          "subs x15, x15, #8\n"
          "add x13, x12, x11\n"
          "cmp x15, #16\n"
          "add x14, x13, x11\n"
          "add %[input_ptr], %[input_ptr], #8\n"

          "smlal v16.4s, v0.4h, v8.4h\n"
          "mov x7, %[filter_ptr]\n"
          "smlal2 v17.4s, v0.8h, v8.8h\n"
          "ld1 {v8.8b}, [x12], x6\n"
          "smlal v16.4s, v1.4h, v9.4h\n"
          "add x9, x7, x5\n"
          "smlal2 v17.4s, v1.8h, v9.8h\n"
          "add x10, x9, x5\n"
          "ld1 {v9.8b}, [x12]\n"
          "smlal v16.4s, v2.4h, v10.4h\n"
          "add %[filter_ptr], %[filter_ptr], #8\n"
          "smlal2 v17.4s, v2.8h, v10.8h\n"
          "ld1 {v10.8b}, [x13], x6\n"
          "smlal v16.4s, v3.4h, v11.4h\n"
          "ld1 {v0.8b}, [x7], x6\n"
          "smlal2 v17.4s, v3.8h, v11.8h\n"
          "ld1 {v11.8b}, [x13]\n"
          "smlal v16.4s, v4.4h, v12.4h\n"
          "ld1 {v1.8b}, [x7]\n"
          "smlal2 v17.4s, v4.8h, v12.8h\n"
          "ld1 {v12.8b}, [x14], x6\n"
          "smlal v16.4s, v5.4h, v13.4h\n"
          "ld1 {v2.8b}, [x9], x6\n"
          "smlal2 v17.4s, v5.8h, v13.8h\n"
          "ld1 {v13.8b}, [x14]\n"

          "sqrdmulh v16.4s, v16.4s, v27.4s\n"
          "ld1 {v3.8b}, [x9]\n"
          "sqrdmulh v17.4s, v17.4s, v27.4s\n"
          "ld1 {v4.8b}, [x10], x6\n"
          "and v18.16b, v16.16b, v29.16b\n"
          "ld1 {v5.8b}, [x10]\n"
          "and v19.16b, v17.16b, v29.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v16.4s, v16.4s, v18.4s\n"
          "sqadd v17.4s, v17.4s, v19.4s\n"
          "srshl v16.4s, v16.4s, v29.4s\n"
          "srshl v17.4s, v17.4s, v29.4s\n"
          "add v16.4s, v16.4s, v28.4s\n"
          "add v17.4s, v17.4s, v28.4s\n"
          "smax v16.4s, v16.4s, v30.4s\n"
          "smax v17.4s, v17.4s, v30.4s\n"
          "smin v16.4s, v16.4s, v31.4s\n"
          "smin v17.4s, v17.4s, v31.4s\n"
          "sqxtn v16.4h, v16.4s\n"
          "sqxtn2 v16.8h, v17.4s\n"
          "sqxtun v16.8b, v16.8h\n"
          "uaddw v8.8h, v26.8h, v8.8b\n"
          "st1 {v16.8b}, [%[output_ptr]], #8\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"

          "uaddw v0.8h, v25.8h, v0.8b\n"
          "uaddw v1.8h, v25.8h, v1.8b\n"
          "uaddw v2.8h, v25.8h, v2.8b\n"
          "ld1 {v16.4s}, [%[bias_ptr]], #16\n"
          "uaddw v3.8h, v25.8h, v3.8b\n"
          "ld1 {v17.4s}, [%[bias_ptr]], #16\n"
          "uaddw v4.8h, v25.8h, v4.8b\n"
          "uaddw v5.8h, v25.8h, v5.8b\n"

          "bge " DEPTHWISECONV_LABEL_DEPTH_8_LOOP "b\n"

        DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP ":\n"
        "smlal v16.4s, v0.4h, v8.4h\n"
        "smlal2 v17.4s, v0.8h, v8.8h\n"
        "smlal v16.4s, v1.4h, v9.4h\n"
        "smlal2 v17.4s, v1.8h, v9.8h\n"
        "smlal v16.4s, v2.4h, v10.4h\n"
        "smlal2 v17.4s, v2.8h, v10.8h\n"
        "smlal v16.4s, v3.4h, v11.4h\n"
        "smlal2 v17.4s, v3.8h, v11.8h\n"
        "smlal v16.4s, v4.4h, v12.4h\n"
        "smlal2 v17.4s, v4.8h, v12.8h\n"
        "smlal v16.4s, v5.4h, v13.4h\n"
        "smlal2 v17.4s, v5.8h, v13.8h\n"

        "sqrdmulh v16.4s, v16.4s, v27.4s\n"
        "sqrdmulh v17.4s, v17.4s, v27.4s\n"
        "and v18.16b, v16.16b, v29.16b\n"
        "and v19.16b, v17.16b, v29.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v16.4s, v16.4s, v18.4s\n"
        "sqadd v17.4s, v17.4s, v19.4s\n"
        "srshl v16.4s, v16.4s, v29.4s\n"
        "srshl v17.4s, v17.4s, v29.4s\n"
        "add v16.4s, v16.4s, v28.4s\n"
        "add v17.4s, v17.4s, v28.4s\n"
        "smax v16.4s, v16.4s, v30.4s\n"
        "smax v17.4s, v17.4s, v30.4s\n"
        "smin v16.4s, v16.4s, v31.4s\n"
        "smin v17.4s, v17.4s, v31.4s\n"
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.8b}, [%[output_ptr]]\n"
        :
        // Outputs.
        [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
        [output_ptr] "+r"(output_ptr), [bias_ptr] "+r"(bias_ptr)
        :
        // Inputs.
        [params_ptr] "r"(params_ptr)
        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers.
        "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
        "v13", "v16", "v17", "v18", "v19", "v25", "v26", "v27", "v28", "v29",
        "v30", "v31",
        // We use these general-purpose registers.
        "x5", "x6", "x7", "x9", "x10", "x11", "x12", "x13", "x14", "x15");
#undef DEPTHWISECONV_LABEL_DEPTH_8_LOOP
#undef DEPTHWISECONV_LABEL_DEPTH_8_AFTER_LOOP
  }
};

#undef OFFSET_INPUT_DEPTH
#undef OFFSET_INPUT_ROW_SIZE
#undef OFFSET_OUTPUT_DEPTH
#undef OFFSET_OUTPUT_ROW_SIZE
#undef OFFSET_INPUT_OFFSET
#undef OFFSET_OUTPUT_OFFSET
#undef OFFSET_FILTER_OFFSET
#undef OFFSET_OUTPUT_MULTIPLIER
#undef OFFSET_OUTPUT_ACTIVATION_MIN
#undef OFFSET_OUTPUT_ACTIVATION_MAX
#undef OFFSET_OUTPUT_RIGHT_SHIFT
#undef OFFSET_INPUT_WIDTH
#undef OFFSET_INPUT_HEIGHT
#undef OFFSET_OUTPUT_WIDTH
#undef OFFSET_OUTPUT_HEIGHT
#undef STR
#undef STR_UNEXPANDED

// Copies a subset of the input designated by |input_ptr| into |output_ptr|
// with the specified output dimensions. Supports output depths of 64 only as
// this is the cache line size.
inline void ShuffleInput(const uint8* input_ptr, int64_t input_depth,
                         int32 input_width, int32 input_height,
                         int64_t output_depth, int32 output_width,
                         int32 output_height, uint8* output_ptr) {
  const int64_t input_row_size = input_depth * input_width;
  for (int32 y = 0; y < output_height; y++) {
    const uint8* ptr = input_ptr;
    for (int32 x = 0; x < output_width; x++) {
      memcpy(output_ptr, ptr, output_depth);
      output_ptr += output_depth;
      ptr += input_depth;
    }
    input_ptr += input_row_size;
  }
}

// Calculates the input size depending on stride and output.
inline int32 get_shuffle_input_size(int32 stride, int32 output) {
  return stride * (output - 1) + 3;
}

// Indicates the input and output dimensions used when shuffling input
// activations.
struct ShuffleParams {
  int32 output_width;
  int32 output_height;
  int32 input_width;
  int32 input_height;

  ShuffleParams() = default;
  ShuffleParams(int32 output_width, int32 output_height, int32 stride_width,
                int32 stride_height)
  : output_width(output_width)
  , output_height(output_height)
  , input_width(get_shuffle_input_size(stride_width, output_width))
  , input_height(get_shuffle_input_size(stride_height, output_height)) {
  }
};

template <int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvThroughDepth {
  // Runs the DepthwiseConvWindow kernels through the depth dimension from
  // |start_depth| to |end_depth|. Keep this not inlined to maintain a small
  // binary size. We use a DepthwiseConvParams struct for read only params
  // to minimize call overhead.
  static __attribute__((noinline)) void Run(const uint8* input_ptr,
      const uint8* filter_ptr, const int32* bias_ptr, uint8* output_ptr,
      int64_t start_depth, int64_t end_depth, int64_t input_depth,
      int64_t input_row_size, int32 output_window_height,
      int32 output_window_width, const DepthwiseConvParams& params) {
    for (; start_depth <= end_depth - 8; start_depth += 8) {
      DepthwiseConvWindow<8, kStrideWidth, kStrideHeight>::Run(
          input_ptr, filter_ptr, bias_ptr, output_ptr, input_depth,
          input_row_size, output_window_height, output_window_width, &params);
      input_ptr += 8;
      output_ptr += 8;
      filter_ptr += 8;
      bias_ptr += 8;
    }
  }
};

template <int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvMultiRow {
  using ConvKernel = DepthwiseConvThroughDepth<kStrideWidth, kStrideHeight>;

  static inline void Run(const uint8* input_data, int32 start_x, int32 end_x,
                         const uint8* filter_data, const int32* bias_data,
                         uint8* output_data, const DepthwiseConvParams& params,
                         const ShuffleParams& shuffle_params,
                         uint8* shuffle_workspace) {
    TFLITE_DCHECK(shuffle_params.input_height ==
        get_shuffle_input_size(kStrideHeight, shuffle_params.output_height));
    TFLITE_DCHECK(shuffle_params.input_width ==
        get_shuffle_input_size(kStrideWidth, shuffle_params.output_width));
    TFLITE_DCHECK(64 * shuffle_params.input_width * shuffle_params.input_height
                  <= DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE);

    int32 out_x = start_x;

    // Run shuffling on inputs with sufficiently large depth and width. When
    // these parameters are large enough, more time is taken to load inputs
    // from memory. At this point, it becomes useful to prefetch and
    // preshuffle the input data to maximize locality.
    if (params.output_depth > 64 ||
        (params.output_depth <= 64 && params.input_width > 150)) {
      for (; out_x <= (end_x - shuffle_params.output_width);
             out_x += shuffle_params.output_width) {
        const uint8* input_ptr = input_data;
        const int32* bias_ptr = bias_data;
        const uint8* filter_ptr = filter_data;
        uint8* output_ptr = output_data;
        int64_t depth = 0;
        const int64_t shuffle_row_size = 64 * shuffle_params.input_width;

        for (; depth <= params.output_depth - 64; depth += 64) {
          // Preload.
          const uint8* h_ptr = input_ptr;
          for (int32 i = 0; i < shuffle_params.input_height; i++) {
            const uint8* ptr = h_ptr;
            for (int32 j = 0; j < shuffle_params.input_width; j++) {
              asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
              ptr += params.input_depth;
            }
            h_ptr += params.input_row_size;
          }

          // For a large enough input, shuffle into buckets.
          ShuffleInput(input_ptr, params.input_depth, params.input_width,
                       params.input_height, 64, shuffle_params.input_width,
                       shuffle_params.input_height, shuffle_workspace);
          ConvKernel::Run(shuffle_workspace, filter_ptr, bias_ptr, output_ptr,
                          0, 64, 64, shuffle_row_size,
                          shuffle_params.output_height,
                          shuffle_params.output_width, params);
          input_ptr += 64;
          output_ptr += 64;
          filter_ptr += 64;
          bias_ptr += 64;
        }

        // Preload.
        const uint8* h_ptr = input_ptr;
        for (int32 i = 0; i < shuffle_params.input_height; i++) {
          const uint8* ptr = h_ptr;
          for (int32 j = 0; j < shuffle_params.input_width; j++) {
            asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
            ptr += params.input_depth;
          }
          h_ptr += params.input_row_size;
        }

        // Handle leftover depth.
        ConvKernel::Run(input_ptr, filter_ptr, bias_ptr, output_ptr,
                        depth, params.output_depth, params.input_depth,
                        params.input_row_size, shuffle_params.output_height,
                        shuffle_params.output_width, params);

        input_data +=
            shuffle_params.output_width * kStrideWidth * params.input_depth;
        output_data += shuffle_params.output_width * params.output_depth;
      }
    }

    const int32 output_leftover_width = end_x - out_x;
    if (output_leftover_width > 0) {
      ConvKernel::Run(input_data, filter_data, bias_data, output_data, 0,
                      params.output_depth, params.input_depth,
                      params.input_row_size, shuffle_params.output_height,
                      output_leftover_width, params);
    }
  }
};

// Processes the borders of the input for pad_width and pad_height = 1.
// Calls 4 asm kernels:
//   * 1x1 input shape.
//   * Corner edges.
//   * Horizontal edges.
//   * Vertical edges.
inline void DepthwiseConvHandlePadding(const uint8* input_data,
    const uint8* filter_data, const int32* bias_data, uint8* output_data,
    const DepthwiseConvParams& params) {
  if (params.input_width == 1 && params.input_height == 1) {
    const uint8* filter_ptr = filter_data + params.filter_row_size
        + params.output_depth;
    DepthwiseConvPartial<EdgeType::kCenter, 1, 1>::Run(input_data, filter_ptr,
        bias_data, output_data, &params);
    return;
  }

  const int32 out_x_start_corner = 0;
  const int32 out_x_end_corner = params.output_width - 1;
  const int32 out_y_start_corner = 0;
  const int32 out_y_end_corner = params.output_height - 1;

  // Handle top row.
  const uint8* input_ptr = input_data;
  const uint8* filter_ptr = filter_data + params.filter_row_size
      + params.output_depth;
  uint8* output_ptr = output_data;

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(input_ptr, filter_ptr,
      bias_data, output_ptr, &params);

  input_ptr += (params.stride_width - 1) * params.input_depth;
  filter_ptr = filter_data + params.filter_row_size;
  output_ptr += params.output_depth;

  for (int32 out_x = out_x_start_corner + 1; out_x < out_x_end_corner;
           out_x++) {
    DepthwiseConvPartial<EdgeType::kHorizontal, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_depth;
    output_ptr += params.output_depth;
  }

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(input_ptr, filter_ptr,
      bias_data, output_ptr, &params);

  // Handle left side.
  input_ptr = input_data + (params.stride_width - 1) * params.input_row_size;
  filter_ptr = filter_data + params.input_depth;
  output_ptr = output_data + params.output_row_size;

  for (int32 out_y = out_y_start_corner + 1; out_y < out_y_end_corner;
           out_y++) {
    DepthwiseConvPartial<EdgeType::kVertical, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_row_size;
    output_ptr += params.output_row_size;
  }

  // Handle right side.
  input_ptr = input_data + (params.input_width - 2) * params.input_depth
      + (params.stride_width - 1) * params.input_row_size;
  filter_ptr = filter_data;
  output_ptr = output_data + params.output_row_size +
      (params.output_width - 1) * params.output_depth;

  for (int32 out_y = out_y_start_corner + 1; out_y < out_y_end_corner;
         out_y++) {
    DepthwiseConvPartial<EdgeType::kVertical, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_row_size;
    output_ptr += params.output_row_size;
  }

  // Handle bottom row.
  input_ptr = input_data + (params.input_height - 2) * params.input_row_size;
  filter_ptr = filter_data + params.output_depth;
  output_ptr = output_data +
      (params.output_height - 1) * params.output_row_size;

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(input_ptr, filter_ptr,
      bias_data, output_ptr, &params);

  input_ptr += (params.stride_width == 1) ? 0 : params.input_depth;
  filter_ptr = filter_data;
  output_ptr += params.output_depth;

  for (int32 out_x = out_x_start_corner + 1; out_x < out_x_end_corner;
           out_x++) {
    DepthwiseConvPartial<EdgeType::kHorizontal, 1, 1>::Run(
        input_ptr, filter_ptr, bias_data, output_ptr, &params);
    input_ptr += params.stride_width * params.input_depth;
    output_ptr += params.output_depth;
  }

  DepthwiseConvPartial<EdgeType::kCorner, 1, 1>::Run(input_ptr, filter_ptr,
      bias_data, output_ptr, &params);
}

inline bool Fast3x3FilterKernelSupported(
    const RuntimeShape& input_shape, const RuntimeShape& filter_shape,
    int32 stride_width, int32 stride_height, int32 dilation_width_factor,
    int32 dilation_height_factor, int32 pad_width, int32 pad_height,
    int32 depth_multiplier, const RuntimeShape& output_shape,
    int32 output_shift) {
  const int32 input_height = input_shape.Dims(1);
  const int32 input_width = input_shape.Dims(2);
  const int32 input_depth = input_shape.Dims(3);
  const int32 filter_height = filter_shape.Dims(1);
  const int32 filter_width = filter_shape.Dims(2);
  const int32 output_height = output_shape.Dims(1);
  const int32 output_width = output_shape.Dims(2);

  bool supported =
      filter_width == 3 && filter_height == 3 && depth_multiplier == 1 &&
      (stride_width == 1 || stride_width == 2) &&
      (stride_height == 1 || stride_height == 2) &&
      (stride_width == stride_height) && (pad_width == 0 || pad_width == 1) &&
      (pad_height == 0 || pad_height == 1) && (pad_width == pad_height) &&
      (input_depth % 8) == 0 && (output_shift <= 0) &&
      dilation_width_factor == 1 && dilation_height_factor == 1;

  if (!supported) {
    return false;
  }

  // Handle case where padding is zero but padding type is not kValid.
  // This would require special boundary case handling that is not supported.

  const int32 out_x = output_width - 1;
  const int32 out_y = output_height - 1;

  const int32 in_x_origin = (out_x * stride_width) - pad_width;
  const int32 in_y_origin = (out_y * stride_height) - pad_height;

  const int32 in_x_end = in_x_origin + filter_width;
  const int32 in_y_end = in_y_origin + filter_height;

  // Supported only if filter on the right and bottom boundary lies completely
  // within the input if padding is zero.
  if (pad_width == 0 && pad_height == 0) {
    return in_x_end <= input_width && in_y_end <= input_height;
  }

  // Else if padding is 1, supported if bottom right filter lies +1 past input
  // width and height.
  supported = in_x_end <= (input_width + 1) && in_y_end <= (input_height + 1);

  if (!supported) {
    return false;
  }

  // Shapes with width 1 and height > 1, and vice versa are not supported yet.
  if (input_width == 1) {
    supported = (input_width == input_height);
  } else if (input_height == 1) {
    supported = (input_width == input_height);
  }
  return supported;
}

inline void DepthwiseConv3x3Filter(
    const DepthwiseParams& rt_params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label(__PRETTY_FUNCTION__);
  DepthwiseConvParams params;

  const int32 stride_width = rt_params.stride_width;
  const int32 stride_height = rt_params.stride_height;
  const int32 pad_width = rt_params.padding_values.width;
  const int32 pad_height = rt_params.padding_values.height;
  const int32 depth_multiplier = rt_params.depth_multiplier;
  const int32 output_activation_min = rt_params.quantized_activation_min;
  const int32 output_activation_max = rt_params.quantized_activation_max;
  const int32 input_offset = rt_params.input_offset;
  const int32 filter_offset = rt_params.weights_offset;
  const int32 output_offset = rt_params.output_offset;
  const int32 output_multiplier = rt_params.output_multiplier;
  const int32 output_shift = rt_params.output_shift;

  params.input_depth = input_shape.Dims(3);
  params.input_width = input_shape.Dims(2);
  params.input_height = input_shape.Dims(1);
  params.input_row_size = params.input_depth * params.input_width;
  params.input_offset = input_offset;
  params.stride_width = stride_width;
  params.stride_height = stride_height;
  params.output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  params.output_width = output_shape.Dims(2);
  params.output_height = output_shape.Dims(1);
  params.output_row_size = params.output_depth * params.output_width;
  params.output_offset = output_offset;
  params.filter_offset = filter_offset;
  params.output_multiplier = output_multiplier;
  params.output_right_shift = -output_shift;
  params.output_activation_min = output_activation_min;
  params.output_activation_max = output_activation_max;

  const int32 filter_height = filter_shape.Dims(1);
  const int32 filter_width = filter_shape.Dims(2);
  params.filter_row_size = params.output_depth * filter_width;

  // Algorithm assumes below constraints. It is optimized for depth
  // multiplier of 1, 3x3 filter, no padding and strides 1 and 2.
  TFLITE_DCHECK(params.output_depth == params.input_depth * depth_multiplier);
  TFLITE_DCHECK(depth_multiplier == 1);
  TFLITE_DCHECK(filter_height == 3);
  TFLITE_DCHECK(filter_width == 3);
  TFLITE_DCHECK(stride_height == 1 || stride_height == 2);
  TFLITE_DCHECK(stride_width == 1 || stride_width == 2);
  TFLITE_DCHECK(stride_width == stride_height);
  TFLITE_DCHECK(pad_height == 0 || pad_height == 1);
  TFLITE_DCHECK(pad_width == 0 || pad_width == 1);
  TFLITE_DCHECK(pad_width == pad_height);

  const int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int64_t input_batch_size = params.input_row_size * params.input_height;
  const int64_t output_batch_size =
      params.output_row_size * params.output_height;

  ShuffleParams one_row_shuffle_params, two_row_shuffle_params,
      four_row_shuffle_params, eight_row_shuffle_params;
  if (stride_width == 1) {
    one_row_shuffle_params = ShuffleParams(30, 1, 1, 1);
    two_row_shuffle_params = ShuffleParams(22, 2, 1, 1);
    four_row_shuffle_params = ShuffleParams(14, 4, 1, 1);
    eight_row_shuffle_params = ShuffleParams(8, 8, 1, 1);
  } else {
    one_row_shuffle_params = ShuffleParams(14, 1, 2, 2);
    two_row_shuffle_params = ShuffleParams(8, 2, 2, 2);
    four_row_shuffle_params = ShuffleParams(4, 4, 2, 2);
    eight_row_shuffle_params = ShuffleParams(2, 8, 2, 2);
  }

  using conv_multirow_func_t = decltype(&DepthwiseConvMultiRow<1, 1>::Run);
  conv_multirow_func_t conv_multirow_func = DepthwiseConvMultiRow<1, 1>::Run;
  if (stride_width == 2) {
    conv_multirow_func = DepthwiseConvMultiRow<2, 2>::Run;
  }

  // Allocate maximum memory needed for shuffled input.
  // TODO(mariewhite): The size of this workspace is small enough to be
  // allocated on the stack. Eventually we will want to move it to the heap
  // and have it allocated outside of this function, like the im2col_array
  // used in gemmlowp.
  uint8 shuffle_workspace[DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE];

  for (int32 b = 0; b < batches; ++b) {
    const uint8* input_ptr = input_data + b * input_batch_size;
    uint8* output_ptr = output_data + b * output_batch_size;

    int32 out_x = 0;
    int32 out_y = 0;
    int32 end_x = params.output_width;
    int32 end_y = params.output_height;

    if (pad_width == 1 && pad_height == 1) {
      DepthwiseConvHandlePadding(input_ptr, filter_data, bias_data, output_ptr,
                                 params);

      // Update extents now that the edges have been handled.
      out_x = 1;
      end_x = params.output_width - 1;
      out_y = 1;
      end_y = params.output_height - 1;
      const int in_x = (out_x * stride_width) - pad_width;
      const int in_y = (out_y * stride_height) - pad_height;
      input_ptr += in_y * params.input_row_size + in_x * params.input_depth;
      output_ptr += out_y * params.output_row_size
          + out_x * params.output_depth;
    }

    // Shuffling shapes that maximize width over the shuffle workspace size
    // perform better since the inputs are closer together, minimizing
    // shuffling time.
    //
    // If the input shape has width large enough for the 2 row kernels,
    // we prefer to use this. The innermost loop of the kernels handle
    // 2 height x 2 width so this is the fastest path.
    //
    // If the input shape has smaller width but larger height, shuffling is
    // still useful and can benefit from kernels 4 row and 8 row kernels.

    // Handle 8 rows at a time.
    if (params.input_width < four_row_shuffle_params.input_width) {
      for (; out_y <= end_y - 8; out_y += 8) {
        conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                           output_ptr, params, eight_row_shuffle_params,
                           shuffle_workspace);
        input_ptr += 8 * stride_height * params.input_row_size;
        output_ptr += 8 * params.output_row_size;
      }
    }

    // Handle 4 rows at a time.
    if (params.input_width < two_row_shuffle_params.input_width) {
      for (; out_y <= end_y - 4; out_y += 4) {
        conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                           output_ptr, params, four_row_shuffle_params,
                           shuffle_workspace);
        input_ptr += 4 * stride_height * params.input_row_size;
        output_ptr += 4 * params.output_row_size;
      }
    }

    // Handle 2 rows at a time.
    for (; out_y <= end_y - 2; out_y += 2) {
      conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                         output_ptr, params, two_row_shuffle_params,
                         shuffle_workspace);
      input_ptr += 2 * stride_height * params.input_row_size;
      output_ptr += 2 * params.output_row_size;
    }

    // Handle one row at a time.
    for (; out_y < end_y; out_y++) {
      conv_multirow_func(input_ptr, out_x, end_x, filter_data, bias_data,
                         output_ptr, params, one_row_shuffle_params,
                         shuffle_workspace);
      input_ptr += stride_height * params.input_row_size;
      output_ptr += params.output_row_size;
    }
  }
}
// clang-format on

#endif  // __aarch64__

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_
