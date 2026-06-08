// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/index_fill_kernel.h"

#include <algorithm>
#include <cstring>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"

namespace phi {

// CPU index_fill core, using the same [outer_size, dim_size, inner_size]
// decomposition as the GPU kernel:
//
//     offset = outer * (dim_size * inner_size) + dim_idx * inner_size
//
// We parallelize over the (outer * index) slices rather than over the index
// dimension alone — index_fill is usually called with very few indices, so
// parallelizing only over indices leaves most cores idle. Each slice is a
// contiguous run of `inner_size` elements, filled with a single std::fill_n.
// `IndT` is the index dtype, so int32 indices need no separate cast pass.
template <typename T, typename IndT>
void IndexFillInner(const IndT* index_data,
                    int64_t index_size,
                    int64_t outer_size,
                    int64_t dim_size,
                    int64_t inner_size,
                    T fill_value,
                    T* out) {
  const int64_t num_slices = outer_size * index_size;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t k = 0; k < num_slices; ++k) {
    const int64_t outer = k / index_size;
    const int64_t i = k % index_size;

    int64_t dim_idx = static_cast<int64_t>(index_data[i]);
    if (dim_idx < 0) dim_idx += dim_size;  // negative indexing

    T* dst = out + outer * dim_size * inner_size + dim_idx * inner_size;
    std::fill_n(dst, inner_size, fill_value);
  }
}

template <typename T, typename Context, typename IndT>
void LaunchIndexFillKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           int axis,
                           const T fill_value,
                           DenseTensor* out) {
  const T* x_data = x.data<T>();
  const int64_t numel = x.numel();
  bool is_initialized = out->initialized();

  T* out_data = dev_ctx.template Alloc<T>(out);

  // Copy-then-overwrite; skip the copy when out already aliases x (inplace).
  if (!is_initialized || (x.data<T>() != out->data<T>())) {
    std::memcpy(out_data, x_data, numel * sizeof(T));
  }

  const int64_t index_size = index.numel();
  if (index_size == 0) {
    return;
  }

  // Three-segment decomposition around `axis`.
  int64_t outer_size = 1;
  int64_t axis_size = 1;
  int64_t inner_size = 1;
  funcs::GetIndexFillDims(x.dims(), axis, &outer_size, &axis_size, &inner_size);

  IndexFillInner<T, IndT>(index.data<IndT>(),
                          index_size,
                          outer_size,
                          axis_size,
                          inner_size,
                          fill_value,
                          out_data);
}

template <typename T, typename Context>
void IndexFillKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     int axis,
                     const Scalar& value,
                     DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const int64_t x_dims_size = x.dims().size();
  if (axis < 0) {
    axis += x_dims_size;
  }

  T fill_value = value.to<T>();

  // Dispatch on index dtype to avoid a separate int32 -> int64 cast pass.
  if (index.dtype() == DataType::INT32) {
    LaunchIndexFillKernel<T, Context, int32_t>(
        dev_ctx, x, index, axis, fill_value, out);
  } else if (index.dtype() == DataType::INT64) {
    LaunchIndexFillKernel<T, Context, int64_t>(
        dev_ctx, x, index, axis, fill_value, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        DataTypeToString(index.dtype())));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexFillKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
