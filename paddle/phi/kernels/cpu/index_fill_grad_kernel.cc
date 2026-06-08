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

#include "paddle/phi/kernels/index_fill_grad_kernel.h"

#include <algorithm>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"

namespace phi {

// Backward of index_fill: x_grad = copy(out_grad), then zero the positions that
// the forward overwrote with a constant. That second step is just an index_fill
// with value 0, sharing the forward's parallelization scheme: parallelize over
// the (outer * index) slices and fill each contiguous inner run with std::fill_n.
template <typename T, typename IndT>
void IndexFillGradInner(const IndT* index_data,
                        int64_t index_size,
                        int64_t outer_size,
                        int64_t dim_size,
                        int64_t inner_size,
                        T* x_grad) {
  const int64_t num_slices = outer_size * index_size;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t k = 0; k < num_slices; ++k) {
    const int64_t outer = k / index_size;
    const int64_t i = k % index_size;

    int64_t dim_idx = static_cast<int64_t>(index_data[i]);
    if (dim_idx < 0) dim_idx += dim_size;  // negative indexing

    T* dst = x_grad + outer * dim_size * inner_size + dim_idx * inner_size;
    std::fill_n(dst, inner_size, static_cast<T>(0));
  }
}

template <typename T, typename Context, typename IndT>
void LaunchIndexFillGradKernel(const Context& dev_ctx,
                               const DenseTensor& index,
                               const DenseTensor& out_grad,
                               const int dim,
                               DenseTensor* x_grad) {
  // Step 1: x_grad = out_grad.
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

  const int64_t index_size = index.numel();
  if (index_size == 0) {
    return;
  }

  // Three-segment decomposition around `dim`.
  int64_t outer_size = 1;
  int64_t dim_size = 1;
  int64_t inner_size = 1;
  funcs::GetIndexFillDims(
      out_grad.dims(), dim, &outer_size, &dim_size, &inner_size);

  // Step 2: zero out the filled positions.
  IndexFillGradInner<T, IndT>(index.data<IndT>(),
                              index_size,
                              outer_size,
                              dim_size,
                              inner_size,
                              x_grad_data);
}

template <typename T, typename Context>
void IndexFillGradKernel(const Context& dev_ctx,
                         const DenseTensor& index,
                         const DenseTensor& out_grad,
                         int dim,
                         DenseTensor* x_grad) {
  if (out_grad.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }

  const int rank = out_grad.dims().size();
  if (dim < 0) {
    dim += rank;
  }

  if (index.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

  // Dispatch on index dtype to avoid a separate int32 -> int64 cast pass.
  if (index.dtype() == DataType::INT32) {
    LaunchIndexFillGradKernel<T, Context, int32_t>(
        dev_ctx, index, out_grad, dim, x_grad);
  } else if (index.dtype() == DataType::INT64) {
    LaunchIndexFillGradKernel<T, Context, int64_t>(
        dev_ctx, index, out_grad, dim, x_grad);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        DataTypeToString(index.dtype())));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexFillGradKernel,
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
