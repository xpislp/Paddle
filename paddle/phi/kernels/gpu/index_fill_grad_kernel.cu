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
#include <climits>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"

namespace phi {

// Backward of index_fill.
//
//   Forward: out[..., index[i], ...] = constant
//   Because the filled positions are overwritten by a constant, their gradient
//   w.r.t. x is zero; every other position passes the gradient through. There
//   is no value gradient (value is a scalar constant). So:
//     1) x_grad = copy(out_grad)
//     2) x_grad[..., index[i], ...] = 0
//
// Step 2 is an index_fill with value 0, using the same small/large index
// kernels (with grid-stride loops) as the forward pass.

template <typename T, typename IndexT, typename IndT>
__global__ void IndexFillGradSmallIndexCudaKernel(const IndT* index,
                                                  const IndexT index_size,
                                                  const IndexT inner_size,
                                                  const IndexT slice_size,
                                                  const int64_t dim_size,
                                                  T* x_grad) {
  const IndexT dim_stride = static_cast<IndexT>(dim_size) * inner_size;
  for (IndexT i = 0; i < index_size; ++i) {
    int64_t dim_idx = static_cast<int64_t>(index[i]);
    if (dim_idx < 0) dim_idx += dim_size;              // negative indexing
    if (dim_idx < 0 || dim_idx >= dim_size) continue;  // out-of-bounds guard

    const IndexT base = static_cast<IndexT>(dim_idx) * inner_size;
    for (IndexT linear =
             static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < slice_size;
         linear += static_cast<IndexT>(gridDim.x) * blockDim.x) {
      const IndexT outer_idx = linear / inner_size;
      const IndexT inner_idx = linear - outer_idx * inner_size;
      x_grad[outer_idx * dim_stride + base + inner_idx] = static_cast<T>(0);
    }
  }
}

template <typename T, typename IndexT, typename IndT>
__global__ void IndexFillGradLargeIndexCudaKernel(const IndT* index,
                                                  const IndexT inner_size,
                                                  const IndexT slice_size,
                                                  const IndexT total,
                                                  const int64_t dim_size,
                                                  T* x_grad) {
  const IndexT dim_stride = static_cast<IndexT>(dim_size) * inner_size;
  for (IndexT linear =
           static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<IndexT>(gridDim.x) * blockDim.x) {
    const IndexT index_idx = linear / slice_size;
    const IndexT slice_off = linear - index_idx * slice_size;

    int64_t dim_idx = static_cast<int64_t>(index[index_idx]);
    if (dim_idx < 0) dim_idx += dim_size;              // negative indexing
    if (dim_idx < 0 || dim_idx >= dim_size) continue;  // out-of-bounds guard

    const IndexT outer_idx = slice_off / inner_size;
    const IndexT inner_idx = slice_off - outer_idx * inner_size;
    x_grad[outer_idx * dim_stride +
           static_cast<IndexT>(dim_idx) * inner_size + inner_idx] =
        static_cast<T>(0);
  }
}

template <typename T, typename IndexT, typename IndT>
void LaunchIndexFillGradCudaKernelImpl(const phi::GPUContext& dev_ctx,
                                       const IndT* index_data,
                                       int64_t index_size,
                                       int64_t outer_size,
                                       int64_t dim_size,
                                       int64_t inner_size,
                                       T* x_grad_data) {
  const IndexT slice_size =
      static_cast<IndexT>(outer_size) * static_cast<IndexT>(inner_size);
  if (slice_size == 0) return;

  constexpr int kBlock = 128;
  const int64_t sm = dev_ctx.GetSMCount();
  auto stream = dev_ctx.stream();

  auto grid_for = [&](int64_t work) -> int {
    int64_t blocks = (work + kBlock - 1) / kBlock;
    blocks = std::min<int64_t>(blocks, sm * 8);  // cap; grid-stride covers rest
    return static_cast<int>(std::max<int64_t>(blocks, 1));
  };

  if (index_size <= funcs::kIndexFillSmallIndexThreshold) {
    IndexFillGradSmallIndexCudaKernel<T, IndexT, IndT>
        <<<grid_for(slice_size), kBlock, 0, stream>>>(
            index_data,
            static_cast<IndexT>(index_size),
            static_cast<IndexT>(inner_size),
            slice_size,
            dim_size,
            x_grad_data);
  } else {
    const IndexT total = slice_size * static_cast<IndexT>(index_size);
    IndexFillGradLargeIndexCudaKernel<T, IndexT, IndT>
        <<<grid_for(total), kBlock, 0, stream>>>(index_data,
                                                 static_cast<IndexT>(inner_size),
                                                 slice_size,
                                                 total,
                                                 dim_size,
                                                 x_grad_data);
  }
}

template <typename T, typename Context, typename IndT>
void LaunchIndexFillGradCudaKernel(const Context& dev_ctx,
                                   const DenseTensor& index,
                                   const DenseTensor& out_grad,
                                   const int dim,
                                   DenseTensor* x_grad) {
  // Step 1: x_grad = out_grad.
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

  int64_t index_size = index.numel();
  if (index_size == 0) {
    return;
  }

  int64_t outer_size = 1;
  int64_t dim_size = 1;
  int64_t inner_size = 1;
  funcs::GetIndexFillDims(
      out_grad.dims(), dim, &outer_size, &dim_size, &inner_size);

  const int64_t slice_size = outer_size * inner_size;
  if (slice_size == 0) return;

  const int64_t total = slice_size * index_size;
  const int64_t max_index = std::max(slice_size * dim_size, total);

  const IndT* index_data = index.data<IndT>();
  if (max_index <= static_cast<int64_t>(INT_MAX)) {
    LaunchIndexFillGradCudaKernelImpl<T, int32_t, IndT>(dev_ctx,
                                                        index_data,
                                                        index_size,
                                                        outer_size,
                                                        dim_size,
                                                        inner_size,
                                                        x_grad_data);
  } else {
    LaunchIndexFillGradCudaKernelImpl<T, int64_t, IndT>(dev_ctx,
                                                        index_data,
                                                        index_size,
                                                        outer_size,
                                                        dim_size,
                                                        inner_size,
                                                        x_grad_data);
  }
}

// Top-level backward kernel entry: validates inputs and dispatches.
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

  auto out_grad_dims = out_grad.dims();
  const int rank = out_grad_dims.size();

  if (dim < 0) {
    dim += rank;
  }

  PADDLE_ENFORCE_GE(
      dim,
      0,
      common::errors::InvalidArgument("The dimension index should be greater "
                                      "than or equal to 0, but got %d.",
                                      dim));
  PADDLE_ENFORCE_LT(
      dim,
      rank,
      common::errors::InvalidArgument(
          "The dimension index should be less than rank %d, but got %d.",
          rank,
          dim));

  if (index.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

  if (index.dtype() == DataType::INT32) {
    LaunchIndexFillGradCudaKernel<T, Context, int32_t>(
        dev_ctx, index, out_grad, dim, x_grad);
  } else if (index.dtype() == DataType::INT64) {
    LaunchIndexFillGradCudaKernel<T, Context, int64_t>(
        dev_ctx, index, out_grad, dim, x_grad);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        DataTypeToString(index.dtype())));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill_grad,
                   GPU,
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
