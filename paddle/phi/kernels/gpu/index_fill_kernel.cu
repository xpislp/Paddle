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
#include <climits>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"

namespace phi {

// index_fill forward, modeled on PyTorch's index_fill_
// (aten/src/ATen/native/cuda/Indexing.cu). After the copy-then-overwrite step
// the output is contiguous, so we treat it as [outer_size, dim_size,
// inner_size] (see funcs::GetIndexFillDims) and write the constant value into
// every indexed slice.
//
// IndexT: int32_t when offsets/`total` fit in 32 bits (faster GPU div/mod),
//         int64_t otherwise.
// IndT:   matches the index tensor dtype (int32_t or int64_t) so int32 indices
//         need no extra cast kernel.

// Small-index kernel: outer loop over the (few) indices, inner grid-stride loop
// over one slice. Loading each index once avoids redundant global reads.
template <typename T, typename IndexT, typename IndT>
__global__ void IndexFillSmallIndexCudaKernel(const IndT* index,
                                              const IndexT index_size,
                                              const IndexT inner_size,
                                              const IndexT slice_size,
                                              const int64_t dim_size,
                                              const T fill_value,
                                              T* out) {
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
      out[outer_idx * dim_stride + base + inner_idx] = fill_value;
    }
  }
}

// Large-index kernel: single grid-stride loop over all `total` written
// elements. Consecutive threads cover consecutive `inner_idx`, keeping writes
// coalesced within each contiguous run.
template <typename T, typename IndexT, typename IndT>
__global__ void IndexFillLargeIndexCudaKernel(const IndT* index,
                                              const IndexT inner_size,
                                              const IndexT slice_size,
                                              const IndexT total,
                                              const int64_t dim_size,
                                              const T fill_value,
                                              T* out) {
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
    out[outer_idx * dim_stride + static_cast<IndexT>(dim_idx) * inner_size +
        inner_idx] = fill_value;
  }
}

template <typename T, typename IndexT, typename IndT>
void LaunchIndexFillCudaKernelImpl(const phi::GPUContext& dev_ctx,
                                   const IndT* index_data,
                                   int64_t index_size,
                                   int64_t outer_size,
                                   int64_t dim_size,
                                   int64_t inner_size,
                                   T fill_value,
                                   T* out_data) {
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
    IndexFillSmallIndexCudaKernel<T, IndexT, IndT>
        <<<grid_for(slice_size), kBlock, 0, stream>>>(
            index_data,
            static_cast<IndexT>(index_size),
            static_cast<IndexT>(inner_size),
            slice_size,
            dim_size,
            fill_value,
            out_data);
  } else {
    const IndexT total = slice_size * static_cast<IndexT>(index_size);
    IndexFillLargeIndexCudaKernel<T, IndexT, IndT>
        <<<grid_for(total), kBlock, 0, stream>>>(index_data,
                                                 static_cast<IndexT>(inner_size),
                                                 slice_size,
                                                 total,
                                                 dim_size,
                                                 fill_value,
                                                 out_data);
  }
}

// Computes the three-segment sizes, picks int32 vs int64 index math, and
// launches the kernels. `IndT` is the index dtype so int32 indices need no
// extra cast.
template <typename T, typename Context, typename IndT>
void LaunchIndexFillCudaKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               int dim,
                               const DenseTensor& index,
                               const Scalar& value,
                               DenseTensor* out) {
  T fill_value = value.to<T>();

  // "Copy-then-overwrite": copy x into out, then fill the indexed positions.
  // Skip the copy when out already aliases x (inplace mode).
  bool is_initialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (!is_initialized || (x.data<T>() != out->data<T>())) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  int64_t index_size = index.numel();
  if (index_size == 0) {
    return;
  }

  int64_t outer_size = 1;
  int64_t dim_size = 1;
  int64_t inner_size = 1;
  funcs::GetIndexFillDims(x.dims(), dim, &outer_size, &dim_size, &inner_size);

  const int64_t slice_size = outer_size * inner_size;
  if (slice_size == 0) return;

  // Largest value the kernels must represent: the largest offset is bounded by
  // numel = slice_size * dim_size, and `total` may exceed it when indices
  // repeat. Use int32 math when it all fits.
  const int64_t total = slice_size * index_size;
  const int64_t max_index = std::max(slice_size * dim_size, total);

  const IndT* index_data = index.data<IndT>();
  if (max_index <= static_cast<int64_t>(INT_MAX)) {
    LaunchIndexFillCudaKernelImpl<T, int32_t, IndT>(dev_ctx,
                                                    index_data,
                                                    index_size,
                                                    outer_size,
                                                    dim_size,
                                                    inner_size,
                                                    fill_value,
                                                    out_data);
  } else {
    LaunchIndexFillCudaKernelImpl<T, int64_t, IndT>(dev_ctx,
                                                    index_data,
                                                    index_size,
                                                    outer_size,
                                                    dim_size,
                                                    inner_size,
                                                    fill_value,
                                                    out_data);
  }
}

// Top-level kernel entry: validates inputs and dispatches on index dtype.
template <typename T, typename Context>
void IndexFillKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     int dim,
                     const Scalar& value,
                     DenseTensor* out) {
  // Early return for zero-element output tensor.
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto x_dims = x.dims();
  const int rank = x_dims.size();

  // Normalize negative dim and validate range.
  int real_dim = dim;
  if (real_dim < 0) {
    real_dim += rank;
  }

  PADDLE_ENFORCE_GE(real_dim,
                    0,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));
  PADDLE_ENFORCE_LT(real_dim,
                    rank,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));

  // index_fill only supports 1-D index tensors (a list of positions along dim).
  PADDLE_ENFORCE_EQ(index.dims().size(),
                    1,
                    common::errors::InvalidArgument(
                        "The index tensor must be 1-D, but received %d-D.",
                        index.dims().size()));

  // Empty index means nothing to fill; just copy x to out.
  if (index.numel() == 0) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  // Dispatch directly on index dtype — no cast kernel needed.
  if (index.dtype() == DataType::INT32) {
    LaunchIndexFillCudaKernel<T, Context, int32_t>(
        dev_ctx, x, real_dim, index, value, out);
  } else if (index.dtype() == DataType::INT64) {
    LaunchIndexFillCudaKernel<T, Context, int64_t>(
        dev_ctx, x, real_dim, index, value, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        DataTypeToString(index.dtype())));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_fill,
                   GPU,
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
