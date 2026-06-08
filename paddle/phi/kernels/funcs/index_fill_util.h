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

#pragma once

#include "paddle/phi/core/ddim.h"

namespace phi {
namespace funcs {

// Below this many indices the GPU kernel uses the "small index" path: each
// index is loaded once into a register and reused across the whole slice.
// PyTorch (aten/src/ATen/native/cuda/Indexing.cu) uses the same threshold.
constexpr int64_t kIndexFillSmallIndexThreshold = 16;

// Collapse an N-D tensor into the 3D logical shape [outer_size, dim_size,
// inner_size] around the target `axis`. This is the shared layout used by both
// the CPU and GPU index_fill forward/backward kernels:
//
//     outer_size = prod(dims[0 .. axis-1])   (dims before axis)
//     dim_size   = dims[axis]                (the target dim)
//     inner_size = prod(dims[axis+1 .. end]) (dims after axis)
//
//     offset = outer_idx * (dim_size * inner_size)
//            + dim_idx   *  inner_size
//            + inner_idx
inline void GetIndexFillDims(const phi::DDim& dims,
                             int axis,
                             int64_t* outer_size,
                             int64_t* dim_size,
                             int64_t* inner_size) {
  const int rank = dims.size();
  int64_t outer = 1;
  int64_t inner = 1;
  for (int i = 0; i < axis; ++i) outer *= dims[i];
  for (int i = axis + 1; i < rank; ++i) inner *= dims[i];
  *outer_size = outer;
  *dim_size = dims[axis];
  *inner_size = inner;
}

}  // namespace funcs
}  // namespace phi
