// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/reverse.h>
#include <thrust/system/detail/generic/tag.h>

#include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator rotate(
  const thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator middle,
  ForwardIterator last)
{
  // TODO(bgruber): first two reverse could be run in parallel
  thrust::reverse(exec, first, middle);
  thrust::reverse(exec, middle, last);
  thrust::reverse(exec, first, last);
  return first + ::cuda::std::distance(middle, last);
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/adjacent_difference.inl>
