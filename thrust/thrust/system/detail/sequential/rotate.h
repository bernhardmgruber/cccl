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

#include <thrust/system/detail/sequential/execution_policy.h>

#include <cuda/std/__algorithm/rotate.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator rotate(
  sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator middle, ForwardIterator last)
{
  return ::cuda::std::rotate(first, middle, last);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
