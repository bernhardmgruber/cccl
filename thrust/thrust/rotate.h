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

#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/rotate.h>
#include <thrust/system/detail/generic/rotate.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

//! \addtogroup reordering
//! \ingroup algorithms
//! \{

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator rotate(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator middle,
  ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::rotate");
  using thrust::system::detail::generic::rotate;
  return rotate(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, middle, last);
}

template <typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator rotate(ForwardIterator first, ForwardIterator middle, ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::rotate");
  using thrust::system::detail::generic::select_system;
  using System = thrust::iterator_system_t<ForwardIterator>;
  System system;
  return thrust::rotate(select_system(system), first, middle, last);
}

//! \}

THRUST_NAMESPACE_END
