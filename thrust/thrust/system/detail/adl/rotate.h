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

#include <thrust/system/detail/sequential/rotate.h>

// Some build systems need a hint to know which files we actually include
#if 0
#  include <thrust/system/cpp/detail/rotate.h>
#  include <thrust/system/cuda/detail/rotate.h>
#  include <thrust/system/omp/detail/rotate.h>
#  include <thrust/system/tbb/detail/rotate.h>
#endif

#define THRUST_HOST_SYSTEM_ROTATE_HEADER <__THRUST_HOST_SYSTEM_ROOT/detail/rotate.h>
#include THRUST_HOST_SYSTEM_ROTATE_HEADER
#undef THRUST_HOST_SYSTEM_ROTATE_HEADER

#define THRUST_DEVICE_SYSTEM_ROTATE_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/detail/rotate.h>
#include THRUST_DEVICE_SYSTEM_ROTATE_HEADER
#undef THRUST_DEVICE_SYSTEM_ROTATE_HEADER
