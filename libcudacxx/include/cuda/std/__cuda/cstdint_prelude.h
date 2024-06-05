// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_CSTDINT_PRELUDE_H
#define _LIBCUDACXX___CUDA_CSTDINT_PRELUDE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _CCCL_COMPILER_NVRTC
#  include <cstdint>
#else // ^^^ !_CCCL_COMPILER_NVRTC ^^^ / vvv _CCCL_COMPILER_NVRTC vvv
using int8_t   = signed char;
using uint8_t  = unsigned char;
using int16_t  = signed short;
using uint16_t = unsigned short;
using int32_t  = signed int;
using uint32_t = unsigned int;
using int64_t  = signed long long;
using uint64_t = unsigned long long;

#  define _LIBCUDACXX_ADDITIONAL_INTS(N)   \
    using int_fast##N##_t   = int##N##_t;  \
    using uint_fast##N##_t  = uint##N##_t; \
    using int_least##N##_t  = int##N##_t;  \
    using uint_least##N##_t = uint##N##_t

_LIBCUDACXX_ADDITIONAL_INTS(8);
_LIBCUDACXX_ADDITIONAL_INTS(16);
_LIBCUDACXX_ADDITIONAL_INTS(32);
_LIBCUDACXX_ADDITIONAL_INTS(64);
#  undef _LIBCUDACXX_ADDITIONAL_INTS

using intptr_t  = int64_t;
using uintptr_t = uint64_t;
using intmax_t  = int64_t;
using uintmax_t = uint64_t;

#  define INT8_MIN        SCHAR_MIN
#  define INT16_MIN       SHRT_MIN
#  define INT32_MIN       INT_MIN
#  define INT64_MIN       LLONG_MIN
#  define INT8_MAX        SCHAR_MAX
#  define INT16_MAX       SHRT_MAX
#  define INT32_MAX       INT_MAX
#  define INT64_MAX       LLONG_MAX
#  define UINT8_MAX       UCHAR_MAX
#  define UINT16_MAX      USHRT_MAX
#  define UINT32_MAX      UINT_MAX
#  define UINT64_MAX      ULLONG_MAX
#  define INT_FAST8_MIN   SCHAR_MIN
#  define INT_FAST16_MIN  SHRT_MIN
#  define INT_FAST32_MIN  INT_MIN
#  define INT_FAST64_MIN  LLONG_MIN
#  define INT_FAST8_MAX   SCHAR_MAX
#  define INT_FAST16_MAX  SHRT_MAX
#  define INT_FAST32_MAX  INT_MAX
#  define INT_FAST64_MAX  LLONG_MAX
#  define UINT_FAST8_MAX  UCHAR_MAX
#  define UINT_FAST16_MAX USHRT_MAX
#  define UINT_FAST32_MAX UINT_MAX
#  define UINT_FAST64_MAX ULLONG_MAX

#  define INT8_C(X)    ((int_least8_t) (X))
#  define INT16_C(X)   ((int_least16_t) (X))
#  define INT32_C(X)   ((int_least32_t) (X))
#  define INT64_C(X)   ((int_least64_t) (X))
#  define UINT8_C(X)   ((uint_least8_t) (X))
#  define UINT16_C(X)  ((uint_least16_t) (X))
#  define UINT32_C(X)  ((uint_least32_t) (X))
#  define UINT64_C(X)  ((uint_least64_t) (X))
#  define INTMAX_C(X)  ((intmax_t) (X))
#  define UINTMAX_C(X) ((uintmax_t) (X))
#endif // _CCCL_COMPILER_NVRTC

#endif // _LIBCUDACXX___CUDA_CSTDINT_PRELUDE_H
