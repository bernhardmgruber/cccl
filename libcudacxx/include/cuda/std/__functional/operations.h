// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H
#define _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/binary_function.h>
#include <cuda/std/__functional/unary_function.h>
#include <cuda/std/__utility/forward.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Arithmetic operations

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS plus : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x + __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(plus);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS plus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS minus : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x - __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(minus);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS minus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS multiplies : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x * __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(multiplies);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS multiplies<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS divides : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x / __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(divides);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS divides<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS modulus : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x % __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(modulus);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS modulus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS negate : __unary_function<_Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x) const
  {
    return -__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(negate);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS negate<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_Tp&& __x) const
    noexcept(noexcept(-_CUDA_VSTD::forward<_Tp>(__x))) -> decltype(-_CUDA_VSTD::forward<_Tp>(__x))
  {
    return -_CUDA_VSTD::forward<_Tp>(__x);
  }
  using is_transparent = void;
};

// Bitwise operations

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS bit_and : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x & __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_and);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS bit_and<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS bit_not : __unary_function<_Tp, _Tp>
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x) const
  {
    return ~__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_not);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS bit_not<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_Tp&& __x) const
    noexcept(noexcept(~_CUDA_VSTD::forward<_Tp>(__x))) -> decltype(~_CUDA_VSTD::forward<_Tp>(__x))
  {
    return ~_CUDA_VSTD::forward<_Tp>(__x);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS bit_or : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x | __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_or);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS bit_or<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS bit_xor : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x ^ __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_xor);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS bit_xor<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

// Comparison operations

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS equal_to : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x == __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(equal_to);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS equal_to<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS not_equal_to : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x != __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(not_equal_to);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS not_equal_to<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS less : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x < __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(less);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS less<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS less_equal : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x <= __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(less_equal);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS less_equal<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS greater_equal : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x >= __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(greater_equal);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS greater_equal<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS greater : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x > __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(greater);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS greater<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

// Logical operations

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS logical_and : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x && __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_and);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS logical_and<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS logical_not : __unary_function<_Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x) const
  {
    return !__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_not);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS logical_not<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_Tp&& __x) const
    noexcept(noexcept(!_CUDA_VSTD::forward<_Tp>(__x))) -> decltype(!_CUDA_VSTD::forward<_Tp>(__x))
  {
    return !_CUDA_VSTD::forward<_Tp>(__x);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _LIBCUDACXX_TEMPLATE_VIS logical_or : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x || __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_or);

template <>
struct _LIBCUDACXX_TEMPLATE_VIS logical_or<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H
