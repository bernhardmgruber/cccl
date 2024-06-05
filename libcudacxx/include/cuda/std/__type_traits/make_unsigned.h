//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_MAKE_UNSIGNED_H
#define _LIBCUDACXX___TYPE_TRAITS_MAKE_UNSIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/apply_cv.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_MAKE_UNSIGNED) && !defined(_LIBCUDACXX_USE_MAKE_UNSIGNED_FALLBACK)

template <class _Tp>
using __make_unsigned_t = _LIBCUDACXX_MAKE_UNSIGNED(_Tp);

#else
using __unsigned_types = __type_list<
  unsigned char,
  __type_list<unsigned short,
              __type_list<unsigned int,
                          __type_list<unsigned long, __type_list<unsigned long long, __type_list<__uint128_t, __nat>>>>>>;

template <class _Tp, bool = is_integral<_Tp>::value || is_enum<_Tp>::value>
struct __make_unsigned_impl
{};

template <class _Tp>
struct __make_unsigned_impl<_Tp, true>
{
  using type = typename __find_first<__unsigned_types, sizeof(_Tp)>::type;
};

template <>
struct __make_unsigned_impl<bool, true>
{};
template <>
struct __make_unsigned_impl<signed short, true>
{
  using type = unsigned short;
};
template <>
struct __make_unsigned_impl<unsigned short, true>
{
  using type = unsigned short;
};
template <>
struct __make_unsigned_impl<signed int, true>
{
  using type = unsigned int;
};
template <>
struct __make_unsigned_impl<unsigned int, true>
{
  using type = unsigned int;
};
template <>
struct __make_unsigned_impl<signed long, true>
{
  using type = unsigned long;
};
template <>
struct __make_unsigned_impl<unsigned long, true>
{
  using type = unsigned long;
};
template <>
struct __make_unsigned_impl<signed long long, true>
{
  using type = unsigned long long;
};
template <>
struct __make_unsigned_impl<unsigned long long, true>
{
  using type = unsigned long long;
};
#  ifndef _LIBCUDACXX_HAS_NO_INT128
template <>
struct __make_unsigned_impl<__int128_t, true>
{
  using type = __uint128_t;
};
template <>
struct __make_unsigned_impl<__uint128_t, true>
{
  using type = __uint128_t;
};
#  endif

template <class _Tp>
using __make_unsigned_t = typename __apply_cv<_Tp, typename __make_unsigned_impl<__remove_cv_t<_Tp>>::type>::type;

#endif // defined(_LIBCUDACXX_MAKE_UNSIGNED) && !defined(_LIBCUDACXX_USE_MAKE_UNSIGNED_FALLBACK)

template <class _Tp>
struct make_unsigned
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __make_unsigned_t<_Tp>;
};

#if _CCCL_STD_VER > 2011
template <class _Tp>
using make_unsigned_t = __make_unsigned_t<_Tp>;
#endif

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __make_unsigned_t<_Tp>
__to_unsigned_like(_Tp __x) noexcept
{
  return static_cast<__make_unsigned_t<_Tp>>(__x);
}

template <class _Tp, class _Up>
using __copy_unsigned_t = __conditional_t<is_unsigned<_Tp>::value, __make_unsigned_t<_Up>, _Up>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_MAKE_UNSIGNED_H
