// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/cstddef>
#include <cuda/std/detail/libcxx/include/iosfwd>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char, class _Traits = char_traits<_CharT>>
class _LIBCUDACXX_TEMPLATE_VIS ostream_iterator
#if _CCCL_STD_VER <= 2014 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<output_iterator_tag, void, void, void, void>
#endif
{
  _CCCL_SUPPRESS_DEPRECATED_POP

public:
  using iterator_category = output_iterator_tag;
  using value_type        = void;
#if _CCCL_STD_VER > 2017
  using difference_type = ptrdiff_t;
#else
  using difference_type = void;
#endif
  using pointer      = void;
  using reference    = void;
  using char_type    = _CharT;
  using traits_type  = _Traits;
  using ostream_type = basic_ostream<_CharT, _Traits>;

private:
  ostream_type* __out_stream_;
  const char_type* __delim_;

public:
  _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator(ostream_type& __s) noexcept
      : __out_stream_(_CUDA_VSTD::addressof(__s))
      , __delim_(nullptr)
  {}
  _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator(ostream_type& __s, const _CharT* __delimiter) noexcept
      : __out_stream_(_CUDA_VSTD::addressof(__s))
      , __delim_(__delimiter)
  {}
  _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator=(const _Tp& __value)
  {
    *__out_stream_ << __value;
    if (__delim_)
    {
      *__out_stream_ << __delim_;
    }
    return *this;
  }

  _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator*()
  {
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator++()
  {
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator++(int)
  {
    return *this;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H
