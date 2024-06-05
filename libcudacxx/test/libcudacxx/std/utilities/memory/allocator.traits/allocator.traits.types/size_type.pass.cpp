//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     using size_type = Alloc::size_type | size_t;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct A
{
  using value_type = T;
  using size_type  = unsigned short;
};

template <class T>
struct B
{
  using value_type = T;
};

template <class T>
struct C
{
  using value_type = T;
  struct pointer
  {};
  struct const_pointer
  {};
  struct void_pointer
  {};
  struct const_void_pointer
  {};
};

#if !defined(TEST_COMPILER_MSVC_2017) // rebind is inaccessible
template <class T>
struct D
{
  using value_type      = T;
  using difference_type = short;

private:
  using size_type = void;
};
#endif // !TEST_COMPILER_MSVC_2017

namespace cuda
{
namespace std
{

template <>
struct pointer_traits<C<char>::pointer>
{
  using difference_type = signed char;
};

} // namespace std
} // namespace cuda

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::size_type, unsigned short>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<char>>::size_type,
                                    cuda::std::make_unsigned<cuda::std::ptrdiff_t>::type>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::size_type, unsigned char>::value), "");
#if !defined(TEST_COMPILER_MSVC_2017) // rebind is inaccessible
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<D<char>>::size_type, unsigned short>::value), "");
#endif // !TEST_COMPILER_MSVC_2017

  return 0;
}
