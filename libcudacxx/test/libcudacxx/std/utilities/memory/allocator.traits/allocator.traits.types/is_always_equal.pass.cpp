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
//   using is_always_equal = Alloc::is_always_equal | is_empty;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct A
{
  using value_type      = T;
  using is_always_equal = cuda::std::true_type;
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
  int not_empty_; // some random member variable
};

int main(int, char**)
{
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<A<char>>::is_always_equal, cuda::std::true_type>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<B<char>>::is_always_equal, cuda::std::true_type>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<C<char>>::is_always_equal, cuda::std::false_type>::value), "");

  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<A<const char>>::is_always_equal, cuda::std::true_type>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<B<const char>>::is_always_equal, cuda::std::true_type>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<C<const char>>::is_always_equal, cuda::std::false_type>::value),
    "");

  return 0;
}
