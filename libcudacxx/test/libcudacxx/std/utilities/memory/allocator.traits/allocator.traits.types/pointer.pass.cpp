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
//     using pointer = Alloc::pointer | value_type*;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct Ptr
{};

template <class T>
struct A
{
  using value_type = T;
  using pointer    = Ptr<T>;
};

template <class T>
struct B
{
  using value_type = T;
};

#if !defined(TEST_COMPILER_MSVC_2017) // pointer is inaccessible
template <class T>
struct C
{
  using value_type = T;

private:
  using pointer = void;
};
#endif // !TEST_COMPILER_MSVC_2017

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::pointer, Ptr<char>>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<char>>::pointer, char*>::value), "");
#if !defined(TEST_COMPILER_MSVC_2017) // pointer is inaccessible
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::pointer, char*>::value), "");
#endif // !TEST_COMPILER_MSVC_2017

  return 0;
}
