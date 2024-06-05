//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template<class Iter>
// struct iterator_traits
// {
//   using difference_type = typename Iter::difference_type;
//   using value_type = typename Iter::value_type;
//   using pointer = typename Iter::pointer;
//   using reference = typename Iter::reference;
//   using iterator_category = typename Iter::iterator_category;
// };

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif // TEST_COMPILER_MSVC

#if !defined(TEST_COMPILER_NVRTC)
#  include <vector>
#endif // !TEST_COMPILER_NVRTC

struct A
{};

struct test_iterator
{
  using difference_type   = int;
  using value_type        = A;
  using pointer           = A*;
  using reference         = A&;
  using iterator_category = cuda::std::forward_iterator_tag;
};

int main(int, char**)
{
  {
    using It = cuda::std::iterator_traits<test_iterator>;
    static_assert((cuda::std::is_same<It::difference_type, int>::value), "");
    static_assert((cuda::std::is_same<It::value_type, A>::value), "");
    static_assert((cuda::std::is_same<It::pointer, A*>::value), "");
    static_assert((cuda::std::is_same<It::reference, A&>::value), "");
    static_assert((cuda::std::is_same<It::iterator_category, cuda::std::forward_iterator_tag>::value), "");
  }

#if !defined(TEST_COMPILER_NVRTC)
  { // std::vector
    using It = cuda::std::iterator_traits<typename std::vector<int>::iterator>;
    static_assert((cuda::std::is_same<It::difference_type, std::ptrdiff_t>::value), "");
    static_assert((cuda::std::is_same<It::value_type, int>::value), "");
    static_assert((cuda::std::is_same<It::pointer, int*>::value), "");
    static_assert((cuda::std::is_same<It::reference, int&>::value), "");
    static_assert((cuda::std::is_same<It::iterator_category, std::random_access_iterator_tag>::value), "");

    static_assert(cuda::std::__is_cpp17_random_access_iterator<typename std::vector<int>::iterator>::value, "");
  }
#endif // !TEST_COMPILER_NVRTC

  return 0;
}
