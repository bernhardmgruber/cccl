//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// back_insert_iterator

// Test nested types and data member:

// template <BackInsertionContainer Cont>
// class back_insert_iterator {
// protected:
//   Cont* container;
// public:
//   using container_type = Cont                       ;
//   using value_type = void                       ;
//   using difference_type = void                       ;
//   using reference = void                       ;
//   using pointer = void                       ;
// };

#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>

#  include "test_macros.h"

template <class C>
struct find_container : private cuda::std::back_insert_iterator<C>
{
  __host__ __device__ explicit find_container(C& c)
      : cuda::std::back_insert_iterator<C>(c)
  {}
  __host__ __device__ void test()
  {
    this->container = 0;
  }
};

template <class C>
__host__ __device__ void test()
{
  using R = cuda::std::back_insert_iterator<C>;
  C c;
  find_container<C> q(c);
  q.test();
  static_assert((cuda::std::is_same<typename R::container_type, C>::value), "");
  static_assert((cuda::std::is_same<typename R::value_type, void>::value), "");
  static_assert((cuda::std::is_same<typename R::difference_type, void>::value), "");
  static_assert((cuda::std::is_same<typename R::reference, void>::value), "");
  static_assert((cuda::std::is_same<typename R::pointer, void>::value), "");
  static_assert((cuda::std::is_same<typename R::iterator_category, cuda::std::output_iterator_tag>::value), "");
}

int main(int, char**)
{
  test<cuda::std::vector<int>>();

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
