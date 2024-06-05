//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, tuple&&);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "MoveOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    using T = cuda::std::tuple<>;
    T t0;
    T t(cuda::std::allocator_arg, A1<int>(), cuda::std::move(t0));
  }
  {
    using T = cuda::std::tuple<MoveOnly>;
    T t0(MoveOnly(0));
    T t(cuda::std::allocator_arg, A1<int>(), cuda::std::move(t0));
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    using T = cuda::std::tuple<alloc_first>;
    T t0(1);
    alloc_first::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t) == 1);
  }
  {
    using T = cuda::std::tuple<alloc_last>;
    T t0(1);
    alloc_last::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 1);
  }
// testing extensions
#ifdef _LIBCUDACXX_VERSION
  {
    using T = cuda::std::tuple<MoveOnly, alloc_first>;
    T t0(0, 1);
    alloc_first::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    using T = cuda::std::tuple<MoveOnly, alloc_first, alloc_last>;
    T t0(1, 2, 3);
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T t(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 2);
    assert(cuda::std::get<2>(t) == 3);
  }
#endif

  return 0;
}
