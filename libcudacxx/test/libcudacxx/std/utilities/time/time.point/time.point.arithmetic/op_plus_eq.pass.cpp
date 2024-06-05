//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// time_point& operator+=(const duration& d);
// constexpr in c++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

#if TEST_STD_VER > 2014
__host__ __device__ constexpr bool constexpr_test()
{
  using Clock    = cuda::std::chrono::system_clock;
  using Duration = cuda::std::chrono::milliseconds;
  cuda::std::chrono::time_point<Clock, Duration> t(Duration(5));
  t += Duration(4);
  return t.time_since_epoch() == Duration(9);
}
#endif

int main(int, char**)
{
  {
    using Clock    = cuda::std::chrono::system_clock;
    using Duration = cuda::std::chrono::milliseconds;
    cuda::std::chrono::time_point<Clock, Duration> t(Duration(3));
    t += Duration(2);
    assert(t.time_since_epoch() == Duration(5));
  }

#if TEST_STD_VER > 2014
  static_assert(constexpr_test(), "");
#endif

  return 0;
}
