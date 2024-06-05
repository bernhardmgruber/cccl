//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// static constexpr time_point max(); // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  using Clock    = cuda::std::chrono::system_clock;
  using Duration = cuda::std::chrono::milliseconds;
  using TP       = cuda::std::chrono::time_point<Clock, Duration>;
  LIBCPP_ASSERT_NOEXCEPT(TP::max());
#if TEST_STD_VER > 2017
  ASSERT_NOEXCEPT(TP::max());
#endif
  assert(TP::max() == TP(Duration::max()));

  return 0;
}
