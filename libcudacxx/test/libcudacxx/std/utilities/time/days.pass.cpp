//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>

// using days = duration<signed integer type of at least 25 bits, ratio_multiply<ratio<24>, hours::period>>;

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using D      = cuda::std::chrono::days;
  using Rep    = D::rep;
  using Period = D::period;
  static_assert(cuda::std::is_signed<Rep>::value, "");
  static_assert(cuda::std::is_integral<Rep>::value, "");
  static_assert(cuda::std::numeric_limits<Rep>::digits >= 25, "");
  static_assert(
    cuda::std::is_same_v<Period, cuda::std::ratio_multiply<cuda::std::ratio<24>, cuda::std::chrono::hours::period>>,
    "");

  return 0;
}
