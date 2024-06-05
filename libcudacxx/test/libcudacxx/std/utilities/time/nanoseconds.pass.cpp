//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// using nanoseconds = duration<signed integral type of at least 64 bits, nano>;

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

int main(int, char**)
{
  using D      = cuda::std::chrono::nanoseconds;
  using Rep    = D::rep;
  using Period = D::period;
  static_assert(cuda::std::is_signed<Rep>::value, "");
  static_assert(cuda::std::is_integral<Rep>::value, "");
  static_assert(cuda::std::numeric_limits<Rep>::digits >= 63, "");
  static_assert((cuda::std::is_same<Period, cuda::std::nano>::value), "");

  return 0;
}
