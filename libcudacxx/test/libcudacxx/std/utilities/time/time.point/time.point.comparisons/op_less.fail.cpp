//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator< (const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator> (const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator<=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator>=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// time_points with different clocks should not compare

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC

#include <cuda/std/chrono>

#include "../../clock.h"

int main(int, char**)
{
  using Clock1    = cuda::std::chrono::system_clock;
  using Clock2    = Clock;
  using Duration1 = cuda::std::chrono::milliseconds;
  using Duration2 = cuda::std::chrono::microseconds;
  using T1        = cuda::std::chrono::time_point<Clock1, Duration1>;
  using T2        = cuda::std::chrono::time_point<Clock2, Duration2>;

  T1 t1(Duration1(3));
  T2 t2(Duration2(3000));
  t1 < t2;

  return 0;
}
