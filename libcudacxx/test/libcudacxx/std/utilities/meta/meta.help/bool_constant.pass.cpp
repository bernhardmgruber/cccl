//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// bool_constant

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_STD_VER > 2011
  using _t = cuda::std::bool_constant<true>;
  static_assert(_t::value, "");
  static_assert((cuda::std::is_same<_t::value_type, bool>::value), "");
  static_assert((cuda::std::is_same<_t::type, _t>::value), "");
  static_assert((_t() == true), "");

  using _f = cuda::std::bool_constant<false>;
  static_assert(!_f::value, "");
  static_assert((cuda::std::is_same<_f::value_type, bool>::value), "");
  static_assert((cuda::std::is_same<_f::type, _f>::value), "");
  static_assert((_f() == false), "");
#endif

  return 0;
}
