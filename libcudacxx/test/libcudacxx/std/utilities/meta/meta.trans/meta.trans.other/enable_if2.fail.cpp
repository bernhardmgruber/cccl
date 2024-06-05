//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// type_traits

// enable_if

#include <cuda/std/type_traits>

int main(int, char**)
{
  using A = cuda::std::enable_if_t<false>;

  return 0;
}
