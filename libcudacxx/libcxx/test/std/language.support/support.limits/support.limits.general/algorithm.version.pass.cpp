//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// WARNING: This test was generated by generate_feature_test_macro_components.py
// and should not be edited manually.

// <algorithm>

// Test the feature test macros defined by <algorithm>

/*  Constant                                 Value
    __cpp_lib_clamp                          201603L [C++17]
    __cpp_lib_constexpr_swap_algorithms      201806L [C++2a]
    __cpp_lib_parallel_algorithm             201603L [C++17]
    __cpp_lib_ranges                         201811L [C++2a]
    __cpp_lib_robust_nonmodifying_seq_ops    201304L [C++14]
    __cpp_lib_sample                         201603L [C++17]
*/

#include <algorithm>
#include "test_macros.h"

#if TEST_STD_VER < 2014

# ifdef __cpp_lib_clamp
#   error "__cpp_lib_clamp should not be defined before c++17"
# endif

# ifdef __cpp_lib_constexpr_swap_algorithms
#   error "__cpp_lib_constexpr_swap_algorithms should not be defined before c++2a"
# endif

# ifdef __cpp_lib_parallel_algorithm
#   error "__cpp_lib_parallel_algorithm should not be defined before c++17"
# endif

# ifdef __cpp_lib_ranges
#   error "__cpp_lib_ranges should not be defined before c++2a"
# endif

# ifdef __cpp_lib_robust_nonmodifying_seq_ops
#   error "__cpp_lib_robust_nonmodifying_seq_ops should not be defined before c++14"
# endif

# ifdef __cpp_lib_sample
#   error "__cpp_lib_sample should not be defined before c++17"
# endif

#elif TEST_STD_VER == 2014

# ifdef __cpp_lib_clamp
#   error "__cpp_lib_clamp should not be defined before c++17"
# endif

# ifdef __cpp_lib_constexpr_swap_algorithms
#   error "__cpp_lib_constexpr_swap_algorithms should not be defined before c++2a"
# endif

# ifdef __cpp_lib_parallel_algorithm
#   error "__cpp_lib_parallel_algorithm should not be defined before c++17"
# endif

# ifdef __cpp_lib_ranges
#   error "__cpp_lib_ranges should not be defined before c++2a"
# endif

# ifndef __cpp_lib_robust_nonmodifying_seq_ops
#   error "__cpp_lib_robust_nonmodifying_seq_ops should be defined in c++14"
# endif
# if __cpp_lib_robust_nonmodifying_seq_ops != 201304L
#   error "__cpp_lib_robust_nonmodifying_seq_ops should have the value 201304L in c++14"
# endif

# ifdef __cpp_lib_sample
#   error "__cpp_lib_sample should not be defined before c++17"
# endif

#elif TEST_STD_VER == 2017

# ifndef __cpp_lib_clamp
#   error "__cpp_lib_clamp should be defined in c++17"
# endif
# if __cpp_lib_clamp != 201603L
#   error "__cpp_lib_clamp should have the value 201603L in c++17"
# endif

# ifdef __cpp_lib_constexpr_swap_algorithms
#   error "__cpp_lib_constexpr_swap_algorithms should not be defined before c++2a"
# endif

# if !defined(_LIBCUDACXX_VERSION)
#   ifndef __cpp_lib_parallel_algorithm
#     error "__cpp_lib_parallel_algorithm should be defined in c++17"
#   endif
#   if __cpp_lib_parallel_algorithm != 201603L
#     error "__cpp_lib_parallel_algorithm should have the value 201603L in c++17"
#   endif
# else // _LIBCUDACXX_VERSION
#   ifdef __cpp_lib_parallel_algorithm
#     error "__cpp_lib_parallel_algorithm should not be defined because it is unimplemented in libc++!"
#   endif
# endif

# ifdef __cpp_lib_ranges
#   error "__cpp_lib_ranges should not be defined before c++2a"
# endif

# ifndef __cpp_lib_robust_nonmodifying_seq_ops
#   error "__cpp_lib_robust_nonmodifying_seq_ops should be defined in c++17"
# endif
# if __cpp_lib_robust_nonmodifying_seq_ops != 201304L
#   error "__cpp_lib_robust_nonmodifying_seq_ops should have the value 201304L in c++17"
# endif

# ifndef __cpp_lib_sample
#   error "__cpp_lib_sample should be defined in c++17"
# endif
# if __cpp_lib_sample != 201603L
#   error "__cpp_lib_sample should have the value 201603L in c++17"
# endif

#elif TEST_STD_VER > 2017

# ifndef __cpp_lib_clamp
#   error "__cpp_lib_clamp should be defined in c++2a"
# endif
# if __cpp_lib_clamp != 201603L
#   error "__cpp_lib_clamp should have the value 201603L in c++2a"
# endif

# if !defined(_LIBCUDACXX_VERSION)
#   ifndef __cpp_lib_constexpr_swap_algorithms
#     error "__cpp_lib_constexpr_swap_algorithms should be defined in c++2a"
#   endif
#   if __cpp_lib_constexpr_swap_algorithms != 201806L
#     error "__cpp_lib_constexpr_swap_algorithms should have the value 201806L in c++2a"
#   endif
# else // _LIBCUDACXX_VERSION
#   ifdef __cpp_lib_constexpr_swap_algorithms
#     error "__cpp_lib_constexpr_swap_algorithms should not be defined because it is unimplemented in libc++!"
#   endif
# endif

# if !defined(_LIBCUDACXX_VERSION)
#   ifndef __cpp_lib_parallel_algorithm
#     error "__cpp_lib_parallel_algorithm should be defined in c++2a"
#   endif
#   if __cpp_lib_parallel_algorithm != 201603L
#     error "__cpp_lib_parallel_algorithm should have the value 201603L in c++2a"
#   endif
# else // _LIBCUDACXX_VERSION
#   ifdef __cpp_lib_parallel_algorithm
#     error "__cpp_lib_parallel_algorithm should not be defined because it is unimplemented in libc++!"
#   endif
# endif

# if !defined(_LIBCUDACXX_VERSION)
#   ifndef __cpp_lib_ranges
#     error "__cpp_lib_ranges should be defined in c++2a"
#   endif
#   if __cpp_lib_ranges != 201811L
#     error "__cpp_lib_ranges should have the value 201811L in c++2a"
#   endif
# else // _LIBCUDACXX_VERSION
#   ifdef __cpp_lib_ranges
#     error "__cpp_lib_ranges should not be defined because it is unimplemented in libc++!"
#   endif
# endif

# ifndef __cpp_lib_robust_nonmodifying_seq_ops
#   error "__cpp_lib_robust_nonmodifying_seq_ops should be defined in c++2a"
# endif
# if __cpp_lib_robust_nonmodifying_seq_ops != 201304L
#   error "__cpp_lib_robust_nonmodifying_seq_ops should have the value 201304L in c++2a"
# endif

# ifndef __cpp_lib_sample
#   error "__cpp_lib_sample should be defined in c++2a"
# endif
# if __cpp_lib_sample != 201603L
#   error "__cpp_lib_sample should have the value 201603L in c++2a"
# endif

#endif // TEST_STD_VER > 2017

int main(int, char**) { return 0; }
