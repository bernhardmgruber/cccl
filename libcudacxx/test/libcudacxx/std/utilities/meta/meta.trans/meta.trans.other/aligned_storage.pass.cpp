//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// aligned_storage
//
//  Issue 3034 added:
//  The member alias type shall be a trivial standard-layout type.

#include <cuda/std/cstddef> // for cuda::std::max_align_t
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  {
    using T1 = cuda::std::aligned_storage<10, 1>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<10, 1>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 10, "");
  }
  {
    using T1 = cuda::std::aligned_storage<10, 2>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<10, 2>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
  }
  {
    using T1 = cuda::std::aligned_storage<10, 4>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<10, 4>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 12, "");
  }
  {
    using T1 = cuda::std::aligned_storage<10, 8>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<10, 8>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    using T1 = cuda::std::aligned_storage<10, 16>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<10, 16>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    using T1 = cuda::std::aligned_storage<10, 32>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<10, 32>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
  }
  {
    using T1 = cuda::std::aligned_storage<20, 32>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<20, 32>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
  }
  {
    using T1 = cuda::std::aligned_storage<40, 32>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<40, 32>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 64, "");
  }
  {
    using T1 = cuda::std::aligned_storage<12, 16>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<12, 16>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    using T1 = cuda::std::aligned_storage<1>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<1>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 1, "");
  }
  {
    using T1 = cuda::std::aligned_storage<2>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<2>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 2, "");
  }
  {
    using T1 = cuda::std::aligned_storage<3>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<3>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 4, "");
  }
  {
    using T1 = cuda::std::aligned_storage<4>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<4>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
  }
  {
    using T1 = cuda::std::aligned_storage<5>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<5>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
  }
  {
    using T1 = cuda::std::aligned_storage<7>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<7>);
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
  }
  {
    using T1 = cuda::std::aligned_storage<8>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<8>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 8, "");
  }
  {
    using T1 = cuda::std::aligned_storage<9>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<9>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    using T1 = cuda::std::aligned_storage<15>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<15>);
#endif
#if TEST_STD_VER <= 2017
    static_assert(cuda::std::is_pod<T1>::value, "");
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
  // Use alignof(cuda::std::max_align_t) below to find the max alignment instead of
  // hardcoding it, because it's different on different platforms.
  // (For example 8 on arm and 16 on x86.)
  {
    using T1 = cuda::std::aligned_storage<16>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<16>);
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == TEST_ALIGNOF(cuda::std::max_align_t), "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    using T1 = cuda::std::aligned_storage<17>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<17>);
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == TEST_ALIGNOF(cuda::std::max_align_t), "");
    static_assert(sizeof(T1) == 16 + TEST_ALIGNOF(cuda::std::max_align_t), "");
  }
  {
    using T1 = cuda::std::aligned_storage<10>::type;
#if TEST_STD_VER > 2011
    ASSERT_SAME_TYPE(T1, cuda::std::aligned_storage_t<10>);
#endif
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
// NVCC doesn't support types that are _this_ overaligned, it seems
#if !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_NVRTC)
  {
    const int Align = 65536;
    using T1        = typename cuda::std::aligned_storage<1, Align>::type;
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == Align, "");
    static_assert(sizeof(T1) == Align, "");
  }
#endif

  return 0;
}
