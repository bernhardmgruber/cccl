// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_VEC_SIZE_POW vec 0:4:1
// %RANGE% TUNE_VECTORS vpt 1:8:1

#include <cuda/__numeric/narrow.h>

#include "common.h"

template <typename RandomAccessIteratorOut, typename... RandomAccessIteratorsIn>
#if TUNE_BASE
using policy_hub_vec_t =
  cub::detail::transform::policy_hub</* stable address */ false,
                                     /* dense output */ true,
                                     ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                     RandomAccessIteratorOut>;
#else
struct policy_hub_vec_t
{
  struct max_policy : cub::ChainedPolicy<500, max_policy, max_policy>
  {
    static constexpr int min_bif    = cub::detail::transform::arch_to_min_bytes_in_flight(__CUDA_ARCH_LIST__);
    static constexpr auto algorithm = cub::detail::transform::Algorithm::vectorized;

    struct tuning
    {
      static constexpr int block_threads    = TUNE_THREADS;
      static constexpr int vec_size         = 1 << TUNE_VEC_SIZE_POW;
      static constexpr int items_per_thread = vec_size * TUNE_VECTORS;
    };
    using algo_policy = cub::detail::transform::vectorized_policy_t<tuning>;
  };
};
#endif

template <typename T>
struct return_constant
{
  T value;

  _CCCL_DEVICE auto operator()() const -> T
  {
    return value;
  }
};

template <typename T>
static void fill(nvbench::state& state, nvbench::type_list<T>)
{
  // A 32-bit offset type or the value 0 or 0xFF... have <1% performance impact
  using offset_t   = int64_t;
  const auto value = T{42};
  const auto n     = cuda::narrow<offset_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(0);
  state.add_global_memory_writes<T>(n);

  bench_transform(
    state, ::cuda::std::tuple{}, out.begin(), n, return_constant<T>{value}, policy_hub_vec_t<decltype(out.begin())>{});
}

NVBENCH_BENCH_TYPES(fill, NVBENCH_TYPE_AXES(integral_types))
  .set_name("fill")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
