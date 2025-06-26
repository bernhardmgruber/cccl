// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.Add commentMore actions
// SPDX-License-Identifier: BSD-3-Clause

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

#include "common.h"

// This benchmark tests overlapping memory regions for reading and is compute intensive

inline constexpr auto startA = 0.1;
inline constexpr auto startB = 0.2;

static void vectoradd(nvbench::state& state)
{
  const auto n = narrow<int32_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<__half> in1(n, startA);
  thrust::device_vector<__half> in2(n, startB);
  thrust::device_vector<__half> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<__half>(n);
  state.add_global_memory_reads<__half>(n);
  state.add_global_memory_writes<__half>(n);

  bench_transform(state, ::cuda::std::tuple{in1.begin(), in2.begin()}, out.begin(), n, ::cuda::std::plus<>{});
}

NVBENCH_BENCH(vectoradd).set_name("vectoradd").add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
