// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/device/dispatch/dispatch_transform.cuh>

#include <nvbench_helper.cuh>

template <typename OffsetT,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub = cub::detail::transform::policy_hub</* stable address */ false,
                                                                  /* dense output */ true,
                                                                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                                                  RandomAccessIteratorOut>>
void bench_transform(
  nvbench::state& state,
  ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
  RandomAccessIteratorOut output,
  OffsetT num_items,
  TransformOp transform_op,
  PolicyHub = {})
{
  state.exec(nvbench::exec_tag::gpu, [&](const nvbench::launch& launch) {
    cub::detail::transform::dispatch_t<
      cub::detail::transform::requires_stable_address::no,
      OffsetT,
      ::cuda::std::tuple<RandomAccessIteratorsIn...>,
      RandomAccessIteratorOut,
      cub::detail::transform::always_true_predicate,
      TransformOp,
      PolicyHub>::dispatch(inputs,
                           output,
                           num_items,
                           cub::detail::transform::always_true_predicate{},
                           transform_op,
                           launch.get_stream());
  });
}
