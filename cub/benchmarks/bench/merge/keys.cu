// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_merge.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

#ifndef TUNE_BASE
#  define TUNE_THREADS_PER_BLOCK (1 << TUNE_THREADS_PER_BLOCK_POW2)

#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename KeyT>
struct policy_hub_t
{
  struct max_policy : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using merge_polcy =
      cub::agent_merge_no_sort_policy<TUNE_THREADS_PER_BLOCK,
                                      cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
                                      TUNE_LOAD_ALGORITHM,
                                      TUNE_LOAD_MODIFIER,
                                      TUNE_STORE_ALGORITHM>;
  };
};
#endif // TUNE_BASE

template <typename T, typename Offset>
void keys(nvbench::state& state, nvbench::type_list<T, Offset>)
{
  using compare_op_t = less_t;

  using dispatch_t = cub::detail::DispatchMerge<
    const T*,
    const cub::NullType*,
    const T*,
    const cub::NullType*,
    T*,
    cub::NullType*,
    Offset,
    compare_op_t
#if !TUNE_BASE
    ,
    policy_hub_t<T>
#endif // !TUNE_BASE
    >;

  // Retrieve axis parameters
  const auto size1          = static_cast<std::size_t>(state.get_int64("Elements{io}")) / 2;
  const auto size2          = size1;
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  const thrust::device_vector<T> input_buffer1 = generate(size1, entropy);
  const thrust::device_vector<T> input_buffer2 = generate(size2, entropy);
  thrust::device_vector<T> output_buffer(size1 + size2);

  const T* d_input1 = thrust::raw_pointer_cast(input_buffer1.data());
  const T* d_input2 = thrust::raw_pointer_cast(input_buffer2.data());
  T* d_output       = thrust::raw_pointer_cast(output_buffer.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(size1 + size2);
  state.add_global_memory_reads<T>(size1 + size2, "Size");
  state.add_global_memory_writes<T>(size1 + size2);

  std::size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_input1,
    nullptr,
    size1,
    d_input2,
    nullptr,
    size2,
    d_output,
    nullptr,
    compare_op_t{},
    cudaStream_t{0});

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      thrust::raw_pointer_cast(temp.data()),
      temp_size,
      d_input1,
      nullptr,
      size1,
      d_input2,
      nullptr,
      size2,
      d_output,
      nullptr,
      compare_op_t{},
      cudaStream_t{0});
  });
}

// TODO(bgruber): for offsets, use what thrust uses
NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(nvbench::type_list<int>, nvbench::type_list<int>))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
