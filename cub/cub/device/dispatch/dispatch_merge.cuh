// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
// TODO(bgruber): revisit this selection logic for the fallback agent
template <typename DefaultPolicyT, class... Args>
class select_agent_with_policy
{
  using default_agent_t = agent_merge_no_sort<DefaultPolicyT, decltype(Args{})...>;

  using fallback_policy_t = policy_wrapper_t<DefaultPolicyT, 64, 1>; // 64 threads per block, 1 item per thread
  using fallback_agent_t  = agent_merge_no_sort<fallback_policy_t, Args...>;

  // Use fallback if the merge agent exceed the maximum shared memory available per block and both (1) the fallback
  // block sort and (2) the fallback merge agent would not exceed the available shared memory
  static constexpr bool use_fallback = sizeof(typename default_agent_t::TempStorage) > max_smem_per_block
                                    && sizeof(typename fallback_agent_t::TempStorage) <= max_smem_per_block;

public:
  using policy_t = ::cuda::std::__conditional_t<use_fallback, fallback_policy_t, DefaultPolicyT>;
  using agent_t  = ::cuda::std::__conditional_t<use_fallback, fallback_agent_t, default_agent_t>;
};

template <typename KeysIt1, typename KeysIt2, typename Offset, typename CompareOp, int items_per_tile>
CUB_DETAIL_KERNEL_ATTRIBUTES void partition_merge_path_kernel(
  KeysIt1 keys1,
  Offset keys1_count,
  KeysIt2 keys2,
  Offset keys2_count,
  Offset num_partitions,
  Offset* merge_partitions,
  CompareOp compare_op)
{
  const Offset partition_idx = blockDim.x * blockIdx.x + threadIdx.x; // TODO(bgruber): can this be an int?
  if (partition_idx < num_partitions)
  {
    AgentPartitionMergePath<KeysIt1, KeysIt2, Offset, CompareOp, items_per_tile>{
      keys1, keys1_count, keys2, keys2_count, merge_partitions, compare_op, partition_idx}();
  }
}

template <typename Policy,
          typename Agent,
          typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp>
__launch_bounds__(Policy::threads_per_block) CUB_DETAIL_KERNEL_ATTRIBUTES void merge_kernel(
  KeyIt1 keys1,
  ValueIt1 items1,
  Offset num_keys1,
  KeyIt2 keys2,
  ValueIt2 items2,
  Offset num_keys2,
  KeyIt3 keys_result,
  ValueIt3 items_result,
  CompareOp compare_op,
  Offset* merge_partitions,
  vsmem_t global_temp_storage)
{
  using vsmem_helper_t = vsmem_helper_impl<Agent>;
  __shared__ typename vsmem_helper_t::static_temp_storage_t shared_temp_storage;
  auto& temp_storage = vsmem_helper_t::get_temp_storage(shared_temp_storage, global_temp_storage);
  Agent{temp_storage.Alias(),
        THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(Policy{}, keys1),
        THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(Policy{}, items1),
        num_keys1,
        THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(Policy{}, keys2),
        THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(Policy{}, items2),
        num_keys2,
        keys_result,
        items_result,
        compare_op,
        merge_partitions}();
  vsmem_helper_t::discard_temp_storage(temp_storage);
}

template <typename KeyT, typename ValueT>
struct device_merge_policy
{
  static constexpr bool has_values = !::cuda::std::is_same<ValueT, NullType>::value;
  using tune_type                  = char[has_values ? sizeof(KeyT) + sizeof(ValueT) : sizeof(KeyT)];

  struct policy300 : ChainedPolicy<350, policy300, policy300>
  {
    using merge_policy =
      agent_merge_no_sort_policy<128,
                                 Nominal4BItemsToItems<tune_type>(7), // TODO(bgruber): merge sort had 11
                                 BLOCK_LOAD_WARP_TRANSPOSE,
                                 LOAD_DEFAULT, // TODO(bgruber): merge sort had LOAD_LDG
                                 BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy350 : ChainedPolicy<520, policy350, policy300>
  {
    using merge_policy =
      agent_merge_no_sort_policy<256,
                                 Nominal4BItemsToItems<KeyT>(11), // TODO(bgruber): merge sort had 13
                                 BLOCK_LOAD_WARP_TRANSPOSE,
                                 LOAD_LDG,
                                 BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy520 : ChainedPolicy<520, policy520, policy350>
  {
    using merge_policy =
      agent_merge_no_sort_policy<512,
                                 Nominal4BItemsToItems<KeyT>(13), // TODO(bgruber): merge sort had 15
                                 BLOCK_LOAD_WARP_TRANSPOSE,
                                 LOAD_LDG,
                                 BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy600 : ChainedPolicy<600, policy600, policy520>
  {
    using merge_policy =
      agent_merge_no_sort_policy<512, // // TODO(bgruber): merge sort had 256
                                 Nominal4BItemsToItems<KeyT>(15), // TODO(bgruber): merge sort had 17
                                 BLOCK_LOAD_WARP_TRANSPOSE,
                                 LOAD_DEFAULT,
                                 BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using max_policy = policy600;
};

template <typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp,
          typename SelectedPolicy = device_merge_policy<value_t<KeyIt1>, value_t<ValueIt1>>>
struct DispatchMerge
{
  using key_t   = cub::detail::value_t<KeyIt1>;
  using value_t = cub::detail::value_t<ValueIt1>;

  // Cannot check output iterators, since they could be discard iterators, which do not have the right value_type
  static_assert(::cuda::std::is_same<cub::detail::value_t<KeyIt2>, key_t>::value, "");
  static_assert(::cuda::std::is_same<cub::detail::value_t<ValueIt2>, value_t>::value, "");
  static_assert(::cuda::std::__invokable<CompareOp, key_t, key_t>::value,
                "Comparison operator cannot compare two keys");
  static_assert(
    ::cuda::std::is_convertible<typename ::cuda::std::__invoke_of<CompareOp, key_t, key_t>::type, bool>::value,
    "Comparison operator must be convertible to bool");

  void* d_temp_storage;
  std::size_t& temp_storage_bytes;
  KeyIt1 d_keys1;
  ValueIt1 d_values1;
  Offset num_items1;
  KeyIt2 d_keys2;
  ValueIt2 d_values2;
  Offset num_items2;
  KeyIt3 d_keys_out;
  ValueIt3 d_values_out;
  CompareOp compare_op;
  cudaStream_t stream;

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using merge_policy_t = typename ActivePolicy::merge_policy;
    using selected_agent_and_policy =
      select_agent_with_policy<merge_policy_t, KeyIt1, ValueIt1, KeyIt2, ValueIt2, KeyIt3, ValueIt3, Offset, CompareOp>;
    using policy_t = typename selected_agent_and_policy::policy_t;
    using agent_t  = typename selected_agent_and_policy::agent_t;

    constexpr int tile_size = policy_t::items_per_tile;
    const auto num_tiles    = cub::DivideAndRoundUp(num_items1 + num_items2, tile_size);
    void* allocations[2]    = {nullptr, nullptr};
    {
      const std::size_t merge_partitions_size      = (1 + num_tiles) * sizeof(Offset);
      const std::size_t virtual_shared_memory_size = num_tiles * vsmem_helper_impl<agent_t>::vsmem_per_block;
      const std::size_t allocation_sizes[2]        = {merge_partitions_size, virtual_shared_memory_size};
      const auto error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    // Return if only temporary storage was requested or there is no work to be done
    if (d_temp_storage == nullptr || num_tiles == 0)
    {
      return cudaSuccess;
    }

    auto merge_partitions = static_cast<Offset*>(allocations[0]);

    // parition the merge path
    {
      const Offset num_partitions               = num_tiles + 1;
      constexpr int threads_per_partition_block = 256; // TODO(bgruber): no policy?
      const int partition_grid_size =
        static_cast<int>(cub::DivideAndRoundUp(num_partitions, threads_per_partition_block));

      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        partition_grid_size, threads_per_partition_block, 0, stream)
        .doit(partition_merge_path_kernel<KeyIt1, KeyIt2, Offset, CompareOp, tile_size>,
              d_keys1,
              num_items1,
              d_keys2,
              num_items2,
              num_partitions,
              merge_partitions,
              compare_op);
      const auto error = CubDebug(DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    // merge
    if (num_tiles > 0)
    {
      auto vshmem_ptr = vsmem_t{allocations[1]};
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        static_cast<int>(num_tiles), static_cast<int>(policy_t::threads_per_block), 0, stream)
        .doit(merge_kernel<policy_t, agent_t, KeyIt1, ValueIt1, KeyIt2, ValueIt2, KeyIt3, ValueIt3, Offset, CompareOp>,
              d_keys1,
              d_values1,
              num_items1,
              d_keys2,
              d_values2,
              num_items2,
              d_keys_out,
              d_values_out,
              compare_op,
              merge_partitions,
              vshmem_ptr);
      const auto error = CubDebug(DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    return cudaSuccess;
  }

  template <typename... Args>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(Args&&... args)
  {
    int ptx_version = 0;
    auto error      = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }
    DispatchMerge dispatch{std::forward<Args>(args)...};
    error = CubDebug(SelectedPolicy::max_policy::Invoke(ptx_version, dispatch));
    if (cudaSuccess != error)
    {
      return error;
    }

    return cudaSuccess;
  }
};

} // namespace detail
CUB_NAMESPACE_END
