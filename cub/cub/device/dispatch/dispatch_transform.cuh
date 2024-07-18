// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda.h>

#include <cub/device/device_for.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/array>
#include <cuda/std/detail/libcxx/include/bit>
#include <cuda/std/expected>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace transform
{
#ifdef __cpp_fold_expressions // C++17
template <typename... Its>
constexpr auto loaded_bytes_per_iteration(::cuda::std::tuple<Its...>*) -> std::size_t
{
  return (sizeof(value_t<Its>) + ... + 0);
}
#else
constexpr std::size_t sum()
{
  return 0;
}

template <typename... Ts>
constexpr std::size_t sum(std::size_t head, Ts... tail)
{
  return head + sum(tail...);
}

template <typename... Its>
constexpr auto loaded_bytes_per_iteration(::cuda::std::tuple<Its...>*) -> std::size_t
{
  return sum(sizeof(value_t<Its>)...);
}
#endif

template <int BlockThreads>
struct prefetch_policy_t
{
  static constexpr int BLOCK_THREADS = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int MIN_ITEMS_PER_THREAD = 1;
  static constexpr int MAX_ITEMS_PER_THREAD = 32;
};

// prefetching out-of-bounds addresses has no side effects
inline _CCCL_DEVICE void prefetch(const void* addr)
{
  // TODO(bgruber): prefetch to L1 may be even better
  asm volatile("prefetch.L2 [%0];" : : "l"(addr) : "memory");
}

// this kernel guarantees stable addresses for the parameters of the user provided function
template <typename Offset, typename F, typename RandomAccessIteartorOut, typename... InTs>
CUB_DETAIL_KERNEL_ATTRIBUTES void transform_prefetch_kernel(
  Offset len, int num_elem_per_thread, F f, RandomAccessIteartorOut out, const InTs*... pointers)
{
  const int tile_size = blockDim.x * num_elem_per_thread;
  const Offset offset = static_cast<Offset>(blockIdx.x) * tile_size;

  for (int j = 0; j < num_elem_per_thread; ++j)
  {
    // TODO(bgruber): replace by fold over comma in C++17
    const auto idx = offset + (j * blockDim.x + threadIdx.x);
    int dummy[]    = {(prefetch(pointers + offset), 0)..., 0}; // extra zero to handle empty packs
    (void) dummy;
  }

  // ahendriksen: various unrolling yields less <1% gains at much higher compile-time cost
  // TODO(bgruber): A6000 disagrees
#pragma unroll 1
  for (int j = 0; j < num_elem_per_thread; ++j)
  {
    const auto idx = offset + (j * blockDim.x + threadIdx.x);
    if (idx < len)
    {
      out[idx] = f(*(pointers + idx)...);
    }
  }
}

template <int BlockThreads, int ItemsPerThread>
struct unrolled_policy_t
{
  static constexpr int BLOCK_THREADS    = BlockThreads;
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
};

// ahendriksen: no __restrict__ should be necessary on the input pointers since we already separated the load stage from
// the store stage.
template <typename MaxPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteartorOut,
          typename... RandomAccessIteartorIn>
__launch_bounds__(MaxPolicy::ActivePolicy::unrolled_policy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void transform_unrolled_staged_kernel(
    Offset len, F f, RandomAccessIteartorOut out, RandomAccessIteartorIn... ins)
{
  constexpr int block_dim        = MaxPolicy::ActivePolicy::unrolled_policy::BLOCK_THREADS;
  constexpr int items_per_thread = MaxPolicy::ActivePolicy::unrolled_policy::ITEMS_PER_THREAD;
  constexpr int tile_size        = block_dim * items_per_thread;
  const Offset offset            = static_cast<Offset>(blockIdx.x) * tile_size;

  [&](cuda::std::array<value_t<RandomAccessIteartorIn>, items_per_thread>&&... arrays) {
  // load items_per_thread elements
#pragma unroll
    for (int j = 0; j < items_per_thread; ++j)
    {
      const auto idx = offset + (j * block_dim + threadIdx.x);
      if (idx < len)
      {
        // TODO(bgruber): replace by fold over comma in C++17
        int dummy[] = {(arrays[j] = ins[idx], 0)..., 0}; // extra zero to handle empty packs
        (void) dummy;
      }
    }
    // process items_per_thread elements
#pragma unroll
    for (int j = 0; j < items_per_thread; ++j)
    {
      const auto idx = offset + (j * block_dim + threadIdx.x);
      if (idx < len)
      {
        out[idx] = f(arrays[j]...);
      }
    }
  }(cuda::std::array<value_t<RandomAccessIteartorIn>, items_per_thread>{}...);
}

// TODO(bgruber) cheap copy of ::cuda::std::apply, which requires C++17.
template <class F, class Tuple, std::size_t... Is>
_CCCL_DEVICE auto poor_apply_impl(F&& f, Tuple&& t, ::cuda::std::index_sequence<Is...>)
  -> decltype(std::forward<F>(f)(::cuda::std::get<Is>(std::forward<Tuple>(t))...))
{
  return std::forward<F>(f)(::cuda::std::get<Is>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
_CCCL_DEVICE auto poor_apply(F&& f, Tuple&& t)
  -> decltype(poor_apply_impl(
    std::forward<F>(f),
    std::forward<Tuple>(t),
    ::cuda::std::make_index_sequence<::cuda::std::tuple_size<::cuda::std::__libcpp_remove_reference_t<Tuple>>::value>{}))
{
  return poor_apply_impl(
    std::forward<F>(f),
    std::forward<Tuple>(t),
    ::cuda::std::make_index_sequence<::cuda::std::tuple_size<::cuda::std::__libcpp_remove_reference_t<Tuple>>::value>{});
}

// mult must be a power of 2
_CCCL_HOST_DEVICE inline auto round_up_to_multiple(int x, int mult) -> int
{
  const auto p = cuda::std::popcount(static_cast<unsigned>(mult));
  assert(p == 1);
  return (x + mult - 1) & ~(mult - 1);
}

// TODO(bgruber): inline this as lambda in C++14
template <typename T>
_CCCL_DEVICE T* copy_and_return_smem_dst(
  cooperative_groups::thread_block& group, int tile_size, char* smem, int& smem_offset, int global_offset, const T* ptr)
{
  // using T          = ::cuda::std::__remove_const_t<::cuda::std::__remove_pointer_t<decltype(ptr)>>;
  const auto count = static_cast<uint32_t>(sizeof(T)) * tile_size;
  smem_offset      = round_up_to_multiple(smem_offset, alignof(T));
  auto smem_dst    = reinterpret_cast<T*>(smem + smem_offset);
  cooperative_groups::memcpy_async(group, smem_dst, ptr + global_offset, count);
  smem_offset += count;
  return smem_dst;
}

template <typename Offset, typename F, typename RandomAccessIteartorOut, typename... InTs>
CUB_DETAIL_KERNEL_ATTRIBUTES void transform_memcpy_async_kernel(
  Offset len, int num_elem_per_thread, F f, RandomAccessIteartorOut out, const InTs*... pointers)
{
  extern __shared__ char smem[];

  const Offset tile_stride   = blockDim.x * num_elem_per_thread;
  const Offset global_offset = std::size_t{blockIdx.x} * tile_stride;
  const int tile_size        = ::cuda::std::min(len - global_offset, tile_stride);

  auto group = cooperative_groups::this_thread_block();

  // TODO(bgruber): if we pass block size as template parameter, we could compute the smem offsets at compile time
  int smem_offset      = 0;
  const auto smem_ptrs = ::cuda::std::tuple<InTs*...>{
    copy_and_return_smem_dst(group, tile_size, smem, smem_offset, global_offset, pointers)...};
  cooperative_groups::wait(group);

#pragma unroll 1
  for (int i = 0; i < num_elem_per_thread; ++i)
  {
    const int smem_idx    = i * blockDim.x + threadIdx.x;
    const Offset gmem_idx = global_offset + smem_idx;
    if (gmem_idx < len)
    {
      out[gmem_idx] = poor_apply(
        [&](const InTs*... smem_base_ptrs) {
          return f(smem_base_ptrs[smem_idx]...);
        },
        smem_ptrs);
    }
  }
}

_CCCL_DEVICE inline bool elect_sync(const std::uint32_t& membermask)
{
  std::uint32_t is_elected;
  asm volatile(
    "{\n\t .reg .pred P_OUT; \n\t"
    "elect.sync _|P_OUT, %1;\n\t"
    "selp.b32 %0, 1, 0, P_OUT; \n"
    "}"
    : "=r"(is_elected)
    : "r"(membermask)
    :);
  return static_cast<bool>(is_elected);
}

_CCCL_HOST_DEVICE constexpr uint32_t round_up_16(std::uint32_t x)
{
  return (x + 15) & ~15;
}

template <typename T>
_CCCL_HOST_DEVICE T* round_down_ptr_128(const T* ptr)
{
  constexpr auto mask = ~std::uintptr_t{128 - 1};
  return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(ptr) & mask);
}

template <typename T>
_CCCL_HOST_DEVICE int offset_to_aligned_ptr_128(const T* ptr)
{
  return static_cast<int>((reinterpret_cast<std::uintptr_t>(ptr) & std::uintptr_t{128 - 1}) / sizeof(T));
}

template <int /*align_bytes*/, typename T>
struct ptr_set
{
  using data_type = T;
  T* base_ptr; // align_bytes-byte aligned base pointer
  T* ptr; // original pointer
  uint32_t over_copy; // number of extra bytes to copy to cover a tile of ptr, i.e. number such that:
  // ptr[idx ... idx + tile_size] ⊆ ptr_base[idx ... idx + tile_size + over_copy]
  // over_copy < align_bytes
  uint32_t base_offset;
  // Offset in number of bytes of T in ptr in ptr_base, i.e. ptr == &ptr_base[base_offset]
};

template <typename T>
_CCCL_HOST_DEVICE ptr_set<128, T> make_ublkcp_ptr_set(T* ptr)
{
  return ptr_set<128, T>{
    round_down_ptr_128(ptr),
    ptr,
    round_up_16(offset_to_aligned_ptr_128(ptr) * sizeof(T)),
    // (ptr - ptr_base) * sizeof(T) rounded up to nearest multiple of 16
    static_cast<uint32_t>(sizeof(T) * (ptr - round_down_ptr_128(ptr))),
  };
}

// used for both, memcpy_async and ublkcp kernels
template <int BlockThreads>
struct async_copy_policy_t
{
  static constexpr int BLOCK_THREADS = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int MIN_ITEMS_PER_THREAD = 1;
  static constexpr int MAX_ITEMS_PER_THREAD = 32;
};

// TODO(bgruber): inline this as lambda in C++14
template <int A, typename T>
_CCCL_DEVICE void copy_ptr_set(
  uint64_t& bar,
  uint32_t tile_size,
  size_t tile_stride,
  char* smem,
  int& smem_offset,
  uint32_t& total_copied,
  size_t global_offset,
  const ptr_set<A, T>& ptr_set)
{
#if CUB_PTX_ARCH >= 900
  // Copy a bit more than tile_size, to cover for base_ptr starting earlier than ptr
  const uint32_t num_bytes = round_up_16(sizeof(T) * tile_size + ptr_set.over_copy);
  ptx::cp_async_bulk(
    ptx::space_cluster,
    ptx::space_global,
    smem + smem_offset,
    ptr_set.base_ptr + global_offset, // Use 128-byte aligned base_ptr here
    num_bytes,
    &bar);
  smem_offset += sizeof(T) * tile_stride + 128;
  total_copied += num_bytes;
#endif
};

// TODO(bgruber): inline this as lambda in C++14
template <int A, typename T>
const T*
fetch_operand(uint32_t tile_stride, const char* smem, int& smem_offset, int smem_idx, const ptr_set<A, T>& ptr_set)
{
  const T* smem_operand_tile_base = reinterpret_cast<const T*>(smem + smem_offset + ptr_set.base_offset);
  smem_offset += sizeof(T) * tile_stride + 128;
  return smem_operand_tile_base[smem_idx];
};

// TODO(bgruber): apply Offset type
template <typename Offset, typename F, typename RandomAccessIteartorOut, typename... InTs>
CUB_DETAIL_KERNEL_ATTRIBUTES void transform_ublkcp_kernel(
  Offset len, int num_elem_per_thread, F f, RandomAccessIteartorOut out, ptr_set<128, const InTs>... pointers)
{
#if CUB_PTX_ARCH >= 900
  __shared__ uint64_t bar;
  extern __shared__ char __attribute((aligned(128))) smem[];

  namespace ptx = cuda::ptx;

  const int tile_stride      = blockDim.x * num_elem_per_thread;
  const Offset global_offset = static_cast<Offset>(blockIdx.x) * tile_stride;

  const bool elected = elect_sync(~0);

  if (threadIdx.x < 32 && elected)
  {
    // Then initialize barriers
    ptx::mbarrier_init(&bar, 1);
    ptx::fence_proxy_async(ptx::space_shared);

    // Compute tile_size (relevant if processing last tile to not read out-of-bounds)
    auto tile_size = tile_stride;
    if (len < global_offset + tile_stride)
    {
      tile_size = len - global_offset;
    }
    int smem_offset            = 0;
    std::uint32_t total_copied = 0;

#  ifdef __cpp_fold_expressions // C++17
    // Order of evaluation is always left-to-right here. So smem_offset is updated in the right order.
    (..., copy_ptr_set(bar, tile_size, tile_stride, smem, smem_offset, total_copied, global_offset, pointers));
#  else
    // Order of evaluation is also left-to-right
    int dummy[] = {
      (copy_ptr_set(bar, tile_size, tile_stride, smem, smem_offset, total_copied, global_offset, pointers), 0)..., 0};
    (void) dummy;
#  endif

    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, total_copied);
  }
  __syncthreads();

  while (!ptx::mbarrier_try_wait_parity(&bar, 0))
  {
  }
  // Intentionally use unroll 1. This tends to improve performance.
#  pragma unroll 1
  for (int j = 0; j < num_elem_per_thread; ++j)
  {
    // ahendriksen: We do not define g_idx in terms of smem_idx. Doing this results in sub-optimal codegen. The common
    // sub-expression elimination logic is smart enough to remove the redundant computations.
    const int smem_idx = j * blockDim.x + threadIdx.x;
    const Offset g_idx = global_offset + j * blockDim.x + threadIdx.x;
    if (g_idx < len)
    {
      int smem_offset = 0;
      out[g_idx]      = f(fetch_operand(tile_stride, smem, smem_offset, smem_idx, pointers)...);
    }
  }
#endif // CUB_PTX_ARCH >= 900
}

enum class Algorithm
{
  fallback_for,
  prefetch,
  unrolled_staged,
  memcpy_async,
  ublkcp
};

constexpr int arch_to_min_bif(int sm_arch)
{
  // TODO(bgruber): use if-else in C++14 for better readability
  return sm_arch >= 900 ? 48 * 1024 // 32 for H100, 48 for H200
       : sm_arch >= 800 ? 16 * 1024 // A100
                        : 12 * 1024; // V100 and below
}

constexpr int
items_per_thread_from_occupancy(int block_dim, int max_block_per_sm, int min_bif, int loaded_bytes_per_iter)
{
#if _CCCL_STD_VER >= 2014
  return ::cuda::ceil_div(min_bif, max_block_per_sm * block_dim * loaded_bytes_per_iter);
#else
  return (min_bif + (max_block_per_sm * block_dim * loaded_bytes_per_iter) - 1)
       / (max_block_per_sm * block_dim * loaded_bytes_per_iter);
#endif
}

template <typename RandomAccessIteratorTupleIn>
struct policy_hub
{
  // TODO(gevtushenko): take a look at radix sort dispatch

  static constexpr int loaded_bytes_per_iter = ::cuda::std::max(
    1, static_cast<int>(loaded_bytes_per_iteration(static_cast<RandomAccessIteratorTupleIn*>(nullptr))));

  // below A100
  struct policy300 : ChainedPolicy<300, policy300, policy300>
  {
    static constexpr int min_bif                 = arch_to_min_bif(300);
    static constexpr Algorithm alg_addr_unstable = Algorithm::unrolled_staged;
    static constexpr Algorithm alg_addr_stable   = Algorithm::prefetch;
    using prefetch_policy                        = prefetch_policy_t<256>;
    using unrolled_policy =
      unrolled_policy_t<256, items_per_thread_from_occupancy(256, 8, min_bif, loaded_bytes_per_iter)>;
    using ublkcp_policy = async_copy_policy_t<256>;
  };

  // TODO(bgruber): should we add a tuning for 750? They should have items_per_thread_from_occupancy(256, 4, ...)

  // A100
  struct policy800 : ChainedPolicy<800, policy800, policy300>
  {
    static constexpr int min_bif                 = arch_to_min_bif(800);
    static constexpr Algorithm alg_addr_unstable = Algorithm::memcpy_async;
    static constexpr Algorithm alg_addr_stable   = Algorithm::prefetch;
    using prefetch_policy                        = prefetch_policy_t<256>;
    using unrolled_policy =
      unrolled_policy_t<256, items_per_thread_from_occupancy(256, 8, min_bif, loaded_bytes_per_iter)>;
    using ublkcp_policy = async_copy_policy_t<256>;
  };

  // TODO(bgruber): should we add a tuning for 860? They should have items_per_thread_from_occupancy(256, 6, ...)

  // H100 and H200
  struct policy900 : ChainedPolicy<900, policy900, policy800>
  {
    static constexpr int min_bif                 = arch_to_min_bif(900);
    static constexpr Algorithm alg_addr_unstable = Algorithm::ublkcp;
    static constexpr Algorithm alg_addr_stable   = Algorithm::prefetch;
    using prefetch_policy                        = prefetch_policy_t<256>;
    using unrolled_policy =
      unrolled_policy_t<256, items_per_thread_from_occupancy(256, 8, min_bif, loaded_bytes_per_iter)>;
    using ublkcp_policy = async_copy_policy_t<256>;
  };

  using max_policy = policy900;
};

// template <typename T>
// struct assert_trivially_relocatable : THRUST_NS_QUALIFIER::is_trivially_relocatable<T>
// {
//   static_assert(THRUST_NS_QUALIFIER::is_trivially_relocatable<T>::value,
//                 "If transform is allowed to copy, T needs to be trivially relocatable");
// };

// TODO(bgruber): replace by ::cuda::std::expected in C++14
template <typename T>
struct PoorExpected
{
  alignas(T) char storage[sizeof(T)];
  cudaError_t error;

  _CCCL_HOST_DEVICE PoorExpected(T value)
      : error(cudaSuccess)
  {
    new (storage) T(::cuda::std::move(value));
  }

  _CCCL_HOST_DEVICE PoorExpected(cudaError_t error)
      : error(error)
  {}

  _CCCL_HOST_DEVICE explicit operator bool() const
  {
    return error == cudaSuccess;
  }

  _CCCL_HOST_DEVICE T& operator*()
  {
    return reinterpret_cast<T&>(storage);
  }
};

template <bool CanCopyArgsForOp,
          typename Offset,
          typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub = policy_hub<RandomAccessIteratorTupleIn>>
struct dispatch_t;

template <bool CanCopyArgsForOp,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub>
struct dispatch_t<CanCopyArgsForOp,
                  Offset,
                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                  RandomAccessIteratorOut,
                  TransformOp,
                  PolicyHub>
{
  Offset num_items;
  ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
  RandomAccessIteratorOut out;
  TransformOp op;
  cudaStream_t stream;
  int max_smem;

  static constexpr int loaded_bytes_per_iter =
    static_cast<int>(loaded_bytes_per_iteration(static_cast<::cuda::std::tuple<RandomAccessIteratorsIn...>*>(nullptr)));

  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running on?
  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto configure_memcpy_async_kernel(cuda::std::index_sequence<Is...>)
    -> PoorExpected<::cuda::std::tuple<THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron,
                                       decltype(&transform_memcpy_async_kernel<Offset,
                                                                               TransformOp,
                                                                               RandomAccessIteratorOut,
                                                                               value_t<RandomAccessIteratorsIn>...>),
                                       int>>
  {
    using policy_t          = typename ActivePolicy::ublkcp_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;

    auto kernel =
      transform_memcpy_async_kernel<Offset, TransformOp, RandomAccessIteratorOut, value_t<RandomAccessIteratorsIn>...>;

    // Increase the number of output elements per thread until we reach the required bytes in flight.
    int chosen_elem_per_thread = 0;
    int chosen_tile_size       = 0;
    int chosen_smem_size       = 0;
    for (int elem_per_thread = +policy_t::MIN_ITEMS_PER_THREAD; elem_per_thread < +policy_t::MAX_ITEMS_PER_THREAD;
         ++elem_per_thread)
    {
      const auto tile_size = block_dim * elem_per_thread;

      int smem_size   = 0;
      auto count_smem = [&](int size, int alignment) {
        smem_size = round_up_to_multiple(smem_size, alignment);
        smem_size += size * tile_size;
      };
      // TODO(bgruber): replace by fold over comma in C++17 (left to right evaluation!)
      int dummy[] = {
        (count_smem(sizeof(value_t<RandomAccessIteratorsIn>), alignof(value_t<RandomAccessIteratorsIn>)), 0)..., 0};
      (void) dummy;

      if (smem_size > max_smem)
      {
        break;
      }

      if (tile_size >= num_items)
      {
        chosen_elem_per_thread = elem_per_thread;
        chosen_tile_size       = tile_size;
        chosen_smem_size       = smem_size;
        break;
      }

      int max_occupancy = 0;
      const auto error  = MaxSmOccupancy(max_occupancy, kernel, block_dim, smem_size);
      if (error != cudaSuccess)
      {
        return error;
      }

      chosen_elem_per_thread = elem_per_thread;
      chosen_tile_size       = tile_size;
      chosen_smem_size       = smem_size;

      const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
      if (bytes_in_flight_SM >= ActivePolicy::min_bif)
      {
        break;
      }
    }
    assert(chosen_elem_per_thread > 0);
    assert(chosen_tile_size > 0);
    assert((sizeof...(RandomAccessIteratorsIn) == 0) != (chosen_smem_size != 0)); // logical xor

    const Offset grid_dim = ::cuda::ceil_div(num_items, Offset{chosen_tile_size});
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, chosen_smem_size, stream),
      kernel,
      chosen_elem_per_thread);
  }

  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running
  // on?
  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto configure_ublkcp_kernel(cuda::std::index_sequence<Is...>)
    -> PoorExpected<::cuda::std::tuple<
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron,
      decltype(&transform_ublkcp_kernel<Offset, TransformOp, RandomAccessIteratorOut, value_t<RandomAccessIteratorsIn>...>),
      int>>
  {
    using policy_t          = typename ActivePolicy::ublkcp_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;

    auto kernel =
      transform_ublkcp_kernel<Offset, TransformOp, RandomAccessIteratorOut, value_t<RandomAccessIteratorsIn>...>;

    // Increase the number of output elements per thread until we reach the required bytes in flight.
    int chosen_elem_per_thread = 0;
    int chosen_tile_size       = 0;
    int chosen_smem_size       = 0;
    for (int elem_per_thread = +policy_t::MIN_ITEMS_PER_THREAD; elem_per_thread < +policy_t::MAX_ITEMS_PER_THREAD;
         ++elem_per_thread)
    {
      constexpr int num_inputs = sizeof...(RandomAccessIteratorsIn);
      const int tile_size      = block_dim * elem_per_thread;
      const int smem_size      = tile_size * loaded_bytes_per_iter + 128 * num_inputs; // 128 bytes of padding for each
                                                                                  // input tile

      if (smem_size > max_smem)
      {
        break;
      }

      if (tile_size >= num_items)
      {
        chosen_elem_per_thread = elem_per_thread;
        chosen_tile_size       = tile_size;
        chosen_smem_size       = smem_size;
        break;
      }

      int max_occupancy = 0;
      const auto error  = MaxSmOccupancy(max_occupancy, kernel, block_dim, smem_size);
      if (error != cudaSuccess)
      {
        return error;
      }

      chosen_elem_per_thread = elem_per_thread;
      chosen_tile_size       = tile_size;
      chosen_smem_size       = smem_size;

      const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
      if (ActivePolicy::min_bif <= bytes_in_flight_SM)
      {
        break;
      }
    }
    assert(chosen_elem_per_thread > 0);
    assert(chosen_tile_size > 0);
    assert((sizeof...(RandomAccessIteratorsIn) == 0) != (chosen_smem_size != 0)); // logical xor

    const Offset grid_dim = ::cuda::ceil_div(num_items, Offset{chosen_tile_size});
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, chosen_smem_size, stream),
      kernel,
      chosen_elem_per_thread);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...> is, ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>)
  {
    auto ret = configure_ublkcp_kernel<ActivePolicy>(is);
    if (!ret)
    {
      return ret.error;
    }
    // TODO(bgruber): use a structured binding in C++17
    // auto [launcher, kernel, elem_per_thread] = *ret;
    return ::cuda::std::get<0>(*ret).doit(
      ::cuda::std::get<1>(*ret),
      num_items,
      ::cuda::std::get<2>(*ret),
      op,
      out,
      make_ublkcp_ptr_set<const value_t<RandomAccessIteratorsIn>>(
        THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(::cuda::std::get<Is>(in)))...);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_algorithm(
    cuda::std::index_sequence<Is...> is, ::cuda::std::integral_constant<Algorithm, Algorithm::memcpy_async>)
  {
    auto ret = configure_memcpy_async_kernel<ActivePolicy>(is);
    if (!ret)
    {
      return ret.error;
    }
    // TODO(bgruber): use a structured binding in C++17
    // auto [launcher, kernel, elem_per_thread] = *ret;
    return ::cuda::std::get<0>(*ret).doit(
      ::cuda::std::get<1>(*ret),
      num_items,
      ::cuda::std::get<2>(*ret),
      op,
      out,
      THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(::cuda::std::get<Is>(in))...);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_algorithm(
    cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::unrolled_staged>)
  {
    using policy_t        = typename ActivePolicy::unrolled_policy;
    const Offset grid_dim = ::cuda::ceil_div(num_items, Offset{policy_t::BLOCK_THREADS * policy_t::ITEMS_PER_THREAD});
    auto kernel =
      transform_unrolled_staged_kernel<ActivePolicy,
                                       Offset,
                                       TransformOp,
                                       RandomAccessIteratorOut,
                                       RandomAccessIteratorsIn...>;
    return THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, policy_t::BLOCK_THREADS, 0, stream)
      .doit(kernel, num_items, op, out, ::cuda::std::get<Is>(in)...);
  }

  template <typename ActivePolicy, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>)
  {
    using policy_t          = typename ActivePolicy::prefetch_policy;
    constexpr int block_dim = policy_t::BLOCK_THREADS;

    auto kernel =
      transform_prefetch_kernel<Offset,
                                TransformOp,
                                RandomAccessIteratorOut,
                                typename std::iterator_traits<RandomAccessIteratorsIn>::value_type...>;

    int max_occupancy = 0;
    const auto error  = MaxSmOccupancy(max_occupancy, kernel, block_dim, 0);
    if (error != cudaSuccess)
    {
      return error;
    }

    const int items_per_thread =
      ::cuda::ceil_div(ActivePolicy::min_bif, max_occupancy * block_dim * loaded_bytes_per_iter);
    const int items_per_thread_clamped =
      ::cuda::std::clamp(items_per_thread, +policy_t::MIN_ITEMS_PER_THREAD, +policy_t::MAX_ITEMS_PER_THREAD);
    const Offset tile_size = block_dim * items_per_thread_clamped;
    const Offset grid_dim  = ::cuda::ceil_div(num_items, tile_size);

    return THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(grid_dim, block_dim, 0, stream)
      .doit(kernel,
            num_items,
            items_per_thread_clamped,
            op,
            out,
            THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(::cuda::std::get<Is>(in))...);
  }

  template <std::size_t... Is>
  struct non_contiguous_fallback_op_t
  {
    ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
    RandomAccessIteratorOut out;
    mutable TransformOp op; // too many users forgot to mark there operator()'s const ...

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void operator()(Offset i) const
    {
      out[i] = op(::cuda::std::get<Is>(in)[i]...);
    }
  };

  template <typename, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_algorithm(cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::fallback_for>)
  {
    using op_t = non_contiguous_fallback_op_t<Is...>;
    return for_each::dispatch_t<Offset, op_t>::dispatch(num_items, op_t{in, out, op}, stream);
  }

  // TODO(bgruber): check if there is any benefit in having a dedicated path for no input streams
  // struct fill_opt_t
  // {
  //   RandomAccessIteratorOut out;
  //   mutable TransformOp op; // too many users forgot to mark their operator()'s const ...
  //
  //   _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void operator()(Offset i) const
  //   {
  //     out[i] = op();
  //   }
  // };
  //
  // template <typename>
  // CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  // invoke_algorithm(cuda::std::index_sequence<>, ::cuda::std::integral_constant<Algorithm, Algorithm::fallback_for>)
  // {
  //   return for_each::dispatch_t<Offset, fill_opt_t>::dispatch(num_items, fill_opt_t{out, op}, stream);
  // }

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    constexpr bool no_input_streams = sizeof...(RandomAccessIteratorsIn) == 0;
    constexpr bool all_contiguous =
      ::cuda::std::conjunction<THRUST_NS_QUALIFIER::is_contiguous_iterator<RandomAccessIteratorsIn>...>::value;
    constexpr bool all_values_trivially_reloc =
      ::cuda::std::conjunction<THRUST_NS_QUALIFIER::is_trivially_relocatable<value_t<RandomAccessIteratorsIn>>...>::value;

    // TODO(bgruber): transform_unrolled_staged_kernel would work for non-contiguous iterators and
    // non-trivially-copyable value types. Try if it's faster.
    // TODO(bgruber): replace the overload set by if constexpr in C++17
    constexpr Algorithm algo =
      no_input_streams || !all_contiguous ? Algorithm::fallback_for
      : CanCopyArgsForOp && all_values_trivially_reloc
        ? ActivePolicy::alg_addr_unstable
        : ActivePolicy::alg_addr_stable;
    return invoke_algorithm<ActivePolicy>(
      ::cuda::std::index_sequence_for<RandomAccessIteratorsIn...>{}, ::cuda::std::integral_constant<Algorithm, algo>{});
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
    Offset num_items,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
    RandomAccessIteratorOut out,
    TransformOp op,
    cudaStream_t stream)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int ptx_version = 0;
    auto error      = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    // gevtushenko promised me that I can assume that stream belongs to the currently active device
    int device = 0;
    error      = CubDebug(cudaGetDevice(&device));
    if (error != cudaSuccess)
    {
      return error;
    }

    int max_smem = 0;
    error        = CubDebug(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (error != cudaSuccess)
    {
      return error;
    }

    dispatch_t dispatch{
      num_items, ::cuda::std::move(in), ::cuda::std::move(out), ::cuda::std::move(op), stream, max_smem};
    return CubDebug(PolicyHub::max_policy::Invoke(ptx_version, dispatch));
  }
};
} // namespace transform
} // namespace detail
CUB_NAMESPACE_END
