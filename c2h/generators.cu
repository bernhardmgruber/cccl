// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#define C2H_EXPORTS

#include <cub/device/device_copy.cuh>
#include <cub/util_type.cuh>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/type_traits>

#include <c2h/bfloat16.cuh>
#include <c2h/custom_type.h>
#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/fill_striped.h>
#include <c2h/generators.cuh>
#include <c2h/generators.h>
#include <c2h/half.cuh>
#include <c2h/vector.h>

#if C2H_HAS_CURAND
#  include <curand.h>
#else
#  include <thrust/random.h>
#endif

namespace c2h
{

#if !C2H_HAS_CURAND
struct i_to_rnd_t
{
  __host__ __device__ i_to_rnd_t(thrust::default_random_engine engine)
      : m_engine(engine)
  {}

  thrust::default_random_engine m_engine{};

  template <typename IndexType>
  __host__ __device__ float operator()(IndexType n)
  {
    m_engine.discard(n);
    return thrust::uniform_real_distribution<float>{0.0f, 1.0f}(m_engine);
  }
};
#endif // !C2H_HAS_CURAND

template <typename T>
struct random_to_item_t<cuda::std::complex<T>, false>
{
  cuda::std::complex<T> m_min;
  cuda::std::complex<T> m_max;

  __host__ __device__ random_to_item_t(cuda::std::complex<T> min, cuda::std::complex<T> max)
      : m_min(min)
      , m_max(max)
  {}

  __device__ cuda::std::complex<T> operator()(float random_value)
  {
    return (m_max - m_min) * cuda::std::complex<T>(random_value) + m_min;
  }
};

generator_t::generator_t()
{
#if C2H_HAS_CURAND
  curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
#endif
}

generator_t::~generator_t()
{
#if C2H_HAS_CURAND
  curandDestroyGenerator(m_gen);
#endif
}

float* generator_t::distribution()
{
  return thrust::raw_pointer_cast(m_distribution.data());
}

void generator_t::generate()
{
#if C2H_HAS_CURAND
  curandGenerateUniform(m_gen, this->distribution(), this->m_distribution.size());
#else
  thrust::tabulate(c2h::device_policy, this->m_distribution.begin(), this->m_distribution.end(), i_to_rnd_t{m_re});
  m_re.discard(this->m_distribution.size());
#endif
}

float* generator_t::prepare_random_generator(seed_t seed, std::size_t num_items)
{
  m_distribution.resize(num_items);

#if C2H_HAS_CURAND
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());
#else
  m_re.seed(seed.get());
#endif

  generate();

  return this->distribution();
}

template <bool SetKeys>
struct random_to_custom_t
{
  static constexpr std::size_t m_max_key = std::numeric_limits<std::size_t>::max();

  __device__ void operator()(std::size_t idx)
  {
    std::size_t in = static_cast<std::size_t>(static_cast<float>(m_max_key) * m_in[idx]);

    custom_type_state_t* out = reinterpret_cast<custom_type_state_t*>(m_out + idx * m_element_size);

    if (SetKeys)
    {
      out->key = in;
    }
    else
    {
      out->val = in;
    }
  }

  random_to_custom_t(float* in, char* out, std::size_t element_size)
      : m_in(in)
      , m_out(out)
      , m_element_size(element_size)
  {}

  float* m_in{};
  char* m_out{};
  std::size_t m_element_size{};
};

template <class T>
void generator_t::operator()(seed_t seed, c2h::device_vector<T>& data, T min, T max)
{
  prepare_random_generator(seed, data.size());

  thrust::transform(
    c2h::device_policy, m_distribution.begin(), m_distribution.end(), data.begin(), random_to_item_t<T>(min, max));
}

template <typename T>
struct count_to_item_t
{
  uint64_t n;

  template <typename CounterT>
  __device__ T operator()(CounterT id)
  {
    // This has to be a type for which extended floating point types like __nv_fp8_e5m2 provide an overload
    return static_cast<T>(static_cast<float>(static_cast<uint64_t>(id) % n));
  }
};

template <typename T>
void generator_t::operator()(modulo_t mod, c2h::device_vector<T>& data)
{
  thrust::tabulate(c2h::device_policy, data.begin(), data.end(), count_to_item_t<T>{mod.get()});
}

generator_t& generator_t::instance()
{
  static generator_t generator;
  return generator;
}

namespace detail
{

void gen(seed_t seed,
         char* d_out,
         custom_type_state_t /* min */,
         custom_type_state_t /* max */,
         std::size_t elements,
         std::size_t element_size)
{
  thrust::counting_iterator<std::size_t> cnt_begin(0);
  thrust::counting_iterator<std::size_t> cnt_end(elements);

  generator_t& generator = generator_t::instance();
  float* d_in            = generator.prepare_random_generator(seed, elements);

  thrust::for_each(c2h::device_policy, cnt_begin, cnt_end, random_to_custom_t<true>{d_in, d_out, element_size});

  generator.generate();

  thrust::for_each(c2h::device_policy, cnt_begin, cnt_end, random_to_custom_t<false>{d_in, d_out, element_size});
}

template <class T>
struct greater_equal_op
{
  T val;

  __device__ bool operator()(T x)
  {
    return x >= val;
  }
};

template <typename T>
struct spaced_out_it_op
{
  char* base_it;
  std::size_t element_size;

  __host__ __device__ __forceinline__ T& operator()(std::size_t offset) const
  {
    return *reinterpret_cast<T*>(base_it + (element_size * offset));
  }
};

template <typename T>
struct offset_to_iterator_t
{
  char* base_it;
  std::size_t element_size;

  __host__
    __device__ __forceinline__ thrust::transform_iterator<spaced_out_it_op<T>, thrust::counting_iterator<std::size_t>>
    operator()(std::size_t offset) const
  {
    // The pointer to the beginning of this "buffer" (aka a series of same "keys")
    auto base_ptr = base_it + (element_size * offset);

    // We need to make sure that the i-th element within this "buffer" is spaced out by
    // `element_size`
    auto counting_it = thrust::make_counting_iterator(std::size_t{0});
    spaced_out_it_op<T> space_out_op{base_ptr, element_size};
    return thrust::make_transform_iterator(counting_it, space_out_op);
  }
};

template <class T>
struct repeat_index_t
{
  __host__ __device__ __forceinline__ thrust::constant_iterator<T> operator()(std::size_t i)
  {
    return thrust::constant_iterator<T>(static_cast<T>(i));
  }
};

template <>
struct repeat_index_t<custom_type_state_t>
{
  __host__ __device__ __forceinline__ thrust::constant_iterator<custom_type_state_t> operator()(std::size_t i)
  {
    custom_type_state_t item{};
    item.key = i;
    item.val = i;
    return thrust::constant_iterator<custom_type_state_t>(item);
  }
};

template <typename OffsetT>
struct offset_to_size_t
{
  const OffsetT* offsets;

  __host__ __device__ __forceinline__ std::size_t operator()(std::size_t i)
  {
    return offsets[i + 1] - offsets[i];
  }
};

/**
 * @brief Initializes key-segment ranges from an offsets-array like the one given by
 * `gen_uniform_offset`.
 */
template <typename OffsetT, typename KeyT>
void init_key_segments(const c2h::device_vector<OffsetT>& segment_offsets, KeyT* d_out, std::size_t element_size)
{
  OffsetT total_segments   = static_cast<OffsetT>(segment_offsets.size() - 1);
  const OffsetT* d_offsets = thrust::raw_pointer_cast(segment_offsets.data());

  thrust::counting_iterator<int> iota(0);
  offset_to_iterator_t<KeyT> dst_transform_op{reinterpret_cast<char*>(d_out), element_size};

  auto d_range_srcs  = thrust::make_transform_iterator(iota, repeat_index_t<KeyT>{});
  auto d_range_dsts  = thrust::make_transform_iterator(d_offsets, dst_transform_op);
  auto d_range_sizes = thrust::make_transform_iterator(iota, offset_to_size_t<OffsetT>{d_offsets});

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  std::uint8_t* d_temp_storage   = nullptr;
  std::size_t temp_storage_bytes = 0;
  // TODO(bgruber): replace by a non-CUB implementation
  cub::DeviceCopy::Batched(
    d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // TODO(bgruber): replace by a non-CUB implementation
  cub::DeviceCopy::Batched(
    d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);
  cudaDeviceSynchronize();
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  static_assert(sizeof(OffsetT) == 0, "Need to implement a non-CUB version of cub::DeviceCopy::Batched");
  // TODO(bgruber): implement and *test* a non-CUB version, here is a sketch:
  // thrust::for_each(
  //   thrust::device,
  //   thrust::counting_iterator<OffsetT>{0},
  //   thrust::counting_iterator<OffsetT>{total_segments},
  //   [&](OffsetT i) {
  //     const auto value = d_range_srcs[i];
  //     const auto start = d_range_sizes[i];
  //     const auto end   = d_range_sizes[i + 1];
  //     for (auto j = start; j < end; ++j)
  //     {
  //       d_range_dsts[j] = value;
  //     }
  //   });
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
}

template void init_key_segments(
  const c2h::device_vector<std::uint32_t>& segment_offsets, std::int32_t* out, std::size_t element_size);
template void init_key_segments(
  const c2h::device_vector<std::uint32_t>& segment_offsets, std::uint8_t* out, std::size_t element_size);
template void
init_key_segments(const c2h::device_vector<std::uint32_t>& segment_offsets, float* out, std::size_t element_size);
template void init_key_segments(
  const c2h::device_vector<std::uint32_t>& segment_offsets, custom_type_state_t* out, std::size_t element_size);
#if TEST_HALF_T()
template void
init_key_segments(const c2h::device_vector<std::uint32_t>& segment_offsets, half_t* out, std::size_t element_size);
#endif // TEST_HALF_T()

#if TEST_BF_T()
template void
init_key_segments(const c2h::device_vector<std::uint32_t>& segment_offsets, bfloat16_t* out, std::size_t element_size);
#endif // TEST_BF_T()
} // namespace detail

template <typename T>
c2h::device_vector<T> gen_uniform_offsets(seed_t seed, T total_elements, T min_segment_size, T max_segment_size)
{
  c2h::device_vector<T> segment_offsets(total_elements + 2);
  gen(seed, segment_offsets, min_segment_size, max_segment_size);
  segment_offsets[total_elements] = total_elements + 1;
  thrust::exclusive_scan(c2h::device_policy, segment_offsets.begin(), segment_offsets.end(), segment_offsets.begin());
  typename c2h::device_vector<T>::iterator iter = thrust::find_if(
    c2h::device_policy, segment_offsets.begin(), segment_offsets.end(), detail::greater_equal_op<T>{total_elements});
  *iter = total_elements;
  segment_offsets.erase(iter + 1, segment_offsets.end());
  return segment_offsets;
}

template c2h::device_vector<int32_t>
gen_uniform_offsets(seed_t seed, int32_t total_elements, int32_t min_segment_size, int32_t max_segment_size);

template c2h::device_vector<uint32_t>
gen_uniform_offsets(seed_t seed, uint32_t total_elements, uint32_t min_segment_size, uint32_t max_segment_size);
template c2h::device_vector<int64_t>
gen_uniform_offsets(seed_t seed, int64_t total_elements, int64_t min_segment_size, int64_t max_segment_size);
template c2h::device_vector<uint64_t>
gen_uniform_offsets(seed_t seed, uint64_t total_elements, uint64_t min_segment_size, uint64_t max_segment_size);

template <typename T>
void gen(seed_t seed, c2h::device_vector<T>& data, T min, T max)
{
  generator_t::instance()(seed, data, min, max);
}

template <typename T>
void gen(modulo_t mod, c2h::device_vector<T>& data)
{
  generator_t::instance()(mod, data);
}

#define INSTANTIATE_RND(TYPE) template void gen<TYPE>(seed_t, c2h::device_vector<TYPE> & data, TYPE min, TYPE max)
#define INSTANTIATE_MOD(TYPE) template void gen<TYPE>(modulo_t, c2h::device_vector<TYPE> & data)

#define INSTANTIATE(TYPE) \
  INSTANTIATE_RND(TYPE);  \
  INSTANTIATE_MOD(TYPE)

INSTANTIATE(std::uint8_t);
INSTANTIATE(std::uint16_t);
INSTANTIATE(std::uint32_t);
INSTANTIATE(std::uint64_t);

INSTANTIATE(std::int8_t);
INSTANTIATE(std::int16_t);
INSTANTIATE(std::int32_t);
INSTANTIATE(std::int64_t);

#if _CCCL_HAS_NVFP8()
INSTANTIATE(__nv_fp8_e5m2);
INSTANTIATE(__nv_fp8_e4m3);
#endif // _CCCL_HAS_NVFP8()
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cuda::std::complex<float>);
INSTANTIATE(cuda::std::complex<double>);

INSTANTIATE(bool);
INSTANTIATE(char);

#if TEST_HALF_T()
INSTANTIATE(half_t);
INSTANTIATE(__half);
#  if _CCCL_CUDACC_AT_LEAST(12, 2)
INSTANTIATE(cuda::std::complex<__half>);
#  endif
#endif // TEST_HALF_T()

#if TEST_BF_T()
INSTANTIATE(bfloat16_t);
INSTANTIATE(__nv_bfloat16);
#  if _CCCL_CUDACC_AT_LEAST(12, 2)
INSTANTIATE(cuda::std::complex<__nv_bfloat16>);
#  endif
#endif // TEST_BF_T()

#if TEST_INT128()
INSTANTIATE(__int128_t);
INSTANTIATE(__uint128_t);
#endif // TEST_INT128()

#undef INSTANTIATE_RND
#undef INSTANTIATE_MOD
#undef INSTANTIATE

} // namespace c2h
