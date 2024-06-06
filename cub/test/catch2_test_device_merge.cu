// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_merge.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <algorithm>

#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"
#include <test_util.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergePairs, merge_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergeKeys, merge_keys);

using types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

// TODO(bgruber): the sizes are from elias from select_if, verify they actually work for merge
using large_types = c2h::type_list<
  // Type large enough to dispatch to the fallback policy
  c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<256>::type>,
  // Type large enough to require virtual shared memory
  c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<512>::type>>;

using offset_types =
  c2h::type_list<std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;

// TODO(bgruber): does Catch2 have a setting to not print large output?
constexpr auto catch_max_vector_size_to_print = 400;

// TODO(bgruber): merge_pairs subsumes a lot of functionality of merge_keys, so we can probably reduce merge_key tests
// by a lot

template <typename Key, typename Offset, typename CompareOp = ::cuda::std::less<Key>>
void test_keys(Offset size1 = 3623, Offset size2 = 6346)
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Offset>(), size1, size2);

  c2h::device_vector<Key> keys1_d(size1);
  c2h::device_vector<Key> keys2_d(size2);

  c2h::gen(CUB_SEED(1), keys1_d);
  c2h::gen(CUB_SEED(1), keys2_d);

  thrust::sort(keys1_d.begin(), keys1_d.end(), CompareOp{});
  thrust::sort(keys2_d.begin(), keys2_d.end(), CompareOp{});

  if (size1 + size2 < catch_max_vector_size_to_print)
  {
    CAPTURE(keys1_d, keys2_d);
  }

  c2h::device_vector<Key> result_d(size1 + size2);
  merge_keys(keys1_d.begin(),
             static_cast<Offset>(keys1_d.size()),
             keys2_d.begin(),
             static_cast<Offset>(keys2_d.size()),
             result_d.begin(),
             CompareOp{});

  c2h::host_vector<Key> keys1_h = keys1_d;
  c2h::host_vector<Key> keys2_h = keys2_d;
  c2h::host_vector<Key> reference_h(size1 + size2);
  std::merge(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), reference_h.begin(), CompareOp{});

  if (reference_h.size() < catch_max_vector_size_to_print)
  {
    CHECK(reference_h == result_d);
  }
  else
  {
    const auto equal = reference_h == result_d;
    REQUIRE(equal);
  }
}

CUB_TEST("DeviceMerge::MergeKeys key types", "[merge][device]", types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = int;
  test_keys<key_t, offset_t>();
}

CUB_TEST("DeviceMerge::MergeKeys large key types", "[merge][device]", large_types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = int;
  test_keys<key_t, offset_t>();
}

CUB_TEST("DeviceMerge::MergeKeys offset types", "[merge][device]", offset_types)
{
  using key_t    = int;
  using offset_t = c2h::get<0, TestType>;
  test_keys<key_t, offset_t>();
}

CUB_TEST("DeviceMerge::MergeKeys input sizes", "[merge][device]")
{
  using key_t    = int;
  using offset_t = int;
  // TODO(bgruber): maybe less combinations
  const auto size1 = offset_t{GENERATE(0, 1, 23, 123, 3234)};
  const auto size2 = offset_t{GENERATE(0, 1, 52, 556, 56767)};
  test_keys<key_t>(size1, size2);
}

namespace
{
using unordered_t = c2h::custom_type_t<c2h::equal_comparable_t>;
struct order
{
  _CCCL_HOST_DEVICE auto operator()(const unordered_t& a, const unordered_t& b) const -> bool
  {
    return a.key < b.key;
  }
};
} // namespace

CUB_TEST("DeviceMerge::MergeKeys no operator<", "[merge][device]")
{
  using key_t    = unordered_t;
  using offset_t = int;
  test_keys<key_t, offset_t, order>();
}

namespace
{
template <typename... Its>
auto zip(Its... its)
{
  return thrust::make_zip_iterator(its...);
}

template <typename Value>
struct key_to_value
{
  template <typename Key>
  _CCCL_HOST_DEVICE auto operator()(const Key& k) const -> Value
  {
    Value v{};
    convert(k, v, 0);
    return v;
  }

  template <typename Key>
  _CCCL_HOST_DEVICE static void convert(const Key& k, Value& v, ...)
  {
    v = static_cast<Value>(k);
  }

  template <template <typename> class... Policies>
  _CCCL_HOST_DEVICE static void convert(const c2h::custom_type_t<Policies...>& k, Value& v, int)
  {
    v = static_cast<Value>(k.val);
  }

  template <typename Key, template <typename> class... Policies>
  _CCCL_HOST_DEVICE static void convert(const Key& k, c2h::custom_type_t<Policies...>& v, int)
  {
    v     = {};
    v.val = static_cast<decltype(v.val)>(k);
  }
};
} // namespace

template <typename Key, typename Value, typename Offset, typename CompareOp = ::cuda::std::less<Key>>
void test_pairs(Offset size1 = 200, Offset size2 = 625)
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Value>(), c2h::type_name<Offset>(), size1, size2);

  // we start with random but sorted keys
  c2h::device_vector<Key> keys1_d(size1);
  c2h::device_vector<Key> keys2_d(size2);
  c2h::gen(CUB_SEED(1), keys1_d);
  c2h::gen(CUB_SEED(1), keys2_d);
  thrust::sort(keys1_d.begin(), keys1_d.end());
  thrust::sort(keys2_d.begin(), keys2_d.end());

  // the values must be functionally dependent on the keys (equal key => equal value), since merge is unstable
  c2h::device_vector<Value> values1_d(size1);
  c2h::device_vector<Value> values2_d(size2);
  thrust::transform(keys1_d.begin(), keys1_d.end(), values1_d.begin(), key_to_value<Value>{});
  thrust::transform(keys2_d.begin(), keys2_d.end(), values2_d.begin(), key_to_value<Value>{});

  if (size1 + size2 < catch_max_vector_size_to_print)
  {
    CAPTURE(keys1_d, keys2_d, values1_d, values2_d);
  }

  // compute CUB result
  c2h::device_vector<Key> result_keys_d(size1 + size2);
  c2h::device_vector<Value> result_values_d(size1 + size2);
  merge_pairs(
    keys1_d.begin(),
    values1_d.begin(),
    static_cast<Offset>(keys1_d.size()),
    keys2_d.begin(),
    values2_d.begin(),
    static_cast<Offset>(keys2_d.size()),
    result_keys_d.begin(),
    result_values_d.begin(),
    CompareOp{});

  // compute reference result
  c2h::host_vector<Key> reference_keys_h(size1 + size2);
  c2h::host_vector<Value> reference_values_h(size1 + size2);
  {
    c2h::host_vector<Key> keys1_h     = keys1_d;
    c2h::host_vector<Value> values1_h = values1_d;
    c2h::host_vector<Key> keys2_h     = keys2_d;
    c2h::host_vector<Value> values2_h = values2_d;
    std::merge(zip(keys1_h.begin(), values1_h.begin()),
               zip(keys1_h.end(), values1_h.end()),
               zip(keys2_h.begin(), values2_h.begin()),
               zip(keys2_h.end(), values2_h.end()),
               zip(reference_keys_h.begin(), reference_values_h.begin()));
  }

  if (reference_keys_h.size() < catch_max_vector_size_to_print)
  {
    CHECK(reference_keys_h == result_keys_d);
    CHECK(reference_values_h == result_values_d);
  }
  else
  {
    const auto keys_equal = reference_keys_h == result_keys_d;
    CHECK(keys_equal);
    const auto values_equal = reference_values_h == result_values_d;
    CHECK(values_equal);
  }
}

CUB_TEST("DeviceMerge::MergePairs key types", "[merge][device]", types)
{
  using key_t    = c2h::get<0, TestType>;
  using value_t  = int;
  using offset_t = int;
  test_pairs<key_t, value_t, offset_t>();
}

CUB_TEST("DeviceMerge::MergePairs large key types", "[merge][device]", large_types)
{
  using key_t    = c2h::get<0, TestType>;
  using value_t  = int;
  using offset_t = int;
  test_pairs<key_t, value_t, offset_t>();
}

CUB_TEST("DeviceMerge::MergePairs value types", "[merge][device]", types)
{
  using key_t    = int;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = int;
  test_pairs<key_t, value_t, offset_t>();
}

CUB_TEST("DeviceMerge::MergePairs offset types", "[merge][device]", offset_types)
{
  using key_t    = int;
  using value_t  = int;
  using offset_t = c2h::get<0, TestType>;
  test_pairs<key_t, value_t, offset_t>();
}

CUB_TEST("DeviceMerge::MergePairs input sizes", "[merge][device]")
{
  using key_t      = int;
  using value_t    = int;
  using offset_t   = int;
  const auto size1 = offset_t{GENERATE(0, 1, 23, 123, 3234234)};
  const auto size2 = offset_t{GENERATE(0, 1, 52, 556, 56767)};
  test_pairs<key_t, value_t>(size1, size2);
}

CUB_TEST("DeviceMerge::MergePairs really large input", "[merge][device]")
{
  using key_t     = char;
  using value_t   = char;
  const auto size = std::int64_t{1} << GENERATE(30, 31, 32, 33);
  test_pairs<key_t, value_t>(size, size);
}

CUB_TEST("DeviceMerge::MergePairs iterators", "[merge][device]")
{
  using key_t             = int;
  using value_t           = int;
  using offset_t          = int;
  const offset_t size1    = 363;
  const offset_t size2    = 634;
  const auto values_start = 123456789;

  auto key_it   = thrust::counting_iterator<key_t>{};
  auto value_it = thrust::counting_iterator<key_t>{values_start};

  // compute CUB result
  c2h::device_vector<key_t> result_keys_d(size1 + size2);
  c2h::device_vector<value_t> result_values_d(size1 + size2);
  merge_pairs(
    key_it,
    value_it,
    size1,
    key_it,
    value_it,
    size2,
    result_keys_d.begin(),
    result_values_d.begin(),
    ::cuda::std::less<key_t>{});

  // check result
  c2h::host_vector<key_t> result_keys_h     = result_keys_d;
  c2h::host_vector<value_t> result_values_h = result_values_d;
  const auto smaller_size                   = std::min(size1, size2);
  for (offset_t i = 0; i < static_cast<offset_t>(result_keys_h.size()); i++)
  {
    if (i < 2 * smaller_size)
    {
      CHECK(result_keys_h[i + 0] == i / 2);
      CHECK(result_values_h[i + 0] == values_start + i / 2);
    }
    else
    {
      CHECK(result_keys_h[i] == i - smaller_size);
      CHECK(result_values_h[i] == values_start + i - smaller_size);
    }
  }
}
