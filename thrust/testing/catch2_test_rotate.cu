#include <thrust/iterator/retag.h>
#include <thrust/rotate.h>

#include <cuda/std/cmath>

#include "catch2_test_helper.h"
#include "unittest/special_types.h"
#include <unittest/random.h>
#include <unittest/special_types.h>

TEMPLATE_LIST_TEST_CASE("RotateSimple", "[rotate]", vector_list)
{
  using Vector = TestType;
  Vector v{1, 4, 6, 7};
  CAPTURE(unittest::type_name<Vector>());

  SECTION("small")
  {
    const auto result = thrust::rotate(v.begin(), v.begin() + 1, v.end());
    CHECK(result == v.begin() + 3);
    CHECK(v == Vector{4, 6, 7, 1});
  }

  SECTION("large")
  {
    const auto result = thrust::rotate(v.begin(), v.begin() + 3, v.end());
    CHECK(result == v.begin() + 1);
    CHECK(v == Vector{7, 1, 4, 6});
  }
}

TEMPLATE_LIST_TEST_CASE("Rotate", "[rotate]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    const auto mid = n == 0 ? 0 : unittest::random_integer<size_t>() % n;
    CAPTURE(n, mid);

    thrust::host_vector<T> h_vec   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_vec = h_vec;

    const auto h_result = thrust::rotate(h_vec.begin(), h_vec.begin() + mid, h_vec.end());
    const auto d_result = thrust::rotate(d_vec.begin(), d_vec.begin() + mid, d_vec.end());

    CHECK(std::size_t(h_result - h_vec.begin()) == n - mid);
    CHECK(std::size_t(d_result - d_vec.begin()) == n - mid);
    CHECK(d_vec == h_vec);
  }
}

template <typename ForwardIterator>
ForwardIterator rotate(my_system& system, ForwardIterator first, ForwardIterator middle, ForwardIterator last)
{
  system.validate_dispatch();
  return first + cuda::std::distance(middle, last);
}

TEST_CASE("RotateDispatchExplicit", "[rotate]")
{
  thrust::device_vector<int> d_vec(1);

  my_system sys(0);
  thrust::rotate(sys, d_vec.begin(), d_vec.begin(), d_vec.end());

  CHECK(sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator rotate(my_tag&, ForwardIterator first, ForwardIterator, ForwardIterator)
{
  *first = 13;
  return first;
}

TEST_CASE("RotateDispatchImplicit", "[rotate]")
{
  thrust::device_vector<int> d_input(1);
  thrust::rotate(thrust::retag<my_tag>(d_input.begin()),
                 thrust::retag<my_tag>(d_input.begin()),
                 thrust::retag<my_tag>(d_input.end()));
  CHECK(13 == d_input.front());
}
