#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include "catch2_test_helper.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // warning C4244: '=': conversion from 'int' to '_Ty', possible loss of data

struct make_host_vector
{
  template <typename T>
  using fn = thrust::host_vector<T>;
};

struct make_device_vector
{
  template <typename T>
  using fn = thrust::device_vector<T>;
};

#define BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, op, reference_functor, type_list)                                  \
  TEMPLATE_LIST_TEST_CASE(                                                                                           \
    #name,                                                                                                           \
    "[functional]",                                                                                                  \
    (cuda::std::__type_cartesian_product<type_list, cuda::std::__type_list<make_host_vector, make_device_vector>>) ) \
  {                                                                                                                  \
    using T                         = cuda::std::__type_get<0>(TestType);                                            \
    using Vector                    = cuda::std::__type_get<1>::template fn<T>;                                      \
    static const size_t num_samples = 10000;                                                                         \
    const size_t zero               = 0;                                                                             \
    Vector lhs                      = unittest::random_samples<T>(num_samples);                                      \
    Vector rhs                      = unittest::random_samples<T>(num_samples);                                      \
    thrust::replace(rhs.begin(), rhs.end(), T(0), T(1));                                                             \
                                                                                                                     \
    Vector reference(lhs.size());                                                                                    \
    Vector result(lhs.size());                                                                                       \
    using namespace thrust::placeholders;                                                                            \
                                                                                                                     \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), reference_functor<T>());               \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 op _2);                                \
    CHECK_APPROX_EQ(reference, Catch::Matchers::Approx(result));                                                     \
                                                                                                                     \
    thrust::transform(                                                                                               \
      lhs.begin(), lhs.end(), thrust::make_constant_iterator<T>(1), reference.begin(), reference_functor<T>());      \
    thrust::transform(lhs.begin(), lhs.end(), result.begin(), _1 op T(1));                                           \
    CHECK_APPROX_EQ(reference, Catch::Matchers::Approx(result));                                                     \
                                                                                                                     \
    thrust::transform(                                                                                               \
      thrust::make_constant_iterator<T>(1, zero),                                                                    \
      thrust::make_constant_iterator<T>(1, num_samples),                                                             \
      rhs.begin(),                                                                                                   \
      reference.begin(),                                                                                             \
      reference_functor<T>());                                                                                       \
    thrust::transform(rhs.begin(), rhs.end(), result.begin(), T(1) op _1);                                           \
    CHECK_APPROX_EQ(reference, Catch::Matchers::Approx(result));                                                     \
  }

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Plus, +, ::cuda::std::plus, ThirtyTwoBitTypes);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Minus, -, ::cuda::std::minus, ThirtyTwoBitTypes);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Multiplies, *, ::cuda::std::multiplies, ThirtyTwoBitTypes);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Divides, /, ::cuda::std::divides, ThirtyTwoBitTypes);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Modulus, %, ::cuda::std::modulus, SmallIntegralTypes);

#define UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor)             \
  TEMPLATE_LIST_TEST_CASE(#name, "[functional]", vector_list)                             \
  {                                                                                       \
    using Vector                 = TestType;                                              \
    constexpr size_t num_samples = 10000;                                                 \
    using T                      = typename Vector::value_type;                           \
    Vector input                 = unittest::random_samples<T>(num_samples);              \
                                                                                          \
    Vector reference(input.size());                                                       \
    thrust::transform(input.begin(), input.end(), reference.begin(), functor<T>());       \
                                                                                          \
    using namespace thrust::placeholders;                                                 \
    Vector result(input.size());                                                          \
    thrust::transform(input.begin(), input.end(), result.begin(), reference_operator _1); \
                                                                                          \
    CHECK(reference == result);                                                           \
  }

template <typename T>
struct unary_plus_reference
{
  _CCCL_HOST_DEVICE T operator()(const T& x) const
  {
    return static_cast<T>(+x); // undo integral promotion
  }
};

UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(UnaryPlus, +, unary_plus_reference);
UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(Negate, -, ::cuda::std::negate);

_CCCL_DIAG_POP
