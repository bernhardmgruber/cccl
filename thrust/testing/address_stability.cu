#include <thrust/address_stability.h>

#include <unittest/unittest.h>

struct MyPlus
{
  _CCCL_HOST_DEVICE auto operator()(int a, int b) const -> int
  {
    return a + b;
  }
};

void TestAddressStability()
{
  using thrust::is_input_address_oblivious;

  static_assert(!is_input_address_oblivious<thrust::plus<int>, int, int>::value, "");
  static_assert(
    is_input_address_oblivious<decltype(thrust::mark_input_address_oblivious(thrust::plus<int>{})), int, int>::value,
    "");

  static_assert(!is_input_address_oblivious<MyPlus, int, int>::value, "");
  static_assert(is_input_address_oblivious<decltype(thrust::mark_input_address_oblivious(MyPlus{})), int, int>::value,
                "");
}
DECLARE_UNITTEST(TestAddressStability);
