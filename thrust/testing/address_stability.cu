#include <thrust/address_stability.h>

#include <unittest/unittest.h>

struct MyPlus
{
  _CCCL_HOST_DEVICE auto operator()(int a, int b) const -> int
  {
    return a + b;
  }
};

struct Overloaded
{
  // oblivious
  _CCCL_HOST_DEVICE auto operator()(int a, int b) const -> int
  {
    return a + b;
  }

  // not oblivious
  _CCCL_HOST_DEVICE auto operator()(const float& a, const float& b) const -> float
  {
    return a + b;
  }
};

struct Addable
{
  _CCCL_HOST_DEVICE friend auto operator+(const Addable&, const Addable&) -> Addable
  {
    return Addable{};
  }
};

void TestAddressStability()
{
  using thrust::can_copy_arguments;
  using thrust::is_input_address_oblivious;

  static_assert(is_input_address_oblivious<thrust::plus<int>, int, int>::value, "");
  static_assert(is_input_address_oblivious<thrust::plus<>, int, int>::value, "");
  static_assert(!is_input_address_oblivious<thrust::plus<MyPlus>, int, int>::value, ""); // TODO should be fine

  static_assert(!is_input_address_oblivious<Overloaded, int, int>::value, ""); // TODO should be fine
  static_assert(!is_input_address_oblivious<Overloaded, float, float>::value, "");

  static_assert(can_copy_arguments<thrust::plus<int>, int*, int*>::value, "");
  static_assert(can_copy_arguments<thrust::plus<>, int*, int*>::value, "");
  static_assert(!can_copy_arguments<thrust::plus<Addable>, Addable*, Addable*>::value, "");
  static_assert(!can_copy_arguments<thrust::plus<>, Addable*, Addable*>::value, "");
}
DECLARE_UNITTEST(TestAddressStability);
