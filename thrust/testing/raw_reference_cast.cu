#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <unittest/unittest.h>

struct plus_one
{
  _CCCL_HOST_DEVICE int operator()(int i) const
  {
    return i + 1;
  }
};

struct pass_through
{
  template <typename T>
  _CCCL_HOST_DEVICE T& operator()(T& i) const
  {
    return i;
  }

  template <typename T>
  _CCCL_HOST_DEVICE const T& operator()(const T& i) const
  {
    return i;
  }
};

void TestRawReferenceCast()
{
  thrust::device_vector<int> v(10);
  const thrust::device_vector<int> v_c(10);

  static_assert(std::is_same<decltype(thrust::raw_reference_cast(v[0])), int&>::value, "");
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(v_c[0])), const int&>::value, "");

  // iterators
  auto it   = v.begin();
  auto it_c = v_c.begin();
  static_assert(std::is_same<typename decltype(it)::reference, thrust::device_reference<int>>::value, "");
  static_assert(std::is_same<typename decltype(it_c)::reference, thrust::device_reference<const int>>::value, "");

  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*it)), int&>::value, "");
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*it_c)), const int&>::value, "");

  // transform iterator which yields a new value

  auto transform_it   = thrust::make_transform_iterator(it, plus_one{});
  auto transform_it_c = thrust::make_transform_iterator(it_c, plus_one{});
  static_assert(std::is_same<typename decltype(transform_it)::reference, int>::value, "");
  static_assert(std::is_same<typename decltype(transform_it_c)::reference, int>::value, "");

  // TODO(bgruber): unexpected, raw_reference_cast turns value into const&, which can dangle
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*transform_it)), const int&>::value, "");
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*transform_it_c)), const int&>::value, "");

  // transform iterator which should yield a reference

  auto transform_ref_it   = thrust::make_transform_iterator(it, pass_through{});
  auto transform_ref_it_c = thrust::make_transform_iterator(it_c, pass_through{});
  // TODO(bgruber): unexpected, transform iterator strips device_reference and magically adds a const
  static_assert(std::is_same<typename decltype(transform_ref_it)::reference, const int&>::value, "");
  static_assert(std::is_same<typename decltype(transform_ref_it_c)::reference, const int&>::value, "");

  // TODO(bgruber): those should probably be thrust::device_reference<int> and thrust::device_reference<const int>
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*transform_ref_it)), const int&>::value, "");
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*transform_ref_it_c)), const int&>::value, ""); // ok

  // zip_iterator

  auto zip_it = thrust::make_zip_iterator(it, it_c);
  static_assert(std::is_same<typename decltype(zip_it)::reference,
                             thrust::detail::tuple_of_iterator_references<thrust::device_reference<int>,
                                                                          thrust::device_reference<const int>>>::value,
                "");
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*zip_it)),
                             thrust::detail::tuple_of_iterator_references<int&, const int&>>::value,
                "");

  // transformed zip_iterator

  auto transform_zip_it = thrust::make_transform_iterator(zip_it, thrust::plus<int>{});
  static_assert(std::is_same<typename decltype(transform_zip_it)::reference, int>::value, "");

  // TODO(bgruber): unexpected, raw_reference_cast turns int into const int&, which can dangle
  static_assert(std::is_same<decltype(thrust::raw_reference_cast(*transform_zip_it)), const int&>::value, "");

  // zipped transform_iterators

  auto zip_transform_it = thrust::make_zip_iterator(transform_it, transform_it_c, transform_ref_it, transform_ref_it_c);
  static_assert(std::is_same<typename decltype(zip_transform_it)::reference,
                             thrust::detail::tuple_of_iterator_references<int, int, const int&, const int&>>::value,
                "");

  (void) transform_zip_it;
  (void) zip_transform_it;
}
DECLARE_UNITTEST(TestRawReferenceCast);
