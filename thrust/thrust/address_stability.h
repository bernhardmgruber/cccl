// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/functional.h>

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

namespace detail
{
// need a separate implementation trait because we SFINAE with a type parameter before the variadic pack
template <typename F, typename SFINAE, typename... Args>
struct is_input_address_oblivious_impl : std::false_type
{};

template <typename F, typename... Args>
struct is_input_address_oblivious_impl<F, ::cuda::std::void_t<decltype(F::is_input_address_oblivious)>, Args...>
{
  static constexpr bool value = F::is_input_address_oblivious;
};

template <typename T>
struct has_builtin_operators
    : ::cuda::std::bool_constant<!::cuda::std::is_class<T>::value && !::cuda::std::is_enum<T>::value
                                 && !::cuda::std::is_void<T>::value>
{};
} // namespace detail

// TODO(bgruber): we may need to include the parameter types in the check, since the call operator of the functor could
// be overloaded.
// TODO(bgruber): bikeshed name, e.g., allow_copied_parameter
/// Trait telling whether a function object relies on the memory address of the input arguments when called with the
/// given set of types. The nested value is true when the addres of the inputs do not matter.
template <typename F, typename... Args>
using is_input_address_oblivious = detail::is_input_address_oblivious_impl<F, void, Args...>;

#define MARK_INPUT_ADDRESS_OBLIVIOUS(functor)                                                         \
  /*we know what thrust::plus<T> etc. do if T is not a type that could have a weird operatorX() */    \
  template <typename T, typename... Args>                                                             \
  struct detail::is_input_address_oblivious_impl<functor<T>, void, Args...>                           \
  {                                                                                                   \
    static constexpr bool value = detail::has_builtin_operators<T>::value;                            \
  };                                                                                                  \
  /*we know what thrust::plus<void> etc. do if T is not a type that could have a weird operatorX() */ \
  template <typename... Args>                                                                         \
  struct detail::is_input_address_oblivious_impl<functor<void>, void, Args...>                        \
      : ::cuda::std::conjunction<detail::has_builtin_operators<Args>...>                              \
  {};

// TODO(bgruber): move those close to where the functors are defined
MARK_INPUT_ADDRESS_OBLIVIOUS(thrust::plus);
MARK_INPUT_ADDRESS_OBLIVIOUS(thrust::minus);
MARK_INPUT_ADDRESS_OBLIVIOUS(thrust::negate);
// TODO(bgruber): need specializations for all function objects we know

#undef MARK_INPUT_ADDRESS_OBLIVIOUS

template <typename F, typename... Args>
struct detail::is_input_address_oblivious_impl<detail::not_fun_t<F>, void, Args...>
    : is_input_address_oblivious<F, Args...>
{};

namespace detail
{
template <typename F>
struct input_address_oblivious_wrapper : F
{
  using F::operator();
  static constexpr bool is_input_address_oblivious = true;
};
} // namespace detail

// TODO(bgruber): bikeshed name, e.g., allow_parameter_copies, etc.
/// Creates a new function object from an existing one, marking it as address oblivious (i.e., the addresses of input
/// arguments are irrelevant).
template <typename F>
_CCCL_HOST_DEVICE constexpr auto mark_input_address_oblivious(F f) -> detail::input_address_oblivious_wrapper<F>
{
  return detail::input_address_oblivious_wrapper<F>{std::move(f)};
}

template <typename TransformOp, typename... Its>
struct can_copy_arguments
{
  // TODO(bgruber): add detection whether user takes arguments by value, similar to how cub::DeviceFor does it
  static constexpr bool value =
    ::cuda::std::conjunction<::cuda::std::is_trivially_copyable<iterator_value_t<Its>>...>::value
    && is_input_address_oblivious<TransformOp, iterator_value_t<Its>...>::value; // TODO(bgruber): is
                                                                                 // iterator_value_t<Its> correct? Why
                                                                                 // not iterator_reference_t<Its>?
};

#if _CCCL_STD_VER >= 2014
template <typename TransformOp, typename... Its>
THRUST_INLINE_CONSTANT bool can_copy_arguments_v = can_copy_arguments<TransformOp, Its...>::value;
#endif

THRUST_NAMESPACE_END
