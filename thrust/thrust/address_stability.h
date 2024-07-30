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
} // namespace detail

// TODO(bgruber): bikeshed name, e.g., allow_copied_parameter
/// Trait telling whether a function object relies on the memory address of the input arguments when called with the
/// given set of types. The nested value is true when the addres of the inputs do not matter.
template <typename F, typename... Args>
using is_input_address_oblivious = detail::is_input_address_oblivious_impl<F, void, Args...>;

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

THRUST_NAMESPACE_END
