/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#  include <thrust/system/cuda/config.h>

#  include <cub/device/device_transform.cuh>

#  include <thrust/address_stability.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/zip_function.h>

#  include <cstdint>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
namespace __transform
{
struct no_stencil_tag
{};

struct always_true_predicate
{
  template <class T>
  constexpr bool THRUST_DEVICE_FUNCTION operator()(T const&) const
  {
    return true;
  }
};

template <class InputIt, class OutputIt, class StencilIt, class TransformOp, class Predicate>
struct unary_transform_f
{
  InputIt input;
  OutputIt output;
  StencilIt stencil;
  TransformOp op;
  Predicate pred;

  THRUST_FUNCTION
  unary_transform_f(InputIt input_, OutputIt output_, StencilIt stencil_, TransformOp op_, Predicate pred_)
      : input(input_)
      , output(output_)
      , stencil(stencil_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(stencil[idx])))
    {
      output[idx] = op(raw_reference_cast(input[idx]));
    }
  }
}; // struct unary_transform_stencil_f

template <class InputIt, class OutputIt, class TransformOp, class Predicate>
struct unary_transform_f<InputIt, OutputIt, no_stencil_tag, TransformOp, Predicate>
{
  InputIt input;
  OutputIt output;
  TransformOp op;
  Predicate pred;

  THRUST_FUNCTION
  unary_transform_f(InputIt input_, OutputIt output_, no_stencil_tag, TransformOp op_, Predicate pred_)
      : input(input_)
      , output(output_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(input[idx])))
    {
      output[idx] = op(raw_reference_cast(input[idx]));
    }
  }
}; // struct unary_transform_f

template <class InputIt1, class InputIt2, class OutputIt, class StencilIt, class TransformOp, class Predicate>
struct binary_transform_f
{
  InputIt1 input1;
  InputIt2 input2;
  OutputIt output;
  StencilIt stencil;
  TransformOp op;
  Predicate pred;

  THRUST_FUNCTION
  binary_transform_f(
    InputIt1 input1_, InputIt2 input2_, OutputIt output_, StencilIt stencil_, TransformOp op_, Predicate pred_)
      : input1(input1_)
      , input2(input2_)
      , output(output_)
      , stencil(stencil_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(stencil[idx])))
    {
      output[idx] = op(raw_reference_cast(input1[idx]), raw_reference_cast(input2[idx]));
    }
  }
}; // struct binary_transform_stencil_f

template <class InputIt1, class InputIt2, class OutputIt, class TransformOp, class Predicate>
struct binary_transform_f<InputIt1, InputIt2, OutputIt, no_stencil_tag, TransformOp, Predicate>
{
  InputIt1 input1;
  InputIt2 input2;
  OutputIt output;
  TransformOp op;
  Predicate pred;

  THRUST_FUNCTION
  binary_transform_f(
    InputIt1 input1_, InputIt2 input2_, OutputIt output_, no_stencil_tag, TransformOp op_, Predicate pred_)
      : input1(input1_)
      , input2(input2_)
      , output(output_)
      , op(op_)
      , pred(pred_)
  {}

  template <class Size>
  void THRUST_DEVICE_FUNCTION operator()(Size idx)
  {
    if (pred(raw_reference_cast(input1[idx])))
    {
      output[idx] = op(raw_reference_cast(input1[idx]), raw_reference_cast(input2[idx]));
    }
  }
}; // struct binary_transform_f

template <class Policy, class InputIt, class Size, class OutputIt, class StencilIt, class TransformOp, class Predicate>
OutputIt THRUST_FUNCTION unary(
  Policy& policy,
  InputIt items,
  OutputIt result,
  Size num_items,
  StencilIt stencil,
  TransformOp transform_op,
  Predicate predicate)
{
  if (num_items == 0)
  {
    return result;
  }

  using unary_transform_t = unary_transform_f<InputIt, OutputIt, StencilIt, TransformOp, Predicate>;

  cuda_cub::parallel_for(policy, unary_transform_t(items, result, stencil, transform_op, predicate), num_items);

  return result + num_items;
}

template <class Policy,
          class InputIt1,
          class InputIt2,
          class Size,
          class OutputIt,
          class StencilIt,
          class TransformOp,
          class Predicate>
OutputIt THRUST_FUNCTION binary(
  Policy& policy,
  InputIt1 items1,
  InputIt2 items2,
  OutputIt result,
  Size num_items,
  StencilIt stencil,
  TransformOp transform_op,
  Predicate predicate)
{
  if (num_items == 0)
  {
    return result;
  }

  using binary_transform_t = binary_transform_f<InputIt1, InputIt2, OutputIt, StencilIt, TransformOp, Predicate>;

  cuda_cub::parallel_for(
    policy, binary_transform_t(items1, items2, result, stencil, transform_op, predicate), num_items);

  return result + num_items;
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class Offset, class... InputIts, class OutputIt, class TransformOp>
OutputIt THRUST_FUNCTION cub_transform_many(
  execution_policy<Derived>& policy,
  Offset num_items,
  ::cuda::std::tuple<InputIts...> firsts,
  OutputIt result,
  TransformOp transform_op)
{
  if (num_items == 0)
  {
    return result;
  }

  // TODO(bgruber): iterator_reference_t or iterator_value_type?
  constexpr auto requires_stable_address = !can_copy_arguments<TransformOp, iterator_reference_t<InputIts>...>::value;
  using dispatch32_t                     = cub::detail::transform::
    dispatch_t<requires_stable_address, std::int32_t, ::cuda::std::tuple<InputIts...>, OutputIt, TransformOp>;
  using dispatch64_t = cub::detail::transform::
    dispatch_t<requires_stable_address, std::int64_t, ::cuda::std::tuple<InputIts...>, OutputIt, TransformOp>;

  cudaError_t status;
  THRUST_INDEX_TYPE_DISPATCH2(
    status,
    dispatch32_t::dispatch,
    dispatch64_t::dispatch,
    num_items,
    (num_items_fixed, firsts, result, transform_op, cuda_cub::stream(policy)));
  throw_on_error(status, "transform: failed inside CUB");

  status = cuda_cub::synchronize_optional(policy);
  throw_on_error(status, "transform: failed to synchronize");

  return result + num_items;
}

template <typename... Ts, std::size_t... Is>
THRUST_FUNCTION auto
convert_to_std_tuple(tuple<Ts...> t, ::cuda::std::index_sequence<Is...>) -> ::cuda::std::tuple<Ts...>
{
  return ::cuda::std::tuple<Ts...>{get<Is>(t)...};
}

// unwrap zip_iterator and zip_function
template <class Derived, class Offset, class... InputIts, class OutputIt, class TransformOp>
OutputIt THRUST_FUNCTION cub_transform_many(
  execution_policy<Derived>& policy,
  Offset num_items,
  ::cuda::std::tuple<zip_iterator<tuple<InputIts...>>> firsts,
  OutputIt result,
  zip_function<TransformOp> transform_op)
{
  return cub_transform_many(
    policy,
    num_items,
    convert_to_std_tuple(get<0>(firsts).get_iterator_tuple(), ::cuda::std::index_sequence_for<InputIts...>{}),
    result,
    transform_op.underlying_function());
}
} // namespace __transform

//-------------------------
// Thrust API entry points
//-------------------------

//-------------------------
//  one input data stream
//-------------------------

template <class Derived, class InputIt, class OutputIt, class StencilInputIt, class TransformOp, class Predicate>
OutputIt THRUST_FUNCTION transform_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilInputIt stencil,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  using size_type     = typename iterator_traits<InputIt>::difference_type;
  size_type num_items = static_cast<size_type>(thrust::distance(first, last));
  return __transform::unary(policy, first, result, num_items, stencil, transform_op, predicate);
} // func transform_if

template <class Derived, class InputIt, class OutputIt, class TransformOp, class Predicate>
OutputIt THRUST_FUNCTION transform_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  return cuda_cub::transform_if(policy, first, last, __transform::no_stencil_tag(), result, transform_op, predicate);
} // func transform_if

template <class Derived, class InputIt, class OutputIt, class TransformOp>
OutputIt THRUST_FUNCTION
transform(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, TransformOp transform_op)
{
  THRUST_CDP_DISPATCH(
    (using size_type      = typename iterator_traits<InputIt>::difference_type;
     const auto num_items = static_cast<size_type>(thrust::distance(first, last));
     return __transform::cub_transform_many(policy, num_items, ::cuda::std::make_tuple(first), result, transform_op);),
    (while (first != last) {
      *result = transform_op(raw_reference_cast(*first));
      ++first;
      ++result;
    } return result;));
} // func transform

//-------------------------
// two input data streams
//-------------------------

template <class Derived,
          class InputIt1,
          class InputIt2,
          class StencilInputIt,
          class OutputIt,
          class TransformOp,
          class Predicate>
OutputIt THRUST_FUNCTION transform_if(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  StencilInputIt stencil,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  using size_type     = typename iterator_traits<InputIt1>::difference_type;
  size_type num_items = static_cast<size_type>(thrust::distance(first1, last1));
  return __transform::binary(policy, first1, first2, result, num_items, stencil, transform_op, predicate);
} // func transform_if

template <class Derived, class InputIt1, class InputIt2, class OutputIt, class TransformOp>
OutputIt THRUST_FUNCTION transform(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  OutputIt result,
  TransformOp transform_op)
{
  THRUST_CDP_DISPATCH(
    (using size_type      = typename iterator_traits<InputIt1>::difference_type;
     const auto num_items = static_cast<size_type>(thrust::distance(first1, last1));
     return __transform::cub_transform_many(
       policy, num_items, ::cuda::std::make_tuple(first1, first2), result, transform_op);),
    (while (first1 != last1) {
      *result = transform_op(raw_reference_cast(*first1), raw_reference_cast(*first2));
      ++first1;
      ++first2;
      ++result;
    } return result;));
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
