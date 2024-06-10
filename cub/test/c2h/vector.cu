/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "vector.cuh"

THRUST_NAMESPACE_BEGIN
namespace detail
{
#define DEFINE_EXTERN_TEMPLATE(type)                                     \
  template class vector_base<type, ::c2h::checked_host_allocator<type>>; \
  template class vector_base<type, ::c2h::checked_cuda_allocator<type>>;

DEFINE_EXTERN_TEMPLATE(signed char);
DEFINE_EXTERN_TEMPLATE(unsigned char);

DEFINE_EXTERN_TEMPLATE(short);
DEFINE_EXTERN_TEMPLATE(unsigned short);
DEFINE_EXTERN_TEMPLATE(int);
DEFINE_EXTERN_TEMPLATE(unsigned int);
DEFINE_EXTERN_TEMPLATE(long);
DEFINE_EXTERN_TEMPLATE(unsigned long);
DEFINE_EXTERN_TEMPLATE(long long);
DEFINE_EXTERN_TEMPLATE(unsigned long long);

DEFINE_EXTERN_TEMPLATE(float);
DEFINE_EXTERN_TEMPLATE(double);

#undef DEFINE_EXTERN_TEMPLATE
} // namespace detail
THRUST_NAMESPACE_END
