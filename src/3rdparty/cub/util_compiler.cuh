/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Detect compiler information.
 */

#pragma once

// enumerate host compilers we know about
#define CUB_HOST_COMPILER_UNKNOWN 0
#define CUB_HOST_COMPILER_MSVC 1
#define CUB_HOST_COMPILER_GCC 2
#define CUB_HOST_COMPILER_CLANG 3

// enumerate device compilers we know about
#define CUB_DEVICE_COMPILER_UNKNOWN 0
#define CUB_DEVICE_COMPILER_MSVC 1
#define CUB_DEVICE_COMPILER_GCC 2
#define CUB_DEVICE_COMPILER_NVCC 3
#define CUB_DEVICE_COMPILER_CLANG 4

// figure out which host compiler we're using
#if defined(_MSC_VER)
#  define CUB_HOST_COMPILER CUB_HOST_COMPILER_MSVC
#  define CUB_MSVC_VERSION _MSC_VER
#  define CUB_MSVC_VERSION_FULL _MSC_FULL_VER
#elif defined(__clang__)
#  define CUB_HOST_COMPILER CUB_HOST_COMPILER_CLANG
#  define CUB_CLANG_VERSION                                                    \
    (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#  define CUB_HOST_COMPILER CUB_HOST_COMPILER_GCC
#  define CUB_GCC_VERSION                                                      \
    (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
#  define CUB_HOST_COMPILER CUB_HOST_COMPILER_UNKNOWN
#endif // CUB_HOST_COMPILER

// figure out which device compiler we're using
#if defined(__CUDACC__)
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_NVCC
#elif CUB_HOST_COMPILER == CUB_HOST_COMPILER_MSVC
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_MSVC
#elif CUB_HOST_COMPILER == CUB_HOST_COMPILER_GCC
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_GCC
#elif CUB_HOST_COMPILER == CUB_HOST_COMPILER_CLANG
// CUDA-capable clang should behave similar to NVCC.
#  if defined(__CUDA__)
#    define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_NVCC
#  else
#    define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_CLANG
#  endif
#else
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_UNKNOWN
#endif
