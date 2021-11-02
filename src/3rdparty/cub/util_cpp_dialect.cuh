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

/*! \file
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include "util_compiler.cuh"

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - CUB_IGNORE_DEPRECATED_CPP_DIALECT:
//   Ignore all deprecated C++ dialects and outdated compilers.
// - CUB_IGNORE_DEPRECATED_CPP_11:
//   Ignore deprecation warnings when compiling with C++11. C++03 and outdated
//   compilers will still issue warnings.
// - CUB_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with C++03 and C++11 will still issue warnings.

// Check for the thrust opt-outs as well:
#if !defined(CUB_IGNORE_DEPRECATED_CPP_DIALECT) && \
     defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#  define    CUB_IGNORE_DEPRECATED_CPP_DIALECT
#endif
#if !defined(CUB_IGNORE_DEPRECATED_CPP_11) && \
     defined(THRUST_IGNORE_DEPRECATED_CPP_11)
#  define    CUB_IGNORE_DEPRECATED_CPP_11
#endif
#if !defined(CUB_IGNORE_DEPRECATED_COMPILER) && \
     defined(THRUST_IGNORE_DEPRECATED_COMPILER)
#  define    CUB_IGNORE_DEPRECATED_COMPILER
#endif

#ifdef CUB_IGNORE_DEPRECATED_CPP_DIALECT
#  define CUB_IGNORE_DEPRECATED_CPP_11
#  define CUB_IGNORE_DEPRECATED_COMPILER
#endif

// Define this to override the built-in detection.
#ifndef CUB_CPP_DIALECT

// MSVC does not define __cplusplus correctly. _MSVC_LANG is used instead.
// This macro is only defined in MSVC 2015U3+.
#  ifdef _MSVC_LANG // Do not replace with CUB_HOST_COMPILER test (see above)
// MSVC2015 reports C++14 but lacks extended constexpr support. Treat as C++11.
#    if CUB_MSVC_VERSION < 1910 && _MSVC_LANG > 201103L /* MSVC < 2017 && CPP > 2011 */
#      define CUB_CPLUSPLUS 201103L /* Fix to 2011 */
#    else
#      define CUB_CPLUSPLUS _MSVC_LANG /* We'll trust this for now. */
#    endif // MSVC 2015 C++14 fix
#  else
#    define CUB_CPLUSPLUS __cplusplus
#  endif

// Detect current dialect:
#  if CUB_CPLUSPLUS < 201103L
#    define CUB_CPP_DIALECT 2003
#  elif CUB_CPLUSPLUS < 201402L
#    define CUB_CPP_DIALECT 2011
#  elif CUB_CPLUSPLUS < 201703L
#    define CUB_CPP_DIALECT 2014
#  elif CUB_CPLUSPLUS == 201703L
#    define CUB_CPP_DIALECT 2017
#  elif CUB_CPLUSPLUS > 201703L // unknown, but is higher than 2017.
#    define CUB_CPP_DIALECT 2020
#  endif

#  undef CUB_CPLUSPLUS // cleanup

#endif // !CUB_CPP_DIALECT

// Define CUB_COMPILER_DEPRECATION macro:
#if CUB_HOST_COMPILER == CUB_HOST_COMPILER_MSVC
#  define CUB_COMP_DEPR_IMPL(msg) \
    __pragma(message(__FILE__ ":" CUB_COMP_DEPR_IMPL0(__LINE__) ": warning: " #msg))
#  define CUB_COMP_DEPR_IMPL0(x) CUB_COMP_DEPR_IMPL1(x)
#  define CUB_COMP_DEPR_IMPL1(x) #x
#else // clang / gcc:
#  define CUB_COMP_DEPR_IMPL(msg) CUB_COMP_DEPR_IMPL0(GCC warning #msg)
#  define CUB_COMP_DEPR_IMPL0(expr) _Pragma(#expr)
#  define CUB_COMP_DEPR_IMPL1 /* intentionally blank */
#endif

#define CUB_COMPILER_DEPRECATION(REQ, FIX) \
  CUB_COMP_DEPR_IMPL(CUB requires REQ. Please FIX. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)

// Minimum required compiler checks:
#ifndef CUB_IGNORE_DEPRECATED_COMPILER
#  if CUB_HOST_COMPILER == CUB_HOST_COMPILER_GCC && CUB_GCC_VERSION < 50000
     CUB_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
#  endif
#  if CUB_HOST_COMPILER == CUB_HOST_COMPILER_CLANG && CUB_CLANG_VERSION < 60000
     CUB_COMPILER_DEPRECATION(Clang 6.0, upgrade your compiler);
#  endif
#  if CUB_HOST_COMPILER == CUB_HOST_COMPILER_MSVC && CUB_MSVC_VERSION < 1910
     CUB_COMPILER_DEPRECATION(MSVC 2017, upgrade your compiler);
#  endif
#endif

#if !defined(CUB_IGNORE_DEPRECATED_CPP_DIALECT) && CUB_CPP_DIALECT < 2014 && \
    (CUB_CPP_DIALECT != 2011 || !defined(CUB_IGNORE_DEPRECATED_CPP_11))
  CUB_COMPILER_DEPRECATION(C++14, pass -std=c++14 to your compiler);
#endif

#undef CUB_COMPILER_DEPRECATION
#undef CUB_COMP_DEPR_IMPL
#undef CUB_COMP_DEPR_IMPL0
#undef CUB_COMP_DEPR_IMPL1
