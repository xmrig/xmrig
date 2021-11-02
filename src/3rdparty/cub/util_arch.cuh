/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \file
 * Static architectural properties by SM version.
 */

#pragma once

#include "util_cpp_dialect.cuh"
#include "util_namespace.cuh"
#include "util_macro.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

#if ((__CUDACC_VER_MAJOR__ >= 9) || defined(__NVCOMPILER_CUDA__)) && \
        !defined(CUB_USE_COOPERATIVE_GROUPS)
    #define CUB_USE_COOPERATIVE_GROUPS
#endif

/// In device code, CUB_PTX_ARCH expands to the PTX version for which we are
/// compiling. In host code, CUB_PTX_ARCH's value is implementation defined.
#ifndef CUB_PTX_ARCH
    #if defined(__NVCOMPILER_CUDA__)
        // __NVCOMPILER_CUDA_ARCH__ is the target PTX version, and is defined
        // when compiling both host code and device code. Currently, only one
        // PTX version can be targeted.
        #define CUB_PTX_ARCH __NVCOMPILER_CUDA_ARCH__
    #elif !defined(__CUDA_ARCH__)
        #define CUB_PTX_ARCH 0
    #else
        #define CUB_PTX_ARCH __CUDA_ARCH__
    #endif
#endif

#ifndef CUB_IS_DEVICE_CODE
    #if defined(__NVCOMPILER_CUDA__)
        #define CUB_IS_DEVICE_CODE __builtin_is_device_code()
        #define CUB_IS_HOST_CODE (!__builtin_is_device_code())
        #define CUB_INCLUDE_DEVICE_CODE 1
        #define CUB_INCLUDE_HOST_CODE 1
    #elif CUB_PTX_ARCH > 0
        #define CUB_IS_DEVICE_CODE 1
        #define CUB_IS_HOST_CODE 0
        #define CUB_INCLUDE_DEVICE_CODE 1
        #define CUB_INCLUDE_HOST_CODE 0
    #else
        #define CUB_IS_DEVICE_CODE 0
        #define CUB_IS_HOST_CODE 1
        #define CUB_INCLUDE_DEVICE_CODE 0
        #define CUB_INCLUDE_HOST_CODE 1
    #endif
#endif

/// Maximum number of devices supported.
#ifndef CUB_MAX_DEVICES
    #define CUB_MAX_DEVICES 128
#endif

#if CUB_CPP_DIALECT >= 2011
    static_assert(CUB_MAX_DEVICES > 0, "CUB_MAX_DEVICES must be greater than 0.");
#endif

/// Whether or not the source targeted by the active compiler pass is allowed to  invoke device kernels or methods from the CUDA runtime API.
#ifndef CUB_RUNTIME_FUNCTION
    #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
        #define CUB_RUNTIME_ENABLED
        #define CUB_RUNTIME_FUNCTION __host__ __device__
    #else
        #define CUB_RUNTIME_FUNCTION __host__
    #endif
#endif


/// Number of threads per warp
#ifndef CUB_LOG_WARP_THREADS
    #define CUB_LOG_WARP_THREADS(arch)                      \
        (5)
    #define CUB_WARP_THREADS(arch)                          \
        (1 << CUB_LOG_WARP_THREADS(arch))

    #define CUB_PTX_WARP_THREADS        CUB_WARP_THREADS(CUB_PTX_ARCH)
    #define CUB_PTX_LOG_WARP_THREADS    CUB_LOG_WARP_THREADS(CUB_PTX_ARCH)
#endif


/// Number of smem banks
#ifndef CUB_LOG_SMEM_BANKS
    #define CUB_LOG_SMEM_BANKS(arch)                        \
        ((arch >= 200) ?                                    \
            (5) :                                           \
            (4))
    #define CUB_SMEM_BANKS(arch)                            \
        (1 << CUB_LOG_SMEM_BANKS(arch))

    #define CUB_PTX_LOG_SMEM_BANKS      CUB_LOG_SMEM_BANKS(CUB_PTX_ARCH)
    #define CUB_PTX_SMEM_BANKS          CUB_SMEM_BANKS(CUB_PTX_ARCH)
#endif


/// Oversubscription factor
#ifndef CUB_SUBSCRIPTION_FACTOR
    #define CUB_SUBSCRIPTION_FACTOR(arch)                   \
        ((arch >= 300) ?                                    \
            (5) :                                           \
            ((arch >= 200) ?                                \
                (3) :                                       \
                (10)))
    #define CUB_PTX_SUBSCRIPTION_FACTOR             CUB_SUBSCRIPTION_FACTOR(CUB_PTX_ARCH)
#endif


/// Prefer padding overhead vs X-way conflicts greater than this threshold
#ifndef CUB_PREFER_CONFLICT_OVER_PADDING
    #define CUB_PREFER_CONFLICT_OVER_PADDING(arch)          \
        ((arch >= 300) ?                                    \
            (1) :                                           \
            (4))
    #define CUB_PTX_PREFER_CONFLICT_OVER_PADDING    CUB_PREFER_CONFLICT_OVER_PADDING(CUB_PTX_ARCH)
#endif


template <
    int NOMINAL_4B_BLOCK_THREADS,
    int NOMINAL_4B_ITEMS_PER_THREAD,
    typename T>
struct RegBoundScaling
{
    enum {
        ITEMS_PER_THREAD    = CUB_MAX(1, NOMINAL_4B_ITEMS_PER_THREAD * 4 / CUB_MAX(4, sizeof(T))),
        BLOCK_THREADS       = CUB_MIN(NOMINAL_4B_BLOCK_THREADS, (((1024 * 48) / (sizeof(T) * ITEMS_PER_THREAD)) + 31) / 32 * 32),
    };
};


template <
    int NOMINAL_4B_BLOCK_THREADS,
    int NOMINAL_4B_ITEMS_PER_THREAD,
    typename T>
struct MemBoundScaling
{
    enum {
        ITEMS_PER_THREAD    = CUB_MAX(1, CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T), NOMINAL_4B_ITEMS_PER_THREAD * 2)),
        BLOCK_THREADS       = CUB_MIN(NOMINAL_4B_BLOCK_THREADS, (((1024 * 48) / (sizeof(T) * ITEMS_PER_THREAD)) + 31) / 32 * 32),
    };
};




#endif  // Do not document

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
