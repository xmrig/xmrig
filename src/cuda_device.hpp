#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <string>


#define CUDA_THROW(error) throw std::runtime_error(std::string("<") + __FUNCTION__ + ">:" + std::to_string(__LINE__) + " \"" + (error) + "\"")


/** execute and check a CUDA api command
*
* @param id gpu id (thread id)
* @param ... CUDA api command
*/
#define CUDA_CHECK(id, ...) {                                                                             \
    cudaError_t error = __VA_ARGS__;                                                                      \
    if (error != cudaSuccess){                                                                            \
        CUDA_THROW(cudaGetErrorString(error));                                                            \
    }                                                                                                     \
}                                                                                                         \
( (void) 0 )

/** execute and check a CUDA kernel
*
* @param id gpu id (thread id)
* @param ... CUDA kernel call
*/
#define CUDA_CHECK_KERNEL(id, ...)      \
    __VA_ARGS__;                        \
    CUDA_CHECK(id, cudaGetLastError())

#if defined(XMRIG_ALGO_KAWPOW) || defined(XMRIG_ALGO_CN_R)
#define CU_CHECK(id, ...) {                                                                             \
    CUresult result = __VA_ARGS__;                                                                      \
    if(result != CUDA_SUCCESS){                                                                         \
        const char* s;                                                                                  \
        cuGetErrorString(result, &s);                                                                   \
        CUDA_THROW(s ? s : "unknown error");                                                            \
    }                                                                                                   \
}                                                                                                       \
( (void) 0 )
#endif
