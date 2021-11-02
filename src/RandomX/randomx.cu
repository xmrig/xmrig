/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/


#include "cryptonight.h"
#include "cuda_device.hpp"


#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


void randomx_prepare(nvid_ctx *ctx, const void *dataset, size_t dataset_size, uint32_t batch_size)
{
    ctx->rx_batch_size      = batch_size;
    ctx->d_scratchpads_size = batch_size * (ctx->algorithm.l3() + 64);

    if (ctx->rx_dataset_host > 0) {
        CUDA_CHECK(ctx->device_id, cudaHostGetDevicePointer(&ctx->d_rx_dataset, const_cast<void *>(dataset), 0));
    }
    else {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_dataset, dataset_size));
        CUDA_CHECK(ctx->device_id, cudaMemcpy(ctx->d_rx_dataset, dataset, dataset_size, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_long_state, ctx->d_scratchpads_size));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_hashes, batch_size * 64));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_entropy, batch_size * (128 + 2560)));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_vm_states, batch_size * 2560));
    CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_rounding, batch_size * sizeof(uint32_t)));
}
