#pragma once

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

__global__ void find_shares(const void* hashes, uint64_t target, uint32_t* shares)
{
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t* p = (const uint64_t*)hashes;

    if (p[global_index * 4 + 3] < target) {
        const uint32_t idx = atomicInc(shares, 0xFFFFFFFF) + 1;
        if (idx < 10) {
            shares[idx] = global_index;
        }
    }
}

void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size)
{
    CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce));
    CUDA_CHECK_KERNEL(ctx->device_id, fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_long_state, batch_size));
    CUDA_CHECK(ctx->device_id, cudaMemset(ctx->d_rx_rounding, 0, batch_size * sizeof(uint32_t)));

    for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i) {
        CUDA_CHECK_KERNEL(ctx->device_id, fillAes4Rx4<ENTROPY_SIZE, false><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_rx_entropy, batch_size));

        CUDA_CHECK_KERNEL(ctx->device_id, init_vm<8><<<batch_size / 4, 4 * 8>>>(ctx->d_rx_entropy, ctx->d_rx_vm_states));
        for (int j = 0, n = 1 << ctx->device_bfactor; j < n; ++j) {
            CUDA_CHECK_KERNEL(ctx->device_id, execute_vm<8, false><<<batch_size / 2, 2 * 8>>>(ctx->d_rx_vm_states, ctx->d_rx_rounding, ctx->d_long_state, ctx->d_rx_dataset, batch_size, RANDOMX_PROGRAM_ITERATIONS >> ctx->device_bfactor, j == 0, j == n - 1));
        }

        if (i == RANDOMX_PROGRAM_COUNT - 1) {
            CUDA_CHECK_KERNEL(ctx->device_id, hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, VM_STATE_SIZE, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_long_state, ctx->d_rx_vm_states, batch_size));
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 32><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        } else {
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 64><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        }
    }

    CUDA_CHECK(ctx->device_id, cudaMemset(ctx->d_result_nonce, 0, 10 * sizeof(uint32_t)));
    CUDA_CHECK_KERNEL(ctx->device_id, find_shares<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, target, ctx->d_result_nonce));
    CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());

    CUDA_CHECK(ctx->device_id, cudaMemcpy(resnonce, ctx->d_result_nonce, 10 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    *rescount = resnonce[0];
    if (*rescount > 9) {
        *rescount = 9;
    }

    for (uint32_t i = 0; i < *rescount; i++) {
        resnonce[i] = resnonce[i + 1] + nonce;
    }
}
