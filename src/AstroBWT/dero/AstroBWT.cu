/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include "cryptonight.h"
#include "cuda_device.hpp"


#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ void sync()
{
#	if (__CUDACC_VER_MAJOR__ >= 9)
    __syncwarp();
#	else
    __syncthreads();
#	endif
}

#include "sha3.h"
#include "salsa20.h"
#include "BWT.h"


#if (__CUDACC_VER_MAJOR__ >= 11)
#include <cub/device/device_segmented_radix_sort.cuh>
#else
#include "3rdparty/cub/device/device_segmented_radix_sort.cuh"
#endif


static constexpr uint32_t BWT_DATA_MAX_SIZE = 560 * 1024 - 256;
static constexpr uint32_t BWT_DATA_STRIDE = (BWT_DATA_MAX_SIZE + 256 + 255) & ~255U;
static constexpr uint32_t STAGE1_DATA_STRIDE = (AstroBWT_Dero::STAGE1_SIZE + 256 + 255) & ~255U;


void astrobwt_prepare(nvid_ctx *ctx, uint32_t batch_size)
{
    if (batch_size != ctx->astrobwt_intensity) {
        ctx->astrobwt_intensity = batch_size;

        const uint32_t BATCH2_SIZE = batch_size;
        const uint32_t BWT_ALLOCATION_SIZE = BATCH2_SIZE * BWT_DATA_STRIDE;
        const uint32_t BATCH1_SIZE = (BWT_ALLOCATION_SIZE / STAGE1_DATA_STRIDE) & ~255U;

        ctx->astrobwt_batch1_size = BATCH1_SIZE;

        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_salsa20_keys));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_bwt_data));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_bwt_data_sizes));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_indices));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_tmp_indices));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_filtered_hashes));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_shares));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_offsets_begin));
        CUDA_CHECK(ctx->device_id, cudaFree(ctx->astrobwt_offsets_end));

        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_salsa20_keys, BATCH1_SIZE * 32));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_bwt_data, BWT_ALLOCATION_SIZE));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_bwt_data_sizes, BATCH1_SIZE * sizeof(uint32_t)));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_indices, BWT_ALLOCATION_SIZE * sizeof(uint64_t)));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_tmp_indices, BWT_ALLOCATION_SIZE * sizeof(uint64_t) + 65536));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_filtered_hashes, (BATCH1_SIZE + BATCH2_SIZE) * (sizeof(uint32_t) + 32) + sizeof(uint32_t)));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_shares, 11 * sizeof(uint32_t)));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_offsets_begin, BATCH1_SIZE * sizeof(uint32_t)));
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->astrobwt_offsets_end, BATCH1_SIZE * sizeof(uint32_t)));
    }

    const uint32_t zero = 0;
    CUDA_CHECK(ctx->device_id, cudaMemcpy((uint8_t*)(ctx->astrobwt_filtered_hashes), &zero, sizeof(zero), cudaMemcpyHostToDevice));
}

namespace AstroBWT_Dero {

template<uint32_t DATA_STRIDE>
static void BWT(nvid_ctx *ctx, uint32_t global_work_size, uint32_t BWT_ALLOCATION_SIZE)
{
    uint32_t* keys_in = (uint32_t*)ctx->astrobwt_indices;
    uint16_t* values_in = (uint16_t*)(keys_in + BWT_ALLOCATION_SIZE);

    CUDA_CHECK_KERNEL(ctx->device_id, BWT_preprocess<<<global_work_size, 1024>>>(
        (uint8_t*)ctx->astrobwt_bwt_data,
        (uint32_t*)ctx->astrobwt_bwt_data_sizes,
        DATA_STRIDE,
        keys_in,
        values_in,
        (uint32_t*)ctx->astrobwt_offsets_begin,
        (uint32_t*)ctx->astrobwt_offsets_end
    ));

    size_t temp_storage_bytes = BWT_ALLOCATION_SIZE * sizeof(uint64_t) + 65536;
    cub::DeviceSegmentedRadixSort::SortPairs(
        ctx->astrobwt_tmp_indices,
        temp_storage_bytes,
        keys_in,
        keys_in,
        values_in,
        values_in,
        global_work_size * DATA_STRIDE,
        global_work_size,
        (uint32_t*)ctx->astrobwt_offsets_begin,
        (uint32_t*)ctx->astrobwt_offsets_end,
        8,
        32
    );

    CUDA_CHECK_KERNEL(ctx->device_id, BWT_fix_order<8><<<global_work_size, 1024>>>(
        (uint8_t*)ctx->astrobwt_bwt_data,
        (uint32_t*)ctx->astrobwt_bwt_data_sizes,
        DATA_STRIDE,
        keys_in,
        values_in
    ));

    CUDA_CHECK_KERNEL(ctx->device_id, BWT_apply<DATA_STRIDE><<<(global_work_size * DATA_STRIDE) / 1024, 1024>>>(
        (uint8_t*)ctx->astrobwt_bwt_data,
        (uint32_t*)ctx->astrobwt_bwt_data_sizes,
        keys_in,
        values_in,
        (uint64_t*)ctx->astrobwt_tmp_indices
    ));
}

void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce)
{
    const uint32_t BATCH1_SIZE = ctx->astrobwt_batch1_size;
    const uint32_t BATCH2_SIZE = ctx->astrobwt_intensity;
    const uint32_t BWT_ALLOCATION_SIZE = BATCH2_SIZE * BWT_DATA_STRIDE;

    const uint32_t zero = 0;
    CUDA_CHECK(ctx->device_id, cudaMemcpy((uint8_t*)(ctx->astrobwt_shares), &zero, sizeof(zero), cudaMemcpyHostToDevice));

    uint32_t global_work_size = BATCH1_SIZE;

    CUDA_CHECK_KERNEL(ctx->device_id, sha3_initial<<<global_work_size, 32>>>((uint8_t*)ctx->d_input, ctx->inputlen, nonce, (uint64_t*)ctx->astrobwt_salsa20_keys));

    std::vector<uint32_t> bwt_data_sizes(BATCH1_SIZE, STAGE1_SIZE);
    CUDA_CHECK(ctx->device_id, cudaMemcpy((uint8_t*)(ctx->astrobwt_bwt_data_sizes), bwt_data_sizes.data(), bwt_data_sizes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK_KERNEL(ctx->device_id, Salsa20_XORKeyStream<<<global_work_size, 32>>>((uint32_t*)ctx->astrobwt_salsa20_keys, (uint32_t*)ctx->astrobwt_bwt_data, (uint32_t*)ctx->astrobwt_bwt_data_sizes, STAGE1_DATA_STRIDE));
    BWT<STAGE1_DATA_STRIDE>(ctx, global_work_size, BWT_ALLOCATION_SIZE);
    CUDA_CHECK_KERNEL(ctx->device_id, sha3<<<global_work_size, 32>>>((uint8_t*)ctx->astrobwt_tmp_indices, (uint32_t*)ctx->astrobwt_bwt_data_sizes, STAGE1_DATA_STRIDE * 8, (uint64_t*)ctx->astrobwt_salsa20_keys));
    CUDA_CHECK_KERNEL(ctx->device_id, filter<<<global_work_size / 32, 32>>>(nonce, BWT_DATA_MAX_SIZE, (uint32_t*)ctx->astrobwt_salsa20_keys, (uint32_t*)ctx->astrobwt_filtered_hashes));

    CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());

    uint32_t num_filtered_hashes;
    CUDA_CHECK(ctx->device_id, cudaMemcpy(&num_filtered_hashes, ctx->astrobwt_filtered_hashes, sizeof(num_filtered_hashes), cudaMemcpyDeviceToHost));

    ctx->astrobwt_processed_hashes = 0;
    while (num_filtered_hashes >= BATCH2_SIZE)
    {
        num_filtered_hashes -= BATCH2_SIZE;
        ctx->astrobwt_processed_hashes += BATCH2_SIZE;

        global_work_size = BATCH2_SIZE;

        CUDA_CHECK_KERNEL(ctx->device_id, prepare_batch2<<<global_work_size / 32, 32>>>((uint32_t*)ctx->astrobwt_salsa20_keys, (uint32_t*)ctx->astrobwt_filtered_hashes, (uint32_t*)ctx->astrobwt_bwt_data_sizes));
        CUDA_CHECK_KERNEL(ctx->device_id, Salsa20_XORKeyStream<<<global_work_size, 32>>>((uint32_t*)ctx->astrobwt_salsa20_keys, (uint32_t*)ctx->astrobwt_bwt_data, (uint32_t*)ctx->astrobwt_bwt_data_sizes, BWT_DATA_STRIDE));
        BWT<BWT_DATA_STRIDE>(ctx, global_work_size, BWT_ALLOCATION_SIZE);
        CUDA_CHECK_KERNEL(ctx->device_id, sha3<<<global_work_size, 32>>>((uint8_t*)ctx->astrobwt_tmp_indices, (uint32_t*)ctx->astrobwt_bwt_data_sizes, BWT_DATA_STRIDE * 8, (uint64_t*)ctx->astrobwt_salsa20_keys));
        CUDA_CHECK_KERNEL(ctx->device_id, find_shares<<<global_work_size / 32, 32>>>((uint64_t*)ctx->astrobwt_salsa20_keys, (uint32_t*)ctx->astrobwt_filtered_hashes, target, (uint32_t*)ctx->astrobwt_shares));
    }

    CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());

    uint32_t shares[11];
    CUDA_CHECK(ctx->device_id, cudaMemcpy(shares, ctx->astrobwt_shares, sizeof(shares), cudaMemcpyDeviceToHost));

    if (shares[0] > 10)
        shares[0] = 10;

    *rescount = shares[0];
    memcpy(resnonce, shares + 1, shares[0] * sizeof(uint32_t));
}

}
