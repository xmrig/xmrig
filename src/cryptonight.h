/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2019      Spudz76     <https://github.com/Spudz76>
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


#pragma once

#include "crypto/common/Algorithm.h"


#include <cstdint>


#if defined(XMRIG_ALGO_KAWPOW) || defined(XMRIG_ALGO_CN_R)
#   include <cuda.h>
#endif


struct nvid_ctx {
#   ifdef XMRIG_ALGO_CN_R
    CUdevice cuDevice                   = -1;
    CUmodule module                     = nullptr;
    CUfunction kernel                   = nullptr;
#   endif

    xmrig_cuda::Algorithm algorithm     = xmrig_cuda::Algorithm::INVALID;
    uint64_t kernel_height              = 0;

    int device_id                       = 0;
    const char *device_name             = nullptr;
    int device_arch[2]                  { 0,};
    int device_mpcount                  = 0;
    int device_blocks                   = 0;
    int device_threads                  = 0;
    int device_bfactor                  = 0;
    int device_bsleep                   = 0;
    int device_clockRate                = 0;
    int device_memoryClockRate          = 0;
    size_t device_memoryTotal           = 0;
    size_t device_memoryFree            = 0;
    int device_pciBusID                 = 0;
    int device_pciDeviceID              = 0;
    int device_pciDomainID              = 0;
    uint32_t syncMode                   = 3;
    bool ready                          = false;

    uint32_t *d_input                   = nullptr;
    int inputlen                        = 0;
    uint32_t *d_result_count            = nullptr;
    uint32_t *d_result_nonce            = nullptr;
    uint32_t *d_long_state              = nullptr;
    uint64_t d_scratchpads_size         = 0;
    uint32_t *d_ctx_state               = nullptr;
    uint32_t *d_ctx_state2              = nullptr;
    uint32_t *d_ctx_a                   = nullptr;
    uint32_t *d_ctx_b                   = nullptr;
    uint32_t *d_ctx_key1                = nullptr;
    uint32_t *d_ctx_key2                = nullptr;
    uint32_t *d_ctx_text                = nullptr;

    uint32_t rx_batch_size              = 0;
    int32_t rx_dataset_host             = -1;
    uint32_t *d_rx_dataset              = nullptr;
    uint32_t *d_rx_hashes               = nullptr;
    uint32_t *d_rx_entropy              = nullptr;
    uint32_t *d_rx_vm_states            = nullptr;
    uint32_t *d_rx_rounding             = nullptr;

    uint32_t astrobwt_intensity         = 0;
    uint32_t astrobwt_batch1_size       = 0;
    uint32_t astrobwt_processed_hashes  = 0;
    void* astrobwt_salsa20_keys         = nullptr;
    void* astrobwt_bwt_data             = nullptr;
    void* astrobwt_bwt_data_sizes       = nullptr;
    void* astrobwt_indices              = nullptr;
    void* astrobwt_tmp_indices          = nullptr;
    void* astrobwt_filtered_hashes      = nullptr;
    void* astrobwt_shares               = nullptr;
    void* astrobwt_offsets_begin        = nullptr;
    void* astrobwt_offsets_end          = nullptr;

#   ifdef XMRIG_ALGO_KAWPOW
    void* kawpow_cache                  = nullptr;
    size_t kawpow_cache_size            = 0;
    size_t kawpow_cache_capacity        = 0;

    void* kawpow_dag                    = nullptr;
    size_t kawpow_dag_size              = 0;
    size_t kawpow_dag_capacity          = 0;

    uint32_t* kawpow_stop_host          = nullptr;
    uint32_t* kawpow_stop_device        = nullptr;

    uint32_t kawpow_period              = 0;

    CUmodule kawpow_module              = nullptr;
    CUfunction kawpow_kernel            = nullptr;
#   endif
};


int cuda_get_devicecount();
int cuda_get_runtime_version();
int cuda_get_driver_version();
int cuda_get_deviceinfo(nvid_ctx *ctx);
int cryptonight_gpu_init(nvid_ctx *ctx);
void cryptonight_extra_cpu_set_data(nvid_ctx *ctx, const void *data, size_t len);
void cryptonight_extra_cpu_prepare(nvid_ctx *ctx, uint32_t startNonce, const xmrig_cuda::Algorithm &algorithm);
void cryptonight_gpu_hash(nvid_ctx *ctx, const xmrig_cuda::Algorithm &algorithm, uint64_t height, uint32_t startNonce);
void cryptonight_extra_cpu_final(nvid_ctx *ctx, uint32_t startNonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, const xmrig_cuda::Algorithm &algorithm);

void cuda_extra_cpu_set_data(nvid_ctx *ctx, const void *data, size_t len);
void randomx_prepare(nvid_ctx *ctx, const void *dataset, size_t dataset_size, uint32_t batch_size);

namespace RandomX_Arqma   { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Monero  { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Wownero { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Keva    { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }
namespace RandomX_Graft   { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size); }

void astrobwt_prepare(nvid_ctx *ctx, uint32_t batch_size);

namespace AstroBWT_Dero   { void hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce); }

#ifdef XMRIG_ALGO_KAWPOW
void kawpow_prepare(nvid_ctx *ctx, const void* cache, size_t cache_size, const void* dag_precalc, size_t dag_size, uint32_t height, const uint64_t* dag_sizes);
void kawpow_stop_hash(nvid_ctx *ctx);

namespace KawPow_Raven    { void hash(nvid_ctx *ctx, uint8_t* job_blob, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t *skipped_hashes); }
#endif
