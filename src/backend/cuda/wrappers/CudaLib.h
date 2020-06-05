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

#ifndef XMRIG_CUDALIB_H
#define XMRIG_CUDALIB_H


using nvid_ctx = struct nvid_ctx;


#include "backend/cuda/wrappers/CudaDevice.h"
#include "base/tools/String.h"


#include <vector>
#include <string>


namespace xmrig {


class CudaLib
{
public:
    enum DeviceProperty : uint32_t
    {
        DeviceId,
        DeviceAlgorithm,
        DeviceArchMajor,
        DeviceArchMinor,
        DeviceSmx,
        DeviceBlocks,
        DeviceThreads,
        DeviceBFactor,
        DeviceBSleep,
        DeviceClockRate,
        DeviceMemoryClockRate,
        DeviceMemoryTotal,
        DeviceMemoryFree,
        DevicePciBusID,
        DevicePciDeviceID,
        DevicePciDomainID,
        DeviceDatasetHost,
        DeviceAstroBWTProcessedHashes,
    };

    static bool init(const char *fileName = nullptr);
    static const char *lastError() noexcept;
    static void close();

    static inline bool isInitialized()    { return m_initialized; }
    static inline bool isReady() noexcept { return m_ready; }
    static inline const String &loader()  { return m_loader; }

    static bool astroBWTHash(nvid_ctx *ctx, uint32_t startNonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce) noexcept;
    static bool astroBWTPrepare(nvid_ctx *ctx, uint32_t batchSize) noexcept;
    static bool cnHash(nvid_ctx *ctx, uint32_t startNonce, uint64_t height, uint64_t target, uint32_t *rescount, uint32_t *resnonce);
    static bool deviceInfo(nvid_ctx *ctx, int32_t blocks, int32_t threads, const Algorithm &algorithm, int32_t dataset_host = -1) noexcept;
    static bool deviceInit(nvid_ctx *ctx) noexcept;
    static bool rxHash(nvid_ctx *ctx, uint32_t startNonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce) noexcept;
    static bool rxPrepare(nvid_ctx *ctx, const void *dataset, size_t datasetSize, bool dataset_host, uint32_t batchSize) noexcept;
    static bool kawPowHash(nvid_ctx *ctx, uint8_t* job_blob, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t *skipped_hashes) noexcept;
    static bool kawPowPrepare(nvid_ctx *ctx, const void* cache, size_t cache_size, const void* dag_precalc, size_t dag_size, uint32_t height, const uint64_t* dag_sizes) noexcept;
    static bool kawPowStopHash(nvid_ctx *ctx) noexcept;
    static bool setJob(nvid_ctx *ctx, const void *data, size_t size, const Algorithm &algorithm) noexcept;
    static const char *deviceName(nvid_ctx *ctx) noexcept;
    static const char *lastError(nvid_ctx *ctx) noexcept;
    static const char *pluginVersion() noexcept;
    static int32_t deviceInt(nvid_ctx *ctx, DeviceProperty property) noexcept;
    static nvid_ctx *alloc(uint32_t id, int32_t bfactor, int32_t bsleep) noexcept;
    static std::string version(uint32_t version);
    static std::vector<CudaDevice> devices(int32_t bfactor, int32_t bsleep, const std::vector<uint32_t> &hints) noexcept;
    static uint32_t deviceCount() noexcept;
    static uint32_t deviceUint(nvid_ctx *ctx, DeviceProperty property) noexcept;
    static uint32_t driverVersion() noexcept;
    static uint32_t runtimeVersion() noexcept;
    static uint64_t deviceUlong(nvid_ctx *ctx, DeviceProperty property) noexcept;
    static void release(nvid_ctx *ctx) noexcept;

private:
    static bool load();
    static String defaultLoader();

    static bool m_initialized;
    static bool m_ready;
    static String m_loader;
};


} // namespace xmrig


#endif /* XMRIG_CUDALIB_H */
