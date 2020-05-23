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


#include "backend/cuda/wrappers/CudaDevice.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/cuda/CudaThreads.h"
#include "backend/cuda/wrappers/CudaLib.h"
#include "base/crypto/Algorithm.h"
#include "base/io/log/Log.h"


#ifdef XMRIG_FEATURE_NVML
#   include "backend/cuda/wrappers/NvmlLib.h"
#endif

#include <algorithm>


xmrig::CudaDevice::CudaDevice(uint32_t index, int32_t bfactor, int32_t bsleep) :
    m_index(index)
{
    auto ctx = CudaLib::alloc(index, bfactor, bsleep);
    if (!CudaLib::deviceInfo(ctx, 0, 0, Algorithm::INVALID)) {
        CudaLib::release(ctx);

        return;
    }

    m_ctx       = ctx;
    m_name      = CudaLib::deviceName(ctx);
    m_topology  = PciTopology(CudaLib::deviceUint(ctx, CudaLib::DevicePciBusID), CudaLib::deviceUint(ctx, CudaLib::DevicePciDeviceID), 0);
}


xmrig::CudaDevice::CudaDevice(CudaDevice &&other) noexcept :
    m_index(other.m_index),
    m_ctx(other.m_ctx),
    m_topology(other.m_topology),
    m_name(std::move(other.m_name))
{
    other.m_ctx = nullptr;
}


xmrig::CudaDevice::~CudaDevice()
{
    CudaLib::release(m_ctx);
}


size_t xmrig::CudaDevice::freeMemSize() const
{
    return CudaLib::deviceUlong(m_ctx, CudaLib::DeviceMemoryFree);
}


size_t xmrig::CudaDevice::globalMemSize() const
{
    return CudaLib::deviceUlong(m_ctx, CudaLib::DeviceMemoryTotal);
}


uint32_t xmrig::CudaDevice::clock() const
{
    return CudaLib::deviceUint(m_ctx, CudaLib::DeviceClockRate) / 1000;
}


uint32_t xmrig::CudaDevice::computeCapability(bool major) const
{
    return CudaLib::deviceUint(m_ctx, major ? CudaLib::DeviceArchMajor : CudaLib::DeviceArchMinor);
}


uint32_t xmrig::CudaDevice::memoryClock() const
{
    return CudaLib::deviceUint(m_ctx, CudaLib::DeviceMemoryClockRate) / 1000;
}


uint32_t xmrig::CudaDevice::smx() const
{
    return CudaLib::deviceUint(m_ctx, CudaLib::DeviceSmx);
}


void xmrig::CudaDevice::generate(const Algorithm &algorithm, CudaThreads &threads) const
{
    if (!CudaLib::deviceInfo(m_ctx, -1, -1, algorithm)) {
        return;
    }

    threads.add(CudaThread(m_index, m_ctx));
}


#ifdef XMRIG_FEATURE_API
void xmrig::CudaDevice::toJSON(rapidjson::Value &out, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    out.AddMember("name",           name().toJSON(doc), allocator);
    out.AddMember("bus_id",         topology().toString().toJSON(doc), allocator);
    out.AddMember("smx",            smx(), allocator);
    out.AddMember("arch",           arch(), allocator);
    out.AddMember("global_mem",     static_cast<uint64_t>(globalMemSize()), allocator);
    out.AddMember("clock",          clock(), allocator);
    out.AddMember("memory_clock",   memoryClock(), allocator);

#   ifdef XMRIG_FEATURE_NVML
    if (m_nvmlDevice) {
        auto data = NvmlLib::health(m_nvmlDevice);

        Value health(kObjectType);
        health.AddMember("temperature", data.temperature, allocator);
        health.AddMember("power",       data.power, allocator);
        health.AddMember("clock",       data.clock, allocator);
        health.AddMember("mem_clock",   data.memClock, allocator);

        Value fanSpeed(kArrayType);
        for (auto speed : data.fanSpeed) {
            fanSpeed.PushBack(speed, allocator);
        }
        health.AddMember("fan_speed", fanSpeed, allocator);

        out.AddMember("health", health, allocator);
    }
#   endif
}
#endif
