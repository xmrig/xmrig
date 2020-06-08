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

#include <algorithm>
#include <cstring>
#include <uv.h>
#include <cinttypes>


#include "core/config/Config.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "crypto/common/Assembly.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "crypto/rx/RxConfig.h"
#endif


#ifdef XMRIG_FEATURE_OPENCL
#   include "backend/opencl/OclConfig.h"
#endif


#ifdef XMRIG_FEATURE_CUDA
#   include "backend/cuda/CudaConfig.h"
#endif


namespace xmrig {

static const char *kCPU     = "cpu";

#ifdef XMRIG_ALGO_RANDOMX
static const char *kRandomX = "randomx";
#endif

#ifdef XMRIG_FEATURE_OPENCL
static const char *kOcl     = "opencl";
#endif

#ifdef XMRIG_FEATURE_CUDA
static const char *kCuda    = "cuda";
#endif


#if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
static const char *kHealthPrintTime = "health-print-time";
#endif


class ConfigPrivate
{
public:
    CpuConfig cpu;

#   ifdef XMRIG_ALGO_RANDOMX
    RxConfig rx;
#   endif

#   ifdef XMRIG_FEATURE_OPENCL
    OclConfig cl;
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    CudaConfig cuda;
#   endif

#   if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
    uint32_t healthPrintTime = 60;
#   endif
};

}


xmrig::Config::Config() :
    d_ptr(new ConfigPrivate())
{
}


xmrig::Config::~Config()
{
    delete d_ptr;
}


const xmrig::CpuConfig &xmrig::Config::cpu() const
{
    return d_ptr->cpu;
}


#ifdef XMRIG_FEATURE_OPENCL
const xmrig::OclConfig &xmrig::Config::cl() const
{
    return d_ptr->cl;
}
#endif


#ifdef XMRIG_FEATURE_CUDA
const xmrig::CudaConfig &xmrig::Config::cuda() const
{
    return d_ptr->cuda;
}
#endif


#ifdef XMRIG_ALGO_RANDOMX
const xmrig::RxConfig &xmrig::Config::rx() const
{
    return d_ptr->rx;
}
#endif


#if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
uint32_t xmrig::Config::healthPrintTime() const
{
    return d_ptr->healthPrintTime;
}
#endif


bool xmrig::Config::isShouldSave() const
{
    if (!isAutoSave()) {
        return false;
    }

#   ifdef XMRIG_FEATURE_OPENCL
    if (cl().isShouldSave()) {
        return true;
    }
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    if (cuda().isShouldSave()) {
        return true;
    }
#   endif

#   ifdef XMRIG_FEATURE_BENCHMARK
    if (m_benchmark.isNewBenchRun()) {
        return true;
    }
#   endif

    return (m_upgrade || cpu().isShouldSave());
}


bool xmrig::Config::read(const IJsonReader &reader, const char *fileName)
{
    if (!BaseConfig::read(reader, fileName)) {
        return false;
    }

    d_ptr->cpu.read(reader.getValue(kCPU));

#   ifdef XMRIG_ALGO_RANDOMX
    if (!d_ptr->rx.read(reader.getValue(kRandomX))) {
        m_upgrade = true;
    }
#   endif

#   ifdef XMRIG_FEATURE_OPENCL
    d_ptr->cl.read(reader.getValue(kOcl));
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    d_ptr->cuda.read(reader.getValue(kCuda));
#   endif

#   if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
    d_ptr->healthPrintTime = reader.getUint(kHealthPrintTime, d_ptr->healthPrintTime);
#   endif

#   ifdef XMRIG_FEATURE_BENCHMARK
    m_benchmark.read(reader.getValue(kAlgoPerf));
#   endif

    return true;
}


void xmrig::Config::getJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    doc.SetObject();

    auto &allocator = doc.GetAllocator();

    Value api(kObjectType);
    api.AddMember(StringRef(kApiId),                    m_apiId.toJSON(), allocator);
    api.AddMember(StringRef(kApiWorkerId),              m_apiWorkerId.toJSON(), allocator);

    doc.AddMember(StringRef(kApi),                      api, allocator);
    doc.AddMember(StringRef(kHttp),                     m_http.toJSON(doc), allocator);
    doc.AddMember(StringRef(kAutosave),                 isAutoSave(), allocator);
    doc.AddMember(StringRef(kBackground),               isBackground(), allocator);
    doc.AddMember(StringRef(kColors),                   Log::isColors(), allocator);
    doc.AddMember(StringRef(kTitle),                    title().toJSON(), allocator);

#   ifdef XMRIG_ALGO_RANDOMX
    doc.AddMember(StringRef(kRandomX),                  rx().toJSON(doc), allocator);
#   endif

    doc.AddMember(StringRef(kCPU),                      cpu().toJSON(doc), allocator);

#   ifdef XMRIG_FEATURE_OPENCL
    doc.AddMember(StringRef(kOcl),                      cl().toJSON(doc), allocator);
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    doc.AddMember(StringRef(kCuda),                     cuda().toJSON(doc), allocator);
#   endif

    doc.AddMember(StringRef(Pools::kDonateLevel),       m_pools.donateLevel(), allocator);
    doc.AddMember(StringRef(Pools::kDonateOverProxy),   m_pools.proxyDonate(), allocator);
    doc.AddMember(StringRef(kLogFile),                  m_logFile.toJSON(), allocator);
    doc.AddMember(StringRef(Pools::kPools),             m_pools.toJSON(doc), allocator);
    doc.AddMember(StringRef(kPrintTime),                printTime(), allocator);
#   if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
    doc.AddMember(StringRef(kHealthPrintTime),          healthPrintTime(), allocator);
#   endif
    doc.AddMember(StringRef(Pools::kRetries),           m_pools.retries(), allocator);
    doc.AddMember(StringRef(Pools::kRetryPause),        m_pools.retryPause(), allocator);
    doc.AddMember(StringRef(kSyslog),                   isSyslog(), allocator);

#   ifdef XMRIG_FEATURE_TLS
    doc.AddMember(StringRef(kTls),                      m_tls.toJSON(doc), allocator);
#   endif

    doc.AddMember(StringRef(kUserAgent),                m_userAgent.toJSON(), allocator);
    doc.AddMember(StringRef(kVerbose),                  Log::verbose(), allocator);
    doc.AddMember(StringRef(kWatch),                    m_watch, allocator);

#   ifdef XMRIG_FEATURE_BENCHMARK
    doc.AddMember(StringRef(kRebenchAlgo),              isRebenchAlgo(), allocator);
    doc.AddMember(StringRef(kBenchAlgoTime),            benchAlgoTime(), allocator);
    doc.AddMember(StringRef(kAlgoPerf),                 m_benchmark.toJSON(doc), allocator);
#   endif
}
