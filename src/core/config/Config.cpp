/* xmlcore
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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
#include <cinttypes>
#include <cstring>
#include <uv.h>


#include "core/config/Config.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "crypto/common/Assembly.h"


#ifdef xmlcore_ALGO_RANDOMX
#   include "crypto/rx/RxConfig.h"
#endif


#ifdef xmlcore_FEATURE_OPENCL
#   include "backend/opencl/OclConfig.h"
#endif


#ifdef xmlcore_FEATURE_CUDA
#   include "backend/cuda/CudaConfig.h"
#endif


namespace xmlcore {


#ifdef xmlcore_FEATURE_OPENCL
const char *Config::kOcl                = "opencl";
#endif

#ifdef xmlcore_FEATURE_CUDA
const char *Config::kCuda               = "cuda";
#endif

#if defined(xmlcore_FEATURE_NVML) || defined (xmlcore_FEATURE_ADL)
const char *Config::kHealthPrintTime    = "health-print-time";
#endif

#ifdef xmlcore_FEATURE_DMI
const char *Config::kDMI                = "dmi";
#endif


class ConfigPrivate
{
public:
    CpuConfig cpu;

#   ifdef xmlcore_ALGO_RANDOMX
    RxConfig rx;
#   endif

#   ifdef xmlcore_FEATURE_OPENCL
    OclConfig cl;
#   endif

#   ifdef xmlcore_FEATURE_CUDA
    CudaConfig cuda;
#   endif

#   if defined(xmlcore_FEATURE_NVML) || defined (xmlcore_FEATURE_ADL)
    uint32_t healthPrintTime = 60;
#   endif

#   ifdef xmlcore_FEATURE_DMI
    bool dmi = true;
#   endif
};

}


xmlcore::Config::Config() :
    d_ptr(new ConfigPrivate())
{
}


xmlcore::Config::~Config()
{
    delete d_ptr;
}


const xmlcore::CpuConfig &xmlcore::Config::cpu() const
{
    return d_ptr->cpu;
}


#ifdef xmlcore_FEATURE_OPENCL
const xmlcore::OclConfig &xmlcore::Config::cl() const
{
    return d_ptr->cl;
}
#endif


#ifdef xmlcore_FEATURE_CUDA
const xmlcore::CudaConfig &xmlcore::Config::cuda() const
{
    return d_ptr->cuda;
}
#endif


#ifdef xmlcore_ALGO_RANDOMX
const xmlcore::RxConfig &xmlcore::Config::rx() const
{
    return d_ptr->rx;
}
#endif


#if defined(xmlcore_FEATURE_NVML) || defined (xmlcore_FEATURE_ADL)
uint32_t xmlcore::Config::healthPrintTime() const
{
    return d_ptr->healthPrintTime;
}
#endif


#ifdef xmlcore_FEATURE_DMI
bool xmlcore::Config::isDMI() const
{
    return d_ptr->dmi;
}
#endif


bool xmlcore::Config::isShouldSave() const
{
    if (!isAutoSave()) {
        return false;
    }

#   ifdef xmlcore_FEATURE_OPENCL
    if (cl().isShouldSave()) {
        return true;
    }
#   endif

#   ifdef xmlcore_FEATURE_CUDA
    if (cuda().isShouldSave()) {
        return true;
    }
#   endif

    return (m_upgrade || cpu().isShouldSave());
}


bool xmlcore::Config::read(const IJsonReader &reader, const char *fileName)
{
    if (!BaseConfig::read(reader, fileName)) {
        return false;
    }

    d_ptr->cpu.read(reader.getValue(CpuConfig::kField));

#   ifdef xmlcore_ALGO_RANDOMX
    if (!d_ptr->rx.read(reader.getValue(RxConfig::kField))) {
        m_upgrade = true;
    }
#   endif

#   ifdef xmlcore_FEATURE_OPENCL
    d_ptr->cl.read(reader.getValue(kOcl));
#   endif

#   ifdef xmlcore_FEATURE_CUDA
    d_ptr->cuda.read(reader.getValue(kCuda));
#   endif

#   if defined(xmlcore_FEATURE_NVML) || defined (xmlcore_FEATURE_ADL)
    d_ptr->healthPrintTime = reader.getUint(kHealthPrintTime, d_ptr->healthPrintTime);
#   endif

#   ifdef xmlcore_FEATURE_DMI
    d_ptr->dmi = reader.getBool(kDMI, d_ptr->dmi);
#   endif

    return true;
}


void xmlcore::Config::getJSON(rapidjson::Document &doc) const
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

#   ifdef xmlcore_ALGO_RANDOMX
    doc.AddMember(StringRef(RxConfig::kField),          rx().toJSON(doc), allocator);
#   endif

    doc.AddMember(StringRef(CpuConfig::kField),         cpu().toJSON(doc), allocator);

#   ifdef xmlcore_FEATURE_OPENCL
    doc.AddMember(StringRef(kOcl),                      cl().toJSON(doc), allocator);
#   endif

#   ifdef xmlcore_FEATURE_CUDA
    doc.AddMember(StringRef(kCuda),                     cuda().toJSON(doc), allocator);
#   endif

    doc.AddMember(StringRef(kLogFile),                  m_logFile.toJSON(), allocator);

    m_pools.toJSON(doc, doc);

    doc.AddMember(StringRef(kPrintTime),                printTime(), allocator);
#   if defined(xmlcore_FEATURE_NVML) || defined (xmlcore_FEATURE_ADL)
    doc.AddMember(StringRef(kHealthPrintTime),          healthPrintTime(), allocator);
#   endif

#   ifdef xmlcore_FEATURE_DMI
    doc.AddMember(StringRef(kDMI),                      isDMI(), allocator);
#   endif

    doc.AddMember(StringRef(kSyslog),                   isSyslog(), allocator);

#   ifdef xmlcore_FEATURE_TLS
    doc.AddMember(StringRef(kTls),                      m_tls.toJSON(doc), allocator);
#   endif

    doc.AddMember(StringRef(kUserAgent),                m_userAgent.toJSON(), allocator);
    doc.AddMember(StringRef(kVerbose),                  Log::verbose(), allocator);
    doc.AddMember(StringRef(kWatch),                    m_watch, allocator);
    doc.AddMember(StringRef(kPauseOnBattery),           isPauseOnBattery(), allocator);
}
