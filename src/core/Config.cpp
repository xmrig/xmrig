/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include <string.h>
#include <uv.h>
#include <inttypes.h>

#include "common/config/ConfigLoader.h"
#include "common/cpu/Cpu.h"
#include "core/Config.h"
#include "core/ConfigCreator.h"
#include "crypto/Argon2_constants.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "HasherConfig.h"

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

static char affinity_tmp[20] = { 0 };


xmrig::Config::Config() : xmrig::CommonConfig(),
    m_shouldSave(false),
    m_priority(-1),
    m_mask(-1)
{
}


bool xmrig::Config::reload(const char *json)
{
    return xmrig::ConfigLoader::reload(this, json);
}


void xmrig::Config::getJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    doc.SetObject();

    auto &allocator = doc.GetAllocator();

    doc.AddMember("algo", StringRef(algorithm().name()), allocator);

    Value api(kObjectType);
    api.AddMember("port",         apiPort(), allocator);
    api.AddMember("access-token", apiToken() ? Value(StringRef(apiToken())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("id",           apiId() ? Value(StringRef(apiId())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("worker-id",    apiWorkerId() ? Value(StringRef(apiWorkerId())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("ipv6",         isApiIPv6(), allocator);
    api.AddMember("restricted",   isApiRestricted(), allocator);
    doc.AddMember("api",          api, allocator);

    doc.AddMember("autosave",     isAutoSave(), allocator);
    doc.AddMember("background",   isBackground(), allocator);
    doc.AddMember("colors",       isColors(), allocator);

    doc.AddMember("cpu-threads", cpuThreads(), allocator);
    if(cpuOptimization().isNull() || cpuOptimization().isEmpty())
        doc.AddMember("cpu-optimization", kNullType, allocator);
    else
        doc.AddMember("cpu-optimization", cpuOptimization().toJSON(doc), allocator);

    if (cpuAffinity() != -1L) {
        snprintf(affinity_tmp, sizeof(affinity_tmp) - 1, "0x%" PRIX64, cpuAffinity());
        doc.AddMember("cpu-affinity", StringRef(affinity_tmp), allocator);
    }
    else {
        doc.AddMember("cpu-affinity", kNullType, allocator);
    }

    doc.AddMember("priority",  priority() != -1 ? Value(priority()) : Value(kNullType), allocator);
    doc.AddMember("donate-level",  donateLevel(), allocator);
    doc.AddMember("log-file",      logFile()             ? Value(StringRef(logFile())).Move() : Value(kNullType).Move(), allocator);
    doc.AddMember("pools",         m_pools.toJSON(doc), allocator);
    doc.AddMember("print-time",    printTime(), allocator);
    doc.AddMember("retries",       m_pools.retries(), allocator);
    doc.AddMember("retry-pause",   m_pools.retryPause(), allocator);

    doc.AddMember("user-agent", userAgent() ? Value(StringRef(userAgent())).Move() : Value(kNullType).Move(), allocator);

#   ifdef HAVE_SYSLOG_H
    doc.AddMember("syslog", isSyslog(), allocator);
#   endif

    doc.AddMember("watch", m_watch, allocator);

    Value gpuEngines(kArrayType);

    for (const String gpuEngine : m_gpuEngine) {
        gpuEngines.PushBack(gpuEngine.toJSON(doc), allocator);
    }

    doc.AddMember("use-gpu", gpuEngines, allocator);

    Value gpuIntensities(kArrayType);

    for (const double gpuIntensity : m_gpuIntensity) {
        gpuIntensities.PushBack(gpuIntensity, allocator);
    }

    doc.AddMember("gpu-intensity", gpuIntensities, allocator);

    Value gpuFilters(kArrayType);

    for (const GPUFilter gpuFilter : m_gpuFilter) {
        gpuFilters.PushBack(toGPUFilterConfig(gpuFilter, doc), allocator);
    }

    doc.AddMember("gpu-filter", gpuFilters, allocator);
}


xmrig::Config *xmrig::Config::load(Process *process, IConfigListener *listener)
{
    return static_cast<Config*>(ConfigLoader::load(process, new ConfigCreator(), listener));
}


bool xmrig::Config::finalize()
{
    if (m_state != NoneState) {
        return CommonConfig::finalize();
    }

    if (!CommonConfig::finalize()) {
        return false;
    }

    if(m_gpuIntensity.size() == 0)
        m_gpuIntensity.push_back(50);

    HasherConfig hasherConfig(m_algorithm.algo(), m_algorithm.variant(), m_priority, m_cpuThreads, m_mask, m_cpuOptimization.isNull() ? "" : m_cpuOptimization.data(), m_gpuIntensity, m_gpuFilter);

    if(m_cpuThreads > 0)
        m_hashers.push_back(hasherConfig.clone(m_hashers.size(), "CPU"));

    if(m_gpuEngine.size() > 0)
        for(String gpuEngine : m_gpuEngine)
            m_hashers.push_back(hasherConfig.clone(m_hashers.size(), gpuEngine.data()));

    m_shouldSave = true;

    return true;
}


bool xmrig::Config::parseBoolean(int key, bool enable)
{
    if (!CommonConfig::parseBoolean(key, enable)) {
        return false;
    }

    return true;
}


bool xmrig::Config::parseString(int key, const char *arg)
{
    if (!CommonConfig::parseString(key, arg)) {
        return false;
    }

    switch (key) {
    case PriorityKey: /* --cpu-priority */
        return parseUint64(key, strtol(arg, nullptr, 10));

    case CPUThreadsKey:  /* --threads */
        if (strncmp(arg, "all", 3) == 0) {
            m_cpuThreads = Cpu::info()->threads();
            return true;
        }

        return parseUint64(key, strtol(arg, nullptr, 10));

    case CPUOptimizationKey:
        {
            String value = arg;
            if(value.isEqual("REF", true))
                value = "REF";
            else if(value.isEqual("SSE2", true))
                value = "SSE2";
            else if(value.isEqual("SSSE3", true))
                value = "SSSE3";
            else if(value.isEqual("AVX", true))
                value = "AVX";
            else if(value.isEqual("AVX2", true))
                value = "AVX2";
            else if(value.isEqual("AVX512F", true))
                value = "AVX512F";
            else if(value.isEqual("NEON", true))
                value = "NEON";
            else {
                printf("Invalid CPU optimization %s.\n", arg);
                return false;
            }
            m_cpuOptimization = value;
            return true;
        }

    case CPUAffinityKey: /* --cpu-affinity */
        {
            const char *p  = strstr(arg, "0x");
            return parseUint64(key, p ? strtoull(p, nullptr, 16) : strtoull(arg, nullptr, 10));
        }

    case UseGPUKey:
        {
            String strArg = arg;
            std::vector<String> gpuEngines = strArg.split(',');
            m_gpuEngine.clear();
            for(String engine : gpuEngines) {
                if(engine.isEqual("OPENCL", true))
                    m_gpuEngine.push_back("OPENCL");
                else if(engine.isEqual("CUDA", true))
                    m_gpuEngine.push_back("CUDA");
                else {
                    printf("Invalid GPU hasher %s, ignoring.\n", engine.data());
                }
            }

            return m_gpuEngine.size() > 0;
        }

    case GPUIntensityKey:
        {
            String strArg = arg;
            std::vector<String> gpuIntensities = strArg.split(',');
            for (const String intensity : gpuIntensities) {
                double value = strtod(intensity.data(), NULL);
                if(value > 100) value = 100;
                if(value < 0) value = 0;
                m_gpuIntensity.push_back(value);
            }
            return true;
        }

    case GPUFilterKey:
        {
            String strArg = arg;
            std::vector<String> gpuFilters = strArg.split(',');
            for (const String filter : gpuFilters) {
                std::vector<String> explodedFilter = filter.split(':');
                if(explodedFilter.size() == 1)
                    m_gpuFilter.push_back(GPUFilter("", explodedFilter[0].data()));
                else if(explodedFilter.size() >= 2)
                    m_gpuFilter.push_back(GPUFilter(explodedFilter[0].data(), explodedFilter[1].data()));
            }
            return true;
        }

    default:
        break;
    }

    return true;
}


bool xmrig::Config::parseUint64(int key, uint64_t arg)
{
    if (!CommonConfig::parseUint64(key, arg)) {
        return false;
    }

    switch (key) {
    case CPUAffinityKey: /* --cpu-affinity */
        if (arg) {
            m_mask = arg;
        }
        break;

    default:
        return parseInt(key, static_cast<int>(arg));
    }

    return true;
}


void xmrig::Config::parseJSON(const rapidjson::Document &doc)
{
    CommonConfig::parseJSON(doc);

    const rapidjson::Value &threads = doc["cpu-threads"];

    if (threads.IsUint())
        m_cpuThreads = threads.GetUint();
    else if(threads.IsString() && strcasecmp(threads.GetString(), "all") == 0)
        m_cpuThreads = Cpu::info()->threads();

    const rapidjson::Value &cpuOptimization = doc["cpu-optimization"];

    if (cpuOptimization.IsString()) {
        String value = cpuOptimization.GetString();
        if(value.isEqual("REF", true))
            value = "REF";
        else if(value.isEqual("SSE2", true))
            value = "SSE2";
        else if(value.isEqual("SSSE3", true))
            value = "SSSE3";
        else if(value.isEqual("AVX", true))
            value = "AVX";
        else if(value.isEqual("AVX2", true))
            value = "AVX2";
        else if(value.isEqual("AVX512F", true))
            value = "AVX512F";
        else if(value.isEqual("NEON", true))
            value = "NEON";
        else {
            printf("Invalid CPU optimization %s, ignoring.\n", value.data());
            value = "";
        }

        if(!value.isEqual(""))
            m_cpuOptimization = value;
    }

    const rapidjson::Value &gpuEngines = doc["use-gpu"];

    if(gpuEngines.IsArray()) {
        m_gpuEngine.clear();

        for(const rapidjson::Value &value : gpuEngines.GetArray()) {
            if(!value.IsString()) {
                continue;
            }

            String engine = value.GetString();
            if(engine.isEqual("OPENCL", true))
                m_gpuEngine.push_back("OPENCL");
            else if(engine.isEqual("CUDA", true))
                m_gpuEngine.push_back("CUDA");
            else {
                printf("Invalid GPU hasher %s, ignoring.\n", engine.data());
            }
        }
    }

    const rapidjson::Value &gpuIntensities = doc["gpu-intensity"];

    if(gpuIntensities.IsArray()) {
        for(const rapidjson::Value &value : gpuIntensities.GetArray()) {
            if(!value.IsDouble()) {
                continue;
            }

            double intensity = value.GetDouble();
            if(intensity > 100) intensity = 100;
            if(intensity < 0) intensity = 0;

            m_gpuIntensity.push_back(intensity);
        }
    }

    const rapidjson::Value &gpuFilters = doc["gpu-filter"];

    if(gpuFilters.IsArray()) {
        for(const rapidjson::Value &value : gpuFilters.GetArray()) {
            if(!value.IsObject()) {
                continue;
            }

            if(value.HasMember("filter")) {
                auto data = parseGPUFilterConfig(value);

                m_gpuFilter.push_back(data);
            }
        }
    }
}


bool xmrig::Config::parseInt(int key, int arg)
{
    switch (key) {
    case CPUThreadsKey: /* --threads */
        if (arg >= 0 && arg < 1024) {
            m_cpuThreads = arg;
        }
        break;

    case PriorityKey: /* --cpu-priority */
        if (arg >= 0 && arg <= 5) {
            m_priority = arg;
        }
        break;

    default:
        break;
    }

    return true;
}
