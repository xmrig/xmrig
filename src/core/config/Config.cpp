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


#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "core/config/Config.h"
#include "crypto/common/Assembly.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "workers/CpuThread.h"


xmrig::Config::Config() :
    m_algoVariant(AV_AUTO),
    m_shouldSave(false)
{
}


bool xmrig::Config::read(const IJsonReader &reader, const char *fileName)
{
    if (!BaseConfig::read(reader, fileName)) {
        return false;
    }

    m_cpu.read(reader.getValue("cpu"));

    setAlgoVariant(reader.getInt("av"));
    setThreads(reader.getValue("threads"));

    return finalize();
}


void xmrig::Config::getJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    doc.SetObject();

    auto &allocator = doc.GetAllocator();

    Value api(kObjectType);
    api.AddMember("id",           m_apiId.toJSON(), allocator);
    api.AddMember("worker-id",    m_apiWorkerId.toJSON(), allocator);
    doc.AddMember("api",          api, allocator);
    doc.AddMember("http",         m_http.toJSON(doc), allocator);
    doc.AddMember("autosave",     isAutoSave(), allocator);
    doc.AddMember("av",           algoVariant(), allocator);
    doc.AddMember("background",   isBackground(), allocator);
    doc.AddMember("colors",       Log::colors, allocator);

//    if (affinity() != -1L) {
//        snprintf(affinity_tmp, sizeof(affinity_tmp) - 1, "0x%" PRIX64, affinity());
//        doc.AddMember("cpu-affinity", StringRef(affinity_tmp), allocator);
//    }
//    else {
//        doc.AddMember("cpu-affinity", kNullType, allocator);
//    }


    doc.AddMember("cpu", m_cpu.toJSON(doc), allocator);

    doc.AddMember("donate-level",      m_pools.donateLevel(), allocator);
    doc.AddMember("donate-over-proxy", m_pools.proxyDonate(), allocator);
    doc.AddMember("log-file",          m_logFile.toJSON(), allocator);
    doc.AddMember("pools",             m_pools.toJSON(doc), allocator);
    doc.AddMember("print-time",        printTime(), allocator);
    doc.AddMember("retries",           m_pools.retries(), allocator);
    doc.AddMember("retry-pause",       m_pools.retryPause(), allocator);

    if (threadsMode() != Simple) {
        Value threads(kArrayType);

        for (const IThread *thread : m_threads.list) {
            threads.PushBack(thread->toConfig(doc), allocator);
        }

        doc.AddMember("threads", threads, allocator);
    }
    else {
        doc.AddMember("threads", threadsCount(), allocator);
    }

    doc.AddMember("user-agent", m_userAgent.toJSON(), allocator);
    doc.AddMember("syslog",     isSyslog(), allocator);
    doc.AddMember("watch",      m_watch, allocator);
}


bool xmrig::Config::finalize()
{
    Algorithm algorithm(Algorithm::CN_0); // FIXME algo

    if (!m_threads.cpu.empty()) {
        m_threads.mode = Advanced;

        for (size_t i = 0; i < m_threads.cpu.size(); ++i) {
            m_threads.list.push_back(CpuThread::createFromData(i, algorithm, m_threads.cpu[i], m_cpu.priority(), !m_cpu.isHwAES()));
        }

        return true;
    }

    const AlgoVariant av = getAlgoVariant();
    m_threads.mode = m_threads.count ? Simple : Automatic;

    const size_t size = CpuThread::multiway(av) * CnAlgo<>::memory(algorithm) / 1024; // FIXME MEMORY

    if (!m_threads.count) {
        m_threads.count = Cpu::info()->optimalThreadsCount(size, 100);
    }
//    else if (m_safe) {
//        const size_t count = Cpu::info()->optimalThreadsCount(size, m_maxCpuUsage);
//        if (m_threads.count > count) {
//            m_threads.count = count;
//        }
//    }

    for (size_t i = 0; i < m_threads.count; ++i) {
        m_threads.list.push_back(CpuThread::createFromAV(i, algorithm, av, m_threads.mask, m_cpu.priority(), m_cpu.assembly()));
    }

    m_shouldSave = m_threads.mode == Automatic;

    return true;
}


void xmrig::Config::setAlgoVariant(int av)
{
    if (av >= AV_AUTO && av < AV_MAX) {
        m_algoVariant = static_cast<AlgoVariant>(av);
    }
}


void xmrig::Config::setThreads(const rapidjson::Value &threads)
{
    if (threads.IsArray()) {
        m_threads.cpu.clear();

        for (const rapidjson::Value &value : threads.GetArray()) {
            if (!value.IsObject()) {
                continue;
            }

            if (value.HasMember("low_power_mode")) {
                auto data = CpuThread::parse(value);

                if (data.valid) {
                    m_threads.cpu.push_back(std::move(data));
                }
            }
        }
    }
    else if (threads.IsUint()) {
        const unsigned count = threads.GetUint();
        if (count < 1024) {
            m_threads.count = count;
        }
    }
}


xmrig::AlgoVariant xmrig::Config::getAlgoVariant() const
{
#   ifdef XMRIG_ALGO_CN_LITE
//    if (m_algorithm.algo() == xmrig::CRYPTONIGHT_LITE) { // FIXME
//        return getAlgoVariantLite();
//    }
#   endif

    if (m_algoVariant <= AV_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::info()->hasAES() ? AV_SINGLE : AV_SINGLE_SOFT;
    }

//    if (m_safe && !Cpu::info()->hasAES() && m_algoVariant <= AV_DOUBLE) {
//        return static_cast<AlgoVariant>(m_algoVariant + 2);
//    }

    return m_algoVariant;
}


#ifdef XMRIG_ALGO_CN_LITE
xmrig::AlgoVariant xmrig::Config::getAlgoVariantLite() const
{
    if (m_algoVariant <= AV_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::info()->hasAES() ? AV_DOUBLE : AV_DOUBLE_SOFT;
    }

//    if (m_safe && !Cpu::info()->hasAES() && m_algoVariant <= AV_DOUBLE) {
//        return static_cast<AlgoVariant>(m_algoVariant + 2);
//    }

    return m_algoVariant;
}
#endif
