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


#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "common/cpu/Cpu.h"
#include "core/config/Config.h"
#include "crypto/Asm.h"
#include "crypto/CryptoNight_constants.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "workers/CpuThread.h"


static char affinity_tmp[20] = { 0 };


xmrig::Config::Config() :
    m_aesMode(AES_AUTO),
    m_algoVariant(AV_AUTO),
    m_assembly(ASM_AUTO),
    m_hugePages(true),
    m_safe(false),
    m_shouldSave(false),
    m_maxCpuUsage(100),
    m_priority(-1)
{
}


bool xmrig::Config::read(const IJsonReader &reader, const char *fileName)
{
    if (!BaseConfig::read(reader, fileName)) {
        return false;
    }

    m_hugePages = reader.getBool("huge-pages", true);
    m_safe      = reader.getBool("safe");

    setAesMode(reader.getValue("hw-aes"));
    setAlgoVariant(reader.getInt("av"));
    setMaxCpuUsage(reader.getInt("max-cpu-usage", 100));
    setPriority(reader.getInt("cpu-priority", -1));
    setThreads(reader.getValue("threads"));

#   ifndef XMRIG_NO_ASM
    setAssembly(reader.getValue("asm"));
#   endif

    return finalize();
}


void xmrig::Config::getJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    doc.SetObject();

    auto &allocator = doc.GetAllocator();

    doc.AddMember("algo", StringRef(algorithm().name()), allocator);

    Value api(kObjectType);
    api.AddMember("id",           m_apiId.toJSON(), allocator);
    api.AddMember("worker-id",    m_apiWorkerId.toJSON(), allocator);
    doc.AddMember("api",          api, allocator);
    doc.AddMember("http",         m_http.toJSON(doc), allocator);

#   ifndef XMRIG_NO_ASM
    doc.AddMember("asm",          Asm::toJSON(m_assembly), allocator);
#   endif

    doc.AddMember("autosave",     isAutoSave(), allocator);
    doc.AddMember("av",           algoVariant(), allocator);
    doc.AddMember("background",   isBackground(), allocator);
    doc.AddMember("colors",       Log::colors, allocator);

    if (affinity() != -1L) {
        snprintf(affinity_tmp, sizeof(affinity_tmp) - 1, "0x%" PRIX64, affinity());
        doc.AddMember("cpu-affinity", StringRef(affinity_tmp), allocator);
    }
    else {
        doc.AddMember("cpu-affinity", kNullType, allocator);
    }

    doc.AddMember("cpu-priority",      priority() != -1 ? Value(priority()) : Value(kNullType), allocator);
    doc.AddMember("donate-level",      m_pools.donateLevel(), allocator);
    doc.AddMember("donate-over-proxy", m_pools.proxyDonate(), allocator);
    doc.AddMember("huge-pages",        isHugePages(), allocator);
    doc.AddMember("hw-aes",            m_aesMode == AES_AUTO ? Value(kNullType) : Value(m_aesMode == AES_HW), allocator);
    doc.AddMember("log-file",          m_logFile.toJSON(), allocator);
    doc.AddMember("max-cpu-usage",     m_maxCpuUsage, allocator);
    doc.AddMember("pools",             m_pools.toJSON(doc), allocator);
    doc.AddMember("print-time",        printTime(), allocator);
    doc.AddMember("retries",           m_pools.retries(), allocator);
    doc.AddMember("retry-pause",       m_pools.retryPause(), allocator);
    doc.AddMember("safe",              m_safe, allocator);

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
    if (!m_threads.cpu.empty()) {
        m_threads.mode     = Advanced;
        const bool softAES = (m_aesMode == AES_AUTO ? (Cpu::info()->hasAES() ? AES_HW : AES_SOFT) : m_aesMode) == AES_SOFT;

        for (size_t i = 0; i < m_threads.cpu.size(); ++i) {
            m_threads.list.push_back(CpuThread::createFromData(i, m_algorithm.algo(), m_threads.cpu[i], m_priority, softAES));
        }

        return true;
    }

    const AlgoVariant av = getAlgoVariant();
    m_threads.mode = m_threads.count ? Simple : Automatic;

    const Variant v = m_algorithm.variant();
    const size_t size = CpuThread::multiway(av) * cn_select_memory(m_algorithm.algo(), v) / 1024;

    if (!m_threads.count) {
        m_threads.count = Cpu::info()->optimalThreadsCount(size, m_maxCpuUsage);
    }
    else if (m_safe) {
        const size_t count = Cpu::info()->optimalThreadsCount(size, m_maxCpuUsage);
        if (m_threads.count > count) {
            m_threads.count = count;
        }
    }

    for (size_t i = 0; i < m_threads.count; ++i) {
        m_threads.list.push_back(CpuThread::createFromAV(i, m_algorithm.algo(), av, m_threads.mask, m_priority, m_assembly));
    }

    m_shouldSave = m_threads.mode == Automatic;

    return true;
}


void xmrig::Config::setAesMode(const rapidjson::Value &aesMode)
{
    if (aesMode.IsBool()) {
        m_aesMode = aesMode.GetBool() ? AES_HW : AES_SOFT;
    }
}


void xmrig::Config::setAlgoVariant(int av)
{
    if (av >= AV_AUTO && av < AV_MAX) {
        m_algoVariant = static_cast<AlgoVariant>(av);
    }
}


void xmrig::Config::setMaxCpuUsage(int max)
{
    if (max > 0 && max <= 100) {
        m_maxCpuUsage = max;
    }
}


void xmrig::Config::setPriority(int priority)
{
    if (priority >= 0 && priority <= 5) {
        m_priority = priority;
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
#   ifndef XMRIG_NO_AEON
    if (m_algorithm.algo() == xmrig::CRYPTONIGHT_LITE) {
        return getAlgoVariantLite();
    }
#   endif

    if (m_algoVariant <= AV_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::info()->hasAES() ? AV_SINGLE : AV_SINGLE_SOFT;
    }

    if (m_safe && !Cpu::info()->hasAES() && m_algoVariant <= AV_DOUBLE) {
        return static_cast<AlgoVariant>(m_algoVariant + 2);
    }

    return m_algoVariant;
}


#ifndef XMRIG_NO_AEON
xmrig::AlgoVariant xmrig::Config::getAlgoVariantLite() const
{
    if (m_algoVariant <= AV_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::info()->hasAES() ? AV_DOUBLE : AV_DOUBLE_SOFT;
    }

    if (m_safe && !Cpu::info()->hasAES() && m_algoVariant <= AV_DOUBLE) {
        return static_cast<AlgoVariant>(m_algoVariant + 2);
    }

    return m_algoVariant;
}
#endif


#ifndef XMRIG_NO_ASM
void xmrig::Config::setAssembly(const rapidjson::Value &assembly)
{
    m_assembly = Asm::parse(assembly);
}
#endif
