/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <string.h>
#include <uv.h>
#include <inttypes.h>


#include "common/config/ConfigLoader.h"
#include "common/cpu/Cpu.h"
#include "core/Config.h"
#include "core/ConfigCreator.h"
#include "crypto/Asm.h"
#include "crypto/CryptoNight_constants.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "workers/CpuThread.h"


static char affinity_tmp[20] = { 0 };


xmrig::Config::Config() : xmrig::CommonConfig(),
    m_aesMode(AES_AUTO),
    m_algoVariant(AV_AUTO),
    m_assembly(ASM_AUTO),
    m_hugePages(true),
    m_safe(false),
    m_shouldSave(false),
    m_maxCpuUsage(75),
    m_priority(-1)
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

#   ifndef XMRIG_NO_ASM
    doc.AddMember("asm",          Asm::toJSON(m_assembly), allocator);
#   endif

    doc.AddMember("autosave",     isAutoSave(), allocator);
    doc.AddMember("av",           algoVariant(), allocator);
    doc.AddMember("background",   isBackground(), allocator);
    doc.AddMember("colors",       isColors(), allocator);

    if (affinity() != -1L) {
        snprintf(affinity_tmp, sizeof(affinity_tmp) - 1, "0x%" PRIX64, affinity());
        doc.AddMember("cpu-affinity", StringRef(affinity_tmp), allocator);
    }
    else {
        doc.AddMember("cpu-affinity", kNullType, allocator);
    }

    doc.AddMember("cpu-priority",  priority() != -1 ? Value(priority()) : Value(kNullType), allocator);
    doc.AddMember("donate-level",  donateLevel(), allocator);
    doc.AddMember("huge-pages",    isHugePages(), allocator);
    doc.AddMember("hw-aes",        m_aesMode == AES_AUTO ? Value(kNullType) : Value(m_aesMode == AES_HW), allocator);
    doc.AddMember("log-file",      logFile()             ? Value(StringRef(logFile())).Move() : Value(kNullType).Move(), allocator);
    doc.AddMember("max-cpu-usage", m_maxCpuUsage, allocator);

    Value pools(kArrayType);

    for (const Pool &pool : m_activePools) {
        pools.PushBack(pool.toJSON(doc), allocator);
    }

    doc.AddMember("pools",         pools, allocator);
    doc.AddMember("print-time",    printTime(), allocator);
    doc.AddMember("retries",       retries(), allocator);
    doc.AddMember("retry-pause",   retryPause(), allocator);
    doc.AddMember("safe",          m_safe, allocator);

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

    doc.AddMember("user-agent", userAgent() ? Value(StringRef(userAgent())).Move() : Value(kNullType).Move(), allocator);

#   ifdef HAVE_SYSLOG_H
    doc.AddMember("syslog", isSyslog(), allocator);
#   endif

    doc.AddMember("watch", m_watch, allocator);
}


xmrig::Config *xmrig::Config::load(int argc, char **argv, IWatcherListener *listener)
{
    return static_cast<Config*>(ConfigLoader::load(argc, argv, new ConfigCreator(), listener));
}


bool xmrig::Config::finalize()
{
    if (m_state != NoneState) {
        return CommonConfig::finalize();
    }

    if (!CommonConfig::finalize()) {
        return false;
    }

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

    const size_t size = CpuThread::multiway(av) * cn_select_memory(m_algorithm.algo()) / 1024;

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


bool xmrig::Config::parseBoolean(int key, bool enable)
{
    if (!CommonConfig::parseBoolean(key, enable)) {
        return false;
    }

    switch (key) {
    case SafeKey: /* --safe */
        m_safe = enable;
        break;

    case HugePagesKey: /* --no-huge-pages */
        m_hugePages = enable;
        break;

    case HardwareAESKey: /* hw-aes config only */
        m_aesMode = enable ? AES_HW : AES_SOFT;
        break;

#   ifndef XMRIG_NO_ASM
    case AssemblyKey:
        m_assembly = Asm::parse(enable);
        break;
#   endif

    default:
        break;
    }

    return true;
}


bool xmrig::Config::parseString(int key, const char *arg)
{
    if (!CommonConfig::parseString(key, arg)) {
        return false;
    }

    switch (key) {
    case AVKey:          /* --av */
    case MaxCPUUsageKey: /* --max-cpu-usage */
    case CPUPriorityKey: /* --cpu-priority */
        return parseUint64(key, strtol(arg, nullptr, 10));

    case SafeKey: /* --safe */
        return parseBoolean(key, true);

    case HugePagesKey: /* --no-huge-pages */
        return parseBoolean(key, false);

    case ThreadsKey:  /* --threads */
        if (strncmp(arg, "all", 3) == 0) {
            m_threads.count = Cpu::info()->threads();
            return true;
        }

        return parseUint64(key, strtol(arg, nullptr, 10));

    case CPUAffinityKey: /* --cpu-affinity */
        {
            const char *p  = strstr(arg, "0x");
            return parseUint64(key, p ? strtoull(p, nullptr, 16) : strtoull(arg, nullptr, 10));
        }

#   ifndef XMRIG_NO_ASM
    case AssemblyKey: /* --asm */
        m_assembly = Asm::parse(arg);
        break;
#   endif

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
            m_threads.mask = arg;
        }
        break;

    default:
        return parseInt(key, static_cast<int>(arg));
    }

    return true;
}


void xmrig::Config::parseJSON(const rapidjson::Document &doc)
{
    const rapidjson::Value &threads = doc["threads"];

    if (threads.IsArray()) {
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
}


bool xmrig::Config::parseInt(int key, int arg)
{
    switch (key) {
    case ThreadsKey: /* --threads */
        if (arg >= 0 && arg < 1024) {
            m_threads.count = arg;
        }
        break;

    case AVKey: /* --av */
        if (arg >= AV_AUTO && arg < AV_MAX) {
            m_algoVariant = static_cast<AlgoVariant>(arg);
        }
        break;

    case MaxCPUUsageKey: /* --max-cpu-usage */
        if (m_maxCpuUsage > 0 && arg <= 100) {
            m_maxCpuUsage = arg;
        }
        break;

    case CPUPriorityKey: /* --cpu-priority */
        if (arg >= 0 && arg <= 5) {
            m_priority = arg;
        }
        break;

    default:
        break;
    }

    return true;
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
