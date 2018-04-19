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
#include "core/Config.h"
#include "core/ConfigCreator.h"
#include "Cpu.h"
#include "crypto/CryptoNight_constants.h"
#include "net/Pool.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "workers/CpuThread.h"


static char affinity_tmp[20] = { 0 };


xmrig::Config::Config() : xmrig::CommonConfig(),
    m_aesMode(AES_AUTO),
    m_algoVariant(AV_AUTO),
    m_dryRun(false),
    m_hugePages(true),
    m_safe(false),
    m_maxCpuUsage(75),
    m_priority(-1)
{
}


xmrig::Config::~Config()
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

    doc.AddMember("algo", StringRef(algoName()), allocator);

    Value api(kObjectType);
    api.AddMember("port",         apiPort(), allocator);
    api.AddMember("access-token", apiToken() ? Value(StringRef(apiToken())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("worker-id",    apiWorkerId() ? Value(StringRef(apiWorkerId())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("ipv6",         isApiIPv6(), allocator);
    api.AddMember("restricted",   isApiRestricted(), allocator);
    doc.AddMember("api",          api, allocator);

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

    for (const Pool &pool : m_pools) {
        Value obj(kObjectType);

        obj.AddMember("url",     StringRef(pool.url()), allocator);
        obj.AddMember("user",    StringRef(pool.user()), allocator);
        obj.AddMember("pass",    StringRef(pool.password()), allocator);

        if (pool.keepAlive() == 0 || pool.keepAlive() == Pool::kKeepAliveTimeout) {
            obj.AddMember("keepalive", pool.keepAlive() > 0, allocator);
        }
        else {
            obj.AddMember("keepalive", pool.keepAlive(), allocator);
        }

        obj.AddMember("nicehash", pool.isNicehash(), allocator);
        obj.AddMember("variant",  pool.variant(), allocator);

        pools.PushBack(obj, allocator);
    }

    doc.AddMember("pools",         pools, allocator);
    doc.AddMember("print-time",    printTime(), allocator);
    doc.AddMember("retries",       retries(), allocator);
    doc.AddMember("retry-pause",   retryPause(), allocator);
    doc.AddMember("safe",          m_safe, allocator);

    if (threadsMode() == Advanced) {
        Value threads(kArrayType);

        for (const IThread *thread : m_threads.list) {
            threads.PushBack(thread->toConfig(doc), doc.GetAllocator());
        }

        doc.AddMember("threads", threads, allocator);
    }
    else {
        doc.AddMember("threads", threadsMode() == Automatic ? Value(kNullType) : Value(threadsCount()), allocator);
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


bool xmrig::Config::adjust()
{
    if (!CommonConfig::adjust()) {
        return false;
    }

    if (!m_threads.cpu.empty()) {
        m_threads.mode     = Advanced;
        const bool softAES = (m_aesMode == AES_AUTO ? (Cpu::hasAES() ? AES_HW : AES_SOFT) : m_aesMode) == AES_SOFT;

        for (size_t i = 0; i < m_threads.cpu.size(); ++i) {
            m_threads.list.push_back(CpuThread::createFromData(i, m_algorithm, m_threads.cpu[i], m_priority, softAES));
        }

        return true;
    }

    m_algoVariant  = getAlgoVariant();
    m_threads.mode = m_threads.count ? Simple : Automatic;

    const size_t size = CpuThread::multiway(m_algoVariant) * cn_select_memory(m_algorithm) / 1024;

    if (!m_threads.count) {
        m_threads.count = Cpu::optimalThreadsCount(size, m_maxCpuUsage);
    }
    else if (m_safe) {
        const size_t count = Cpu::optimalThreadsCount(size, m_maxCpuUsage);
        if (m_threads.count > count) {
            m_threads.count = count;
        }
    }

    for (size_t i = 0; i < m_threads.count; ++i) {
        m_threads.list.push_back(CpuThread::createFromAV(i, m_algorithm, m_algoVariant, m_threads.mask, m_priority));
    }

    return true;
}


bool xmrig::Config::parseBoolean(int key, bool enable)
{
    if (!CommonConfig::parseBoolean(key, enable)) {
        return false;
    }

    switch (key) {
    case IConfig::SafeKey: /* --safe */
        m_safe = enable;
        break;

    case IConfig::HugePagesKey: /* --no-huge-pages */
        m_hugePages = enable;
        break;

    case IConfig::DryRunKey: /* --dry-run */
        m_dryRun = enable;
        break;

    case IConfig::HardwareAESKey: /* hw-aes config only */
        m_aesMode = enable ? AES_HW : AES_SOFT;
        break;

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
    case xmrig::IConfig::AVKey:          /* --av */
    case xmrig::IConfig::MaxCPUUsageKey: /* --max-cpu-usage */
    case xmrig::IConfig::CPUPriorityKey: /* --cpu-priority */
        return parseUint64(key, strtol(arg, nullptr, 10));

    case xmrig::IConfig::SafeKey:   /* --safe */
    case xmrig::IConfig::DryRunKey: /* --dry-run */
        return parseBoolean(key, true);

    case xmrig::IConfig::HugePagesKey: /* --no-huge-pages */
        return parseBoolean(key, false);

    case xmrig::IConfig::ThreadsKey:  /* --threads */
        if (strncmp(arg, "all", 3) == 0) {
            m_threads.count = Cpu::threads();
            return true;
        }

        return parseUint64(key, strtol(arg, nullptr, 10));

    case xmrig::IConfig::CPUAffinityKey: /* --cpu-affinity */
        {
            const char *p  = strstr(arg, "0x");
            return parseUint64(key, p ? strtoull(p, nullptr, 16) : strtoull(arg, nullptr, 10));
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
    case xmrig::IConfig::CPUAffinityKey: /* --cpu-affinity */
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
    case xmrig::IConfig::ThreadsKey: /* --threads */
        if (arg >= 0 && arg < 1024) {
            m_threads.count = arg;
        }
        break;

    case xmrig::IConfig::AVKey: /* --av */
        if (arg >= AV_AUTO && arg < AV_MAX) {
            m_algoVariant = static_cast<AlgoVariant>(arg);
        }
        break;

    case xmrig::IConfig::MaxCPUUsageKey: /* --max-cpu-usage */
        if (m_maxCpuUsage > 0 && arg <= 100) {
            m_maxCpuUsage = arg;
        }
        break;

    case xmrig::IConfig::CPUPriorityKey: /* --cpu-priority */
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
    if (m_algorithm == xmrig::CRYPTONIGHT_LITE) {
        return getAlgoVariantLite();
    }
#   endif

    if (m_algoVariant <= AV_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::hasAES() ? AV_SINGLE : AV_SINGLE_SOFT;
    }

    if (m_safe && !Cpu::hasAES() && m_algoVariant <= AV_DOUBLE) {
        return static_cast<AlgoVariant>(m_algoVariant + 2);
    }

    return m_algoVariant;
}


#ifndef XMRIG_NO_AEON
xmrig::AlgoVariant xmrig::Config::getAlgoVariantLite() const
{
    if (m_algoVariant <= AV_AUTO || m_algoVariant >= AV_MAX) {
        return Cpu::hasAES() ? AV_DOUBLE : AV_DOUBLE_SOFT;
    }

    if (m_safe && !Cpu::hasAES() && m_algoVariant <= AV_DOUBLE) {
        return static_cast<AlgoVariant>(m_algoVariant + 2);
    }

    return m_algoVariant;
}
#endif
