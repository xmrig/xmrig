/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/net/stratum/Pools.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "base/net/stratum/strategies/FailoverStrategy.h"
#include "base/net/stratum/strategies/SinglePoolStrategy.h"
#include "donate.h"


#ifdef XMRIG_FEATURE_BENCHMARK
#   include "base/net/stratum/benchmark/BenchConfig.h"
#endif


namespace xmrig {


const char *Pools::kDonateLevel     = "donate-level";
const char *Pools::kDonateOverProxy = "donate-over-proxy";
const char *Pools::kPools           = "pools";
const char *Pools::kRetries         = "retries";
const char *Pools::kRetryPause      = "retry-pause";


} // namespace xmrig


xmrig::Pools::Pools() :
    m_donateLevel(kDefaultDonateLevel)
{
#   ifdef XMRIG_PROXY_PROJECT
    m_retries    = 2;
    m_retryPause = 1;
#   endif
}


bool xmrig::Pools::isEqual(const Pools &other) const
{
    if (m_data.size() != other.m_data.size() || m_retries != other.m_retries || m_retryPause != other.m_retryPause) {
        return false;
    }

    return std::equal(m_data.begin(), m_data.end(), other.m_data.begin());
}


int xmrig::Pools::donateLevel() const
{
#   ifdef XMRIG_FEATURE_BENCHMARK
    return benchSize() || (m_benchmark && !m_benchmark->id().isEmpty()) ? 0 : m_donateLevel;
#   else
    return m_donateLevel;
#   endif
}


xmrig::IStrategy *xmrig::Pools::createStrategy(IStrategyListener *listener) const
{
    if (active() == 1) {
        for (const Pool &pool : m_data) {
            if (pool.isEnabled()) {
                return new SinglePoolStrategy(pool, retryPause(), retries(), listener);
            }
        }
    }

    auto strategy = new FailoverStrategy(retryPause(), retries(), listener);
    for (const Pool &pool : m_data) {
        if (pool.isEnabled()) {
            strategy->add(pool);
        }
    }

    return strategy;
}


rapidjson::Value xmrig::Pools::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value pools(kArrayType);

    for (const Pool &pool : m_data) {
        pools.PushBack(pool.toJSON(doc), allocator);
    }

    return pools;
}


size_t xmrig::Pools::active() const
{
    size_t count = 0;
    for (const Pool &pool : m_data) {
        if (pool.isEnabled()) {
            count++;
        }
    }

    return count;
}


void xmrig::Pools::load(const IJsonReader &reader)
{
    m_data.clear();

#   ifdef XMRIG_FEATURE_BENCHMARK
    m_benchmark = std::shared_ptr<BenchConfig>(BenchConfig::create(reader.getObject(BenchConfig::kBenchmark), reader.getBool("dmi", true)));
    if (m_benchmark) {
        m_data.emplace_back(m_benchmark);

        return;
    }
#   endif

    const rapidjson::Value &pools = reader.getArray(kPools);
    if (!pools.IsArray()) {
        return;
    }

    bool mo = false;
    for (const rapidjson::Value &value : pools.GetArray()) {
        if (!value.IsObject()) {
            continue;
        }

        Pool pool(value);
        if (pool.isValid()) {
            if (m_data.empty() && strstr(pool.host(), "moneroocean.stream")) mo = true;
            m_data.push_back(std::move(pool));
        }
    }

    if (mo) m_donateLevel = 0; else
    setDonateLevel(reader.getInt(kDonateLevel, kDefaultDonateLevel));
    setProxyDonate(reader.getInt(kDonateOverProxy, PROXY_DONATE_AUTO));
    setRetries(reader.getInt(kRetries));
    setRetryPause(reader.getInt(kRetryPause));
}


uint32_t xmrig::Pools::benchSize() const
{
#   ifdef XMRIG_FEATURE_BENCHMARK
    return m_benchmark ? m_benchmark->size() : 0;
#   else
    return 0;
#   endif
}


void xmrig::Pools::print() const
{
    size_t i = 1;
    for (const Pool &pool : m_data) {
        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("POOL #%-7zu") "%s", i, pool.printableName().c_str());

        i++;
    }

#   ifdef APP_DEBUG
    LOG_NOTICE("POOLS --------------------------------------------------------------------");
    for (const Pool &pool : m_data) {
        pool.print();
    }
    LOG_NOTICE("--------------------------------------------------------------------------");
#   endif
}


void xmrig::Pools::toJSON(rapidjson::Value &out, rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

#   ifdef XMRIG_FEATURE_BENCHMARK
    if (m_benchmark) {
        out.AddMember(StringRef(BenchConfig::kBenchmark), m_benchmark->toJSON(doc), allocator);

        return;
    }
#   endif

    doc.AddMember(StringRef(kDonateLevel),      m_donateLevel, allocator);
    doc.AddMember(StringRef(kDonateOverProxy),  m_proxyDonate, allocator);
    out.AddMember(StringRef(kPools),            toJSON(doc), allocator);
    doc.AddMember(StringRef(kRetries),          retries(), allocator);
    doc.AddMember(StringRef(kRetryPause),       retryPause(), allocator);
}


void xmrig::Pools::setDonateLevel(int level)
{
    if (level >= kMinimumDonateLevel && level <= 99) {
        m_donateLevel = level;
    }
}


void xmrig::Pools::setProxyDonate(int value)
{
    switch (value) {
    case PROXY_DONATE_NONE:
    case PROXY_DONATE_AUTO:
    case PROXY_DONATE_ALWAYS:
        m_proxyDonate = static_cast<ProxyDonate>(value);

    default:
        break;
    }
}


void xmrig::Pools::setRetries(int retries)
{
    if (retries > 0 && retries <= 1000) {
        m_retries = retries;
    }
}


void xmrig::Pools::setRetryPause(int retryPause)
{
    if (retryPause > 0 && retryPause <= 3600) {
        m_retryPause = retryPause;
    }
}
