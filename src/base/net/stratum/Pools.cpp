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


#include "base/io/log/Log.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "base/net/stratum/Pools.h"
#include "base/net/stratum/strategies/FailoverStrategy.h"
#include "base/net/stratum/strategies/SinglePoolStrategy.h"
#include "donate.h"
#include "rapidjson/document.h"


xmrig::Pools::Pools() :
    m_donateLevel(kDefaultDonateLevel),
    m_retries(5),
    m_retryPause(5),
    m_proxyDonate(PROXY_DONATE_AUTO)
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


xmrig::IStrategy *xmrig::Pools::createStrategy(IStrategyListener *listener) const
{
    if (active() == 1) {
        for (const Pool &pool : m_data) {
            if (pool.isEnabled()) {
                return new SinglePoolStrategy(pool, retryPause(), retries(), listener);
            }
        }
    }

    FailoverStrategy *strategy = new FailoverStrategy(retryPause(), retries(), listener);
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

    const rapidjson::Value &pools = reader.getArray("pools");
    if (!pools.IsArray()) {
        return;
    }

    for (const rapidjson::Value &value : pools.GetArray()) {
        if (!value.IsObject()) {
            continue;
        }

        Pool pool(value);
        if (pool.isValid()) {
            m_data.push_back(std::move(pool));
        }
    }

    setDonateLevel(reader.getInt("donate-level", kDefaultDonateLevel));
    setProxyDonate(reader.getInt("donate-over-proxy", PROXY_DONATE_AUTO));
    setRetries(reader.getInt("retries"));
    setRetryPause(reader.getInt("retry-pause"));
}


void xmrig::Pools::print() const
{
    size_t i = 1;
    for (const Pool &pool : m_data) {
        Log::print(GREEN_BOLD(" * ") WHITE_BOLD("POOL #%-7zu") CSI "1;%dm%s" CLEAR " algo " WHITE_BOLD("%s"),
                   i,
                   (pool.isEnabled() ? (pool.isTLS() ? 32 : 36) : 31),
                   pool.url().data(),
                   pool.algorithm().isValid() ? pool.algorithm().shortName() : "auto"
                   );

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
