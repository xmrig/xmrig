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


#include "base/net/Pools.h"
#include "common/log/Log.h"
#include "common/net/strategies/FailoverStrategy.h"
#include "common/net/strategies/SinglePoolStrategy.h"
#include "rapidjson/document.h"


xmrig::Pools::Pools() :
    m_retries(5),
    m_retryPause(5)
{
#   ifdef XMRIG_PROXY_PROJECT
    m_retries    = 2;
    m_retryPause = 1;
#   endif
}


xmrig::Pool &xmrig::Pools::current()
{
    if (m_data.empty()) {
        m_data.push_back(Pool());
    }

    return m_data.back();
}


bool xmrig::Pools::isEqual(const Pools &other) const
{
    if (m_data.size() != other.m_data.size() || m_retries != other.m_retries || m_retryPause != other.m_retryPause) {
        return false;
    }

    return std::equal(m_data.begin(), m_data.end(), other.m_data.begin());
}


bool xmrig::Pools::setUrl(const char *url)
{
    if (m_data.empty() || m_data.back().isValid()) {
        Pool pool(url);

        if (pool.isValid()) {
            m_data.push_back(std::move(pool));
            return true;
        }

        return false;
    }

    current().parse(url);

    return m_data.back().isValid();
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


void xmrig::Pools::adjust(const Algorithm &algorithm)
{
    for (Pool &pool : m_data) {
        pool.adjust(algorithm);
    }
}


void xmrig::Pools::load(const rapidjson::Value &pools)
{
    m_data.clear();

    for (const rapidjson::Value &value : pools.GetArray()) {
        if (!value.IsObject()) {
            continue;
        }

        Pool pool(value);
        if (pool.isValid()) {
            m_data.push_back(std::move(pool));
        }
    }
}


void xmrig::Pools::print() const
{
    size_t i = 1;
    for (const Pool &pool : m_data) {
        if (Log::colors) {
            const int color = pool.isEnabled() ? (pool.isTLS() ? 32 : 36) : 31;

            Log::i()->text(GREEN_BOLD(" * ") WHITE_BOLD("POOL #%-7zu") "\x1B[1;%dm%s\x1B[0m variant " WHITE_BOLD("%s"),
                           i,
                           color,
                           pool.url(),
                           pool.algorithm().variantName()
                           );
        }
        else {
            Log::i()->text(" * POOL #%-7zu%s%s variant=%s %s",
                           i,
                           pool.isEnabled() ? "" : "-",
                           pool.url(),
                           pool.algorithm().variantName(),
                           pool.isTLS() ? "TLS" : ""
                           );
        }

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
