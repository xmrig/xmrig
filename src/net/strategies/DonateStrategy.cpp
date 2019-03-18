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


#include "base/net/stratum/Client.h"
#include "base/net/stratum/Job.h"
#include "base/net/stratum/strategies/FailoverStrategy.h"
#include "base/net/stratum/strategies/SinglePoolStrategy.h"
#include "base/tools/Buffer.h"
#include "base/tools/Timer.h"
#include "common/crypto/keccak.h"
#include "common/Platform.h"
#include "common/xmrig.h"
#include "net/strategies/DonateStrategy.h"


namespace xmrig {

static inline double randomf(double min, double max)                  { return (max - min) * (((static_cast<double>(rand())) / static_cast<double>(RAND_MAX))) + min; }
static inline uint64_t random(uint64_t base, double min, double max) { return static_cast<uint64_t>(base * randomf(min, max)); }

} /* namespace xmrig */


xmrig::DonateStrategy::DonateStrategy(int level, const char *user, Algo algo, IStrategyListener *listener) :
    m_active(false),
    m_donateTime(static_cast<uint64_t>(level) * 60 * 1000),
    m_idleTime((100 - static_cast<uint64_t>(level)) * 60 * 1000),
    m_strategy(nullptr),
    m_listener(listener),
    m_now(0),
    m_stop(0)
{
    uint8_t hash[200];
    char userId[65] = { 0 };

    keccak(reinterpret_cast<const uint8_t *>(user), strlen(user), hash);
    Buffer::toHex(hash, 32, userId);

#   ifdef XMRIG_FEATURE_TLS
    m_pools.push_back(Pool("donate.ssl.xmrig.com", 443, userId, nullptr, false, true, true));
#   endif

    m_pools.push_back(Pool("donate.v2.xmrig.com", 3333, userId, nullptr, false, true));

    for (Pool &pool : m_pools) {
        pool.adjust(Algorithm(algo, VARIANT_AUTO));
    }

    if (m_pools.size() > 1) {
        m_strategy = new FailoverStrategy(m_pools, 1, 2, this, true);
    }
    else {
        m_strategy = new SinglePoolStrategy(m_pools.front(), 1, 2, this, true);
    }

    m_timer = new Timer(this);

    idle(random(m_idleTime, 0.5, 1.5));
}


xmrig::DonateStrategy::~DonateStrategy()
{
    delete m_timer;
    delete m_strategy;
}


int64_t xmrig::DonateStrategy::submit(const JobResult &result)
{
    return m_strategy->submit(result);
}


void xmrig::DonateStrategy::connect()
{
    m_strategy->connect();
}


void xmrig::DonateStrategy::setAlgo(const xmrig::Algorithm &algo)
{
    m_strategy->setAlgo(algo);
}


void xmrig::DonateStrategy::stop()
{
    m_timer->stop();
    m_strategy->stop();
}


void xmrig::DonateStrategy::tick(uint64_t now)
{
    m_now = now;

    m_strategy->tick(now);

    if (m_stop && now > m_stop) {
        m_strategy->stop();
        m_stop = 0;
    }
}


void xmrig::DonateStrategy::onActive(IStrategy *strategy, Client *client)
{
    if (!isActive()) {
        m_timer->start(m_donateTime, 0);
    }

    m_active = true;
    m_listener->onActive(this, client);
}


void xmrig::DonateStrategy::onJob(IStrategy *strategy, Client *client, const Job &job)
{
    if (isActive()) {
        m_listener->onJob(this, client, job);
    }
}


void xmrig::DonateStrategy::onPause(IStrategy *strategy)
{
}


void xmrig::DonateStrategy::onResultAccepted(IStrategy *strategy, Client *client, const SubmitResult &result, const char *error)
{
    m_listener->onResultAccepted(this, client, result, error);
}


void xmrig::DonateStrategy::onTimer(const Timer *)
{
    if (!isActive()) {
        return connect();
    }

    suspend();
}


void xmrig::DonateStrategy::idle(uint64_t timeout)
{
    m_timer->start(timeout, 0);
}


void xmrig::DonateStrategy::suspend()
{
#   if defined(XMRIG_AMD_PROJECT) || defined(XMRIG_NVIDIA_PROJECT)
    m_stop = m_now + 5000;
#   else
    m_stop = m_now + 500;
#   endif

    m_active = false;
    m_listener->onPause(this);

    idle(random(m_idleTime, 0.8, 1.2));
}
