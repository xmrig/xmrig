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


#include <assert.h>


#include "base/net/stratum/Client.h"
#include "base/net/stratum/Job.h"
#include "base/net/stratum/strategies/FailoverStrategy.h"
#include "base/net/stratum/strategies/SinglePoolStrategy.h"
#include "base/tools/Buffer.h"
#include "base/tools/Timer.h"
#include "common/crypto/keccak.h"
#include "common/Platform.h"
#include "common/xmrig.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "net/Network.h"
#include "net/strategies/DonateStrategy.h"


namespace xmrig {

static inline double randomf(double min, double max)                 { return (max - min) * (((static_cast<double>(rand())) / static_cast<double>(RAND_MAX))) + min; }
static inline uint64_t random(uint64_t base, double min, double max) { return static_cast<uint64_t>(base * randomf(min, max)); }

static const char *kDonateHost = "donate.v2.xmrig.com";
#ifdef XMRIG_FEATURE_TLS
static const char *kDonateHostTls = "donate.ssl.xmrig.com";
#endif

} /* namespace xmrig */


xmrig::DonateStrategy::DonateStrategy(Controller *controller, IStrategyListener *listener) :
    m_proxy(nullptr),
    m_donateTime(static_cast<uint64_t>(controller->config()->pools().donateLevel()) * 60 * 1000),
    m_idleTime((100 - static_cast<uint64_t>(controller->config()->pools().donateLevel())) * 60 * 1000),
    m_controller(controller),
    m_strategy(nullptr),
    m_listener(listener),
    m_state(STATE_NEW),
    m_now(0),
    m_stop(0)
{
    uint8_t hash[200];
    char userId[65] = { 0 };

    const String &user = controller->config()->pools().data().front().user();
    keccak(reinterpret_cast<const uint8_t *>(user.data()), user.size(), hash);
    Buffer::toHex(hash, 32, userId);

#   ifdef XMRIG_FEATURE_TLS
    m_pools.push_back(Pool(kDonateHostTls, 443, userId, nullptr, false, true, true));
#   endif
    m_pools.push_back(Pool(kDonateHost, 3333, userId, nullptr, false, true));

    for (Pool &pool : m_pools) {
        pool.adjust(Algorithm(controller->config()->algorithm().algo(), VARIANT_AUTO));
    }

    if (m_pools.size() > 1) {
        m_strategy = new FailoverStrategy(m_pools, 1, 2, this, true);
    }
    else {
        m_strategy = new SinglePoolStrategy(m_pools.front(), 1, 2, this, true);
    }

    m_timer = new Timer(this);

    setState(STATE_IDLE);
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

    if (state() == STATE_WAIT && now > m_stop) {
        setState(STATE_IDLE);
    }
}


void xmrig::DonateStrategy::onActive(IStrategy *strategy, Client *client)
{
    if (isActive()) {
        return;
    }

    setState(STATE_ACTIVE);
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
    setState(isActive() ? STATE_WAIT : STATE_CONNECT);
}


void xmrig::DonateStrategy::idle(double min, double max)
{
    m_timer->start(random(m_idleTime, min, max), 0);
}


void xmrig::DonateStrategy::setState(State state)
{
    constexpr const uint64_t waitTime = 3000;

    assert(m_state != state && state != STATE_NEW);
    if (m_state == state) {
        return;
    }

    const State prev = m_state;
    m_state = state;

    switch (state) {
    case STATE_NEW:
        break;

    case STATE_IDLE:
        if (prev == STATE_NEW) {
            idle(0.5, 1.5);
        }
        else {
            m_strategy->stop();
            idle(0.8, 1.2);
        }
        break;

    case STATE_CONNECT:
        connect();
        break;

    case STATE_ACTIVE:
        m_timer->start(m_donateTime, 0);
        break;

    case STATE_WAIT:
        m_stop = m_now + waitTime;
        m_listener->onPause(this);
        break;
    }
}
