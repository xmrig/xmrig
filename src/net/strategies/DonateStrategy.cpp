/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include <cassert>
#include <iterator>


#include "net/strategies/DonateStrategy.h"
#include "3rdparty/rapidjson/document.h"
#include "base/crypto/keccak.h"
#include "base/kernel/Platform.h"
#include "base/net/stratum/Client.h"
#include "base/net/stratum/Job.h"
#include "base/net/stratum/strategies/FailoverStrategy.h"
#include "base/net/stratum/strategies/SinglePoolStrategy.h"
#include "base/tools/Buffer.h"
#include "base/tools/Cvt.h"
#include "base/tools/Timer.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "core/Miner.h"
#include "net/Network.h"


namespace xmrig {

static inline double randomf(double min, double max)                 { return (max - min) * (((static_cast<double>(rand())) / static_cast<double>(RAND_MAX))) + min; }
static inline uint64_t random(uint64_t base, double min, double max) { return static_cast<uint64_t>(base * randomf(min, max)); }

static const char *kDonateHost = "donate.v2.xmrig.com";
#ifdef XMRIG_FEATURE_TLS
static const char *kDonateHostTls = "donate.ssl.xmrig.com";
#endif

} /* namespace xmrig */


xmrig::DonateStrategy::DonateStrategy(Controller *controller, IStrategyListener *listener) :
    m_donateTime(static_cast<uint64_t>(controller->config()->pools().donateLevel()) * 60 * 1000),
    m_idleTime((100 - static_cast<uint64_t>(controller->config()->pools().donateLevel())) * 60 * 1000),
    m_controller(controller),
    m_listener(listener)
{
    uint8_t hash[200];

    const auto &user = controller->config()->pools().data().front().user();
    keccak(reinterpret_cast<const uint8_t *>(user.data()), user.size(), hash);
    Cvt::toHex(m_userId, sizeof(m_userId), hash, 32);

#   ifdef XMRIG_ALGO_KAWPOW
    constexpr Pool::Mode mode = Pool::MODE_AUTO_ETH;
#   else
    constexpr Pool::Mode mode = Pool::MODE_POOL;
#   endif

#   ifdef XMRIG_FEATURE_TLS
    m_pools.emplace_back(kDonateHostTls, 443, m_userId, nullptr, nullptr, 0, true, true, mode);
#   endif
    m_pools.emplace_back(kDonateHost, 3333, m_userId, nullptr, nullptr, 0, true, false, mode);

    if (m_pools.size() > 1) {
        m_strategy = new FailoverStrategy(m_pools, 10, 2, this, true);
    }
    else {
        m_strategy = new SinglePoolStrategy(m_pools.front(), 10, 2, this, true);
    }

    m_timer = new Timer(this);

    setState(STATE_IDLE);
}


xmrig::DonateStrategy::~DonateStrategy()
{
    delete m_timer;
    delete m_strategy;

    if (m_proxy) {
        m_proxy->deleteLater();
    }
}


int64_t xmrig::DonateStrategy::submit(const JobResult &result)
{
    return m_proxy ? m_proxy->submit(result) : m_strategy->submit(result);
}


void xmrig::DonateStrategy::connect()
{
    m_proxy = createProxy();
    if (m_proxy) {
        m_proxy->connect();
    }

    else {
        m_strategy->connect();
    }
}


void xmrig::DonateStrategy::setAlgo(const xmrig::Algorithm &algo)
{
    m_algorithm = algo;

    m_strategy->setAlgo(algo);
}


void xmrig::DonateStrategy::setProxy(const ProxyUrl &proxy)
{
    m_strategy->setProxy(proxy);
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

    if (m_proxy) {
        m_proxy->tick(now);
    }

    if (state() == STATE_WAIT && now > m_timestamp) {
        setState(STATE_IDLE);
    }
}


void xmrig::DonateStrategy::onActive(IStrategy *, IClient *client)
{
    if (isActive()) {
        return;
    }

    setState(STATE_ACTIVE);
    m_listener->onActive(this, client);
}


void xmrig::DonateStrategy::onPause(IStrategy *)
{
}


void xmrig::DonateStrategy::onClose(IClient *, int failures)
{
    if (failures == 2 && m_controller->config()->pools().proxyDonate() == Pools::PROXY_DONATE_AUTO) {
        m_proxy->deleteLater();
        m_proxy = nullptr;

        m_strategy->connect();
    }
}


void xmrig::DonateStrategy::onLogin(IClient *, rapidjson::Document &doc, rapidjson::Value &params)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

#   ifdef XMRIG_FEATURE_TLS
    if (m_tls) {
        char buf[40] = { 0 };
        snprintf(buf, sizeof(buf), "stratum+ssl://%s", m_pools[0].url().data());
        params.AddMember("url", Value(buf, allocator), allocator);
    }
    else {
        params.AddMember("url", m_pools[1].url().toJSON(), allocator);
    }
#   else
    params.AddMember("url", m_pools[0].url().toJSON(), allocator);
#   endif

    setAlgorithms(doc, params);
}


void xmrig::DonateStrategy::onLogin(IStrategy *, IClient *, rapidjson::Document &doc, rapidjson::Value &params)
{
    setAlgorithms(doc, params);
}


void xmrig::DonateStrategy::onLoginSuccess(IClient *client)
{
    if (isActive()) {
        return;
    }

    setState(STATE_ACTIVE);
    m_listener->onActive(this, client);
}


void xmrig::DonateStrategy::onVerifyAlgorithm(const IClient *client, const Algorithm &algorithm, bool *ok)
{
    m_listener->onVerifyAlgorithm(this, client, algorithm, ok);
}


void xmrig::DonateStrategy::onVerifyAlgorithm(IStrategy *, const  IClient *client, const Algorithm &algorithm, bool *ok)
{
    m_listener->onVerifyAlgorithm(this, client, algorithm, ok);
}


void xmrig::DonateStrategy::onTimer(const Timer *)
{
    setState(isActive() ? STATE_WAIT : STATE_CONNECT);
}


xmrig::IClient *xmrig::DonateStrategy::createProxy()
{
    if (m_controller->config()->pools().proxyDonate() == Pools::PROXY_DONATE_NONE) {
        return nullptr;
    }

    IStrategy *strategy = m_controller->network()->strategy();
    if (!strategy->isActive() || !strategy->client()->hasExtension(IClient::EXT_CONNECT)) {
        return nullptr;
    }

    const IClient *client = strategy->client();
    m_tls                 = client->hasExtension(IClient::EXT_TLS);

    Pool pool(client->pool().proxy().isValid() ? client->pool().host() : client->ip(), client->pool().port(), m_userId, client->pool().password(), client->pool().spendSecretKey(), 0, true, client->isTLS(), Pool::MODE_POOL);
    pool.setAlgo(client->pool().algorithm());
    pool.setProxy(client->pool().proxy());

    IClient *proxy = new Client(-1, Platform::userAgent(), this);
    proxy->setPool(pool);
    proxy->setQuiet(true);

    return proxy;
}


void xmrig::DonateStrategy::idle(double min, double max)
{
    m_timer->start(random(m_idleTime, min, max), 0);
}


void xmrig::DonateStrategy::setAlgorithms(rapidjson::Document &doc, rapidjson::Value &params)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Algorithms algorithms = m_controller->miner()->algorithms();
    const size_t index = static_cast<size_t>(std::distance(algorithms.begin(), std::find(algorithms.begin(), algorithms.end(), m_algorithm)));
    if (index > 0 && index < algorithms.size()) {
        std::swap(algorithms[0], algorithms[index]);
    }

    Value algo(kArrayType);

    for (const auto &a : algorithms) {
        algo.PushBack(StringRef(a.name()), allocator);
    }

    params.AddMember("algo", algo, allocator);
}


void xmrig::DonateStrategy::setJob(IClient *client, const Job &job, const rapidjson::Value &params)
{
    if (isActive()) {
        m_listener->onJob(this, client, job, params);
    }
}


void xmrig::DonateStrategy::setResult(IClient *client, const SubmitResult &result, const char *error)
{
    m_listener->onResultAccepted(this, client, result, error);
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
        else if (prev == STATE_CONNECT) {
            m_timer->start(20000, 0);
        }
        else {
            m_strategy->stop();
            if (m_proxy) {
                m_proxy->deleteLater();
                m_proxy = nullptr;
            }

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
        m_timestamp = m_now + waitTime;
        m_listener->onPause(this);
        break;
    }
}
