/* XMRig
 * Copyright (c) 2019      Howard Chu  <https://github.com/hyc>
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

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>


#include "base/net/stratum/Pool.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "base/kernel/Platform.h"
#include "base/net/stratum/Client.h"

#ifdef XMRIG_ALGO_KAWPOW
#   include "base/net/stratum/AutoClient.h"
#   include "base/net/stratum/EthStratumClient.h"
#endif


#ifdef XMRIG_FEATURE_HTTP
#   include "base/net/stratum/DaemonClient.h"
#   include "base/net/stratum/SelfSelectClient.h"
#endif


#ifdef XMRIG_FEATURE_BENCHMARK
#   include "base/net/stratum/benchmark/BenchClient.h"
#   include "base/net/stratum/benchmark/BenchConfig.h"
#endif


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
#endif


namespace xmrig {


const String Pool::kDefaultPassword       = "x";
const String Pool::kDefaultUser           = "x";


const char *Pool::kAlgo                   = "algo";
const char *Pool::kCoin                   = "coin";
const char *Pool::kDaemon                 = "daemon";
const char *Pool::kDaemonPollInterval     = "daemon-poll-interval";
const char *Pool::kDaemonZMQPort          = "daemon-zmq-port";
const char *Pool::kEnabled                = "enabled";
const char *Pool::kFingerprint            = "tls-fingerprint";
const char *Pool::kKeepalive              = "keepalive";
const char *Pool::kNicehash               = "nicehash";
const char *Pool::kPass                   = "pass";
const char *Pool::kRigId                  = "rig-id";
const char *Pool::kSelfSelect             = "self-select";
const char *Pool::kSOCKS5                 = "socks5";
const char *Pool::kSubmitToOrigin         = "submit-to-origin";
const char *Pool::kTls                    = "tls";
const char *Pool::kUrl                    = "url";
const char *Pool::kUser                   = "user";
const char *Pool::kSpendSecretKey         = "spend-secret-key";
const char *Pool::kNicehashHost           = "nicehash.com";


}


xmrig::Pool::Pool(const char *url) :
    m_flags(1 << FLAG_ENABLED),
    m_pollInterval(kDefaultPollInterval),
    m_url(url)
{
}


xmrig::Pool::Pool(const char *host, uint16_t port, const char *user, const char *password, const char* spendSecretKey, int keepAlive, bool nicehash, bool tls, Mode mode) :
    m_keepAlive(keepAlive),
    m_mode(mode),
    m_flags(1 << FLAG_ENABLED),
    m_password(password),
    m_user(user),
    m_spendSecretKey(spendSecretKey),
    m_pollInterval(kDefaultPollInterval),
    m_url(host, port, tls)
{
    m_flags.set(FLAG_NICEHASH, nicehash || strstr(host, kNicehashHost));
    m_flags.set(FLAG_TLS,      tls);
}


xmrig::Pool::Pool(const rapidjson::Value &object) :
    m_flags(1 << FLAG_ENABLED),
    m_pollInterval(kDefaultPollInterval),
    m_url(Json::getString(object, kUrl))
{
    if (!m_url.isValid()) {
        return;
    }

    m_user           = Json::getString(object, kUser);
    m_spendSecretKey = Json::getString(object, kSpendSecretKey);
    m_password       = Json::getString(object, kPass);
    m_rigId          = Json::getString(object, kRigId);
    m_fingerprint    = Json::getString(object, kFingerprint);
    m_pollInterval   = Json::getUint64(object, kDaemonPollInterval, kDefaultPollInterval);
    m_algorithm      = Json::getString(object, kAlgo);
    m_coin           = Json::getString(object, kCoin);
    m_daemon         = Json::getString(object, kSelfSelect);
    m_proxy          = Json::getValue(object, kSOCKS5);
    m_zmqPort        = Json::getInt(object, kDaemonZMQPort, m_zmqPort);

    m_flags.set(FLAG_ENABLED,  Json::getBool(object, kEnabled, true));
    m_flags.set(FLAG_NICEHASH, Json::getBool(object, kNicehash) || m_url.host().contains(kNicehashHost));
    m_flags.set(FLAG_TLS,      Json::getBool(object, kTls) || m_url.isTLS());

    setKeepAlive(Json::getValue(object, kKeepalive));

    if (m_daemon.isValid()) {
        m_mode           = MODE_SELF_SELECT;
        m_submitToOrigin = Json::getBool(object, kSubmitToOrigin, m_submitToOrigin);
    }
    else if (Json::getBool(object, kDaemon)) {
        m_mode = MODE_DAEMON;
    }
}


#ifdef XMRIG_FEATURE_BENCHMARK
xmrig::Pool::Pool(const std::shared_ptr<BenchConfig> &benchmark) :
    m_mode(MODE_BENCHMARK),
    m_flags(1 << FLAG_ENABLED),
    m_url(BenchConfig::kBenchmark),
    m_benchmark(benchmark)
{
}


xmrig::BenchConfig *xmrig::Pool::benchmark() const
{
    assert(m_mode == MODE_BENCHMARK && m_benchmark);

    return m_benchmark.get();
}


uint32_t xmrig::Pool::benchSize() const
{
    return benchmark()->size();
}
#endif


bool xmrig::Pool::isEnabled() const
{
#   ifndef XMRIG_FEATURE_TLS
    if (isTLS()) {
        return false;
    }
#   endif

#   ifndef XMRIG_FEATURE_HTTP
    if (m_mode == MODE_DAEMON) {
        return false;
    }
#   endif

#   ifndef XMRIG_FEATURE_HTTP
    if (m_mode == MODE_SELF_SELECT) {
        return false;
    }
#   endif

    if (m_mode == MODE_DAEMON && (!algorithm().isValid() && !coin().isValid())) {
        return false;
    }

    return m_flags.test(FLAG_ENABLED) && isValid();
}


bool xmrig::Pool::isEqual(const Pool &other) const
{
    return (m_flags           == other.m_flags
            && m_keepAlive    == other.m_keepAlive
            && m_algorithm    == other.m_algorithm
            && m_coin         == other.m_coin
            && m_mode         == other.m_mode
            && m_fingerprint  == other.m_fingerprint
            && m_password     == other.m_password
            && m_rigId        == other.m_rigId
            && m_url          == other.m_url
            && m_user         == other.m_user
            && m_pollInterval == other.m_pollInterval
            && m_daemon       == other.m_daemon
            && m_proxy        == other.m_proxy
            );
}


xmrig::IClient *xmrig::Pool::createClient(int id, IClientListener *listener) const
{
    IClient *client = nullptr;

    if (m_mode == MODE_POOL) {
#       ifdef XMRIG_ALGO_KAWPOW
        if ((m_algorithm.family() == Algorithm::KAWPOW) || (m_coin == Coin::RAVEN)) {
            client = new EthStratumClient(id, Platform::userAgent(), listener);
        }
        else
#       endif
        {
            client = new Client(id, Platform::userAgent(), listener);
        }
    }
#   ifdef XMRIG_FEATURE_HTTP
    else if (m_mode == MODE_DAEMON) {
        client = new DaemonClient(id, listener);
    }
    else if (m_mode == MODE_SELF_SELECT) {
        client = new SelfSelectClient(id, Platform::userAgent(), listener, m_submitToOrigin);
    }
#   endif
#   ifdef XMRIG_ALGO_KAWPOW
    else if (m_mode == MODE_AUTO_ETH) {
        client = new AutoClient(id, Platform::userAgent(), listener);
    }
#   endif
#   ifdef XMRIG_FEATURE_BENCHMARK
    else if (m_mode == MODE_BENCHMARK) {
        client = new BenchClient(m_benchmark, listener);
    }
#   endif

    assert(client != nullptr);

    if (client) {
        client->setPool(*this);
    }

    return client;
}


rapidjson::Value xmrig::Pool::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kAlgo),  m_algorithm.toJSON(), allocator);
    obj.AddMember(StringRef(kCoin),  m_coin.toJSON(), allocator);
    obj.AddMember(StringRef(kUrl),   url().toJSON(), allocator);
    obj.AddMember(StringRef(kUser),  m_user.toJSON(), allocator);

    if (!m_spendSecretKey.isEmpty()) {
        obj.AddMember(StringRef(kSpendSecretKey), m_spendSecretKey.toJSON(), allocator);
    }

    if (m_mode != MODE_DAEMON) {
        obj.AddMember(StringRef(kPass),  m_password.toJSON(), allocator);
        obj.AddMember(StringRef(kRigId), m_rigId.toJSON(), allocator);

#       ifndef XMRIG_PROXY_PROJECT
        obj.AddMember(StringRef(kNicehash), isNicehash(), allocator);
#       endif

        if (m_keepAlive == 0 || m_keepAlive == kKeepAliveTimeout) {
            obj.AddMember(StringRef(kKeepalive), m_keepAlive > 0, allocator);
        }
        else {
            obj.AddMember(StringRef(kKeepalive), m_keepAlive, allocator);
        }
    }

    obj.AddMember(StringRef(kEnabled),      m_flags.test(FLAG_ENABLED), allocator);
    obj.AddMember(StringRef(kTls),          isTLS(), allocator);
    obj.AddMember(StringRef(kFingerprint),  m_fingerprint.toJSON(), allocator);
    obj.AddMember(StringRef(kDaemon),       m_mode == MODE_DAEMON, allocator);
    obj.AddMember(StringRef(kSOCKS5),       m_proxy.toJSON(doc), allocator);

    if (m_mode == MODE_DAEMON) {
        obj.AddMember(StringRef(kDaemonPollInterval), m_pollInterval, allocator);
        obj.AddMember(StringRef(kDaemonZMQPort), m_zmqPort, allocator);
    }
    else {
        obj.AddMember(StringRef(kSelfSelect),     m_daemon.url().toJSON(), allocator);
        obj.AddMember(StringRef(kSubmitToOrigin), m_submitToOrigin, allocator);
    }

    return obj;
}


std::string xmrig::Pool::printableName() const
{
    std::string out(CSI "1;" + std::to_string(isEnabled() ? (isTLS() ? 32 : 36) : 31) + "m" + url().data() + CLEAR);

    if (m_coin.isValid()) {
        out += std::string(" coin ") + WHITE_BOLD_S + m_coin.name() + CLEAR;
    }
    else {
        out += std::string(" algo ") + WHITE_BOLD_S + (m_algorithm.isValid() ? m_algorithm.name() : "auto") + CLEAR;
    }

    if (m_mode == MODE_SELF_SELECT) {
        out += std::string(" self-select ") + CSI "1;" + std::to_string(m_daemon.isTLS() ? 32 : 36) + "m" + m_daemon.url().data() + WHITE_BOLD_S + (m_submitToOrigin ? " submit-to-origin" : "") + CLEAR;
    }

    return out;
}


#ifdef APP_DEBUG
void xmrig::Pool::print() const
{
    LOG_NOTICE("url:       %s", url().data());
    LOG_DEBUG ("host:      %s", host().data());
    LOG_DEBUG ("port:      %d", static_cast<int>(port()));
    if (m_zmqPort >= 0) {
        LOG_DEBUG("zmq-port:  %d", m_zmqPort);
    }
    LOG_DEBUG ("user:      %s", m_user.data());
    LOG_DEBUG ("pass:      %s", m_password.data());
    LOG_DEBUG ("rig-id     %s", m_rigId.data());
    LOG_DEBUG ("algo:      %s", m_algorithm.name());
    LOG_DEBUG ("nicehash:  %d", static_cast<int>(m_flags.test(FLAG_NICEHASH)));
    LOG_DEBUG ("keepAlive: %d", m_keepAlive);
}
#endif


void xmrig::Pool::setKeepAlive(const rapidjson::Value &value)
{
    if (value.IsInt()) {
        setKeepAlive(value.GetInt());
    }
    else if (value.IsBool()) {
        setKeepAlive(value.GetBool());
    }
}
