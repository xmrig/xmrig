/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      Howard Chu  <https://github.com/hyc>
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
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#include "base/io/json/Json.h"
#include "base/net/stratum/Pool.h"
#include "rapidjson/document.h"


#ifdef APP_DEBUG
#   include "base/io/log/Log.h"
#endif


#ifdef _MSC_VER
#   define strncasecmp _strnicmp
#endif


namespace xmrig {

static const char *kAlgo                   = "algo";
static const char *kDaemon                 = "daemon";
static const char *kDaemonPollInterval     = "daemon-poll-interval";
static const char *kEnabled                = "enabled";
static const char *kFingerprint            = "tls-fingerprint";
static const char *kKeepalive              = "keepalive";
static const char *kNicehash               = "nicehash";
static const char *kPass                   = "pass";
static const char *kRigId                  = "rig-id";
static const char *kTls                    = "tls";
static const char *kUrl                    = "url";
static const char *kUser                   = "user";

const String Pool::kDefaultPassword        = "x";
const String Pool::kDefaultUser            = "x";

static const char kStratumTcp[]            = "stratum+tcp://";
static const char kStratumSsl[]            = "stratum+ssl://";

#ifdef XMRIG_FEATURE_HTTP
static const char kDaemonHttp[]            = "daemon+http://";
static const char kDaemonHttps[]           = "daemon+https://";
#endif

}


xmrig::Pool::Pool() :
    m_keepAlive(0),
    m_flags(0),
    m_port(kDefaultPort),
    m_pollInterval(kDefaultPollInterval)
{
}


/**
 * @brief Parse url.
 *
 * Valid urls:
 * example.com
 * example.com:3333
 * stratum+tcp://example.com
 * stratum+tcp://example.com:3333
 *
 * @param url
 */
xmrig::Pool::Pool(const char *url) :
    m_keepAlive(0),
    m_flags(1),
    m_port(kDefaultPort),
    m_pollInterval(kDefaultPollInterval)
{
    parse(url);
}


xmrig::Pool::Pool(const rapidjson::Value &object) :
    m_keepAlive(0),
    m_flags(1),
    m_port(kDefaultPort),
    m_pollInterval(kDefaultPollInterval)
{
    if (!parse(Json::getString(object, kUrl))) {
        return;
    }

    m_user         = Json::getString(object, kUser);
    m_password     = Json::getString(object, kPass);
    m_rigId        = Json::getString(object, kRigId);
    m_fingerprint  = Json::getString(object, kFingerprint);
    m_pollInterval = Json::getUint64(object, kDaemonPollInterval, kDefaultPollInterval);
    m_algorithm    = Json::getString(object, kAlgo);

    m_flags.set(FLAG_ENABLED,  Json::getBool(object, kEnabled, true));
    m_flags.set(FLAG_NICEHASH, Json::getBool(object, kNicehash));
    m_flags.set(FLAG_TLS,      Json::getBool(object, kTls, m_flags.test(FLAG_TLS)));
    m_flags.set(FLAG_DAEMON,   Json::getBool(object, kDaemon, m_flags.test(FLAG_DAEMON)));

    const rapidjson::Value &keepalive = Json::getValue(object, kKeepalive);
    if (keepalive.IsInt()) {
        setKeepAlive(keepalive.GetInt());
    }
    else if (keepalive.IsBool()) {
        setKeepAlive(keepalive.GetBool());
    }
}


xmrig::Pool::Pool(const char *host, uint16_t port, const char *user, const char *password, int keepAlive, bool nicehash, bool tls) :
    m_keepAlive(keepAlive),
    m_flags(1),
    m_host(host),
    m_password(password),
    m_user(user),
    m_port(port),
    m_pollInterval(kDefaultPollInterval)
{
    const size_t size = m_host.size() + 8;
    assert(size > 8);

    char *url = new char[size]();
    snprintf(url, size - 1, "%s:%d", m_host.data(), m_port);

    m_url = url;

    m_flags.set(FLAG_NICEHASH, nicehash);
    m_flags.set(FLAG_TLS,      tls);
}


bool xmrig::Pool::isEnabled() const
{
#   ifndef XMRIG_FEATURE_TLS
    if (isTLS()) {
        return false;
    }
#   endif

#   ifndef XMRIG_FEATURE_HTTP
    if (isDaemon()) {
        return false;
    }
#   endif

    if (isDaemon() && !algorithm().isValid()) {
        return false;
    }

    return m_flags.test(FLAG_ENABLED) && isValid();
}


bool xmrig::Pool::isEqual(const Pool &other) const
{
    return (m_flags           == other.m_flags
            && m_keepAlive    == other.m_keepAlive
            && m_port         == other.m_port
            && m_algorithm    == other.m_algorithm
            && m_fingerprint  == other.m_fingerprint
            && m_host         == other.m_host
            && m_password     == other.m_password
            && m_rigId        == other.m_rigId
            && m_url          == other.m_url
            && m_user         == other.m_user
            && m_pollInterval == other.m_pollInterval
            );
}


bool xmrig::Pool::parse(const char *url)
{
    assert(url != nullptr);

    const char *p    = strstr(url, "://");
    const char *base = url;

    if (p) {
        if (strncasecmp(url, kStratumTcp, sizeof(kStratumTcp) - 1) == 0) {
            m_flags.set(FLAG_DAEMON, false);
            m_flags.set(FLAG_TLS,    false);
        }
        else if (strncasecmp(url, kStratumSsl, sizeof(kStratumSsl) - 1) == 0) {
            m_flags.set(FLAG_DAEMON, false);
            m_flags.set(FLAG_TLS,    true);
        }
#       ifdef XMRIG_FEATURE_HTTP
        else if (strncasecmp(url, kDaemonHttps, sizeof(kDaemonHttps) - 1) == 0) {
            m_flags.set(FLAG_DAEMON, true);
            m_flags.set(FLAG_TLS,    true);
        }
        else if (strncasecmp(url, kDaemonHttp, sizeof(kDaemonHttp) - 1) == 0) {
            m_flags.set(FLAG_DAEMON, true);
            m_flags.set(FLAG_TLS,    false);
        }
#       endif
        else {
            return false;
        }

        base = p + 3;
    }

    if (!strlen(base) || *base == '/') {
        return false;
    }

    m_url = url;
    if (base[0] == '[') {
        return parseIPv6(base);
    }

    const char *port = strchr(base, ':');
    if (!port) {
        m_host = base;
        return true;
    }

    const size_t size = static_cast<size_t>(port++ - base + 1);
    char *host        = new char[size]();
    memcpy(host, base, size - 1);

    m_host = host;
    m_port = static_cast<uint16_t>(strtol(port, nullptr, 10));

    return true;
}


rapidjson::Value xmrig::Pool::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kAlgo),  m_algorithm.toJSON(), allocator);
    obj.AddMember(StringRef(kUrl),   m_url.toJSON(), allocator);
    obj.AddMember(StringRef(kUser),  m_user.toJSON(), allocator);

    if (!isDaemon()) {
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

    obj.AddMember(StringRef(kEnabled),            m_flags.test(FLAG_ENABLED), allocator);
    obj.AddMember(StringRef(kTls),                isTLS(), allocator);
    obj.AddMember(StringRef(kFingerprint),        m_fingerprint.toJSON(), allocator);
    obj.AddMember(StringRef(kDaemon),             m_flags.test(FLAG_DAEMON), allocator);

    if (isDaemon()) {
        obj.AddMember(StringRef(kDaemonPollInterval), m_pollInterval, allocator);
    }

    return obj;
}


#ifdef APP_DEBUG
void xmrig::Pool::print() const
{
    LOG_NOTICE("url:       %s", m_url.data());
    LOG_DEBUG ("host:      %s", m_host.data());
    LOG_DEBUG ("port:      %d", static_cast<int>(m_port));
    LOG_DEBUG ("user:      %s", m_user.data());
    LOG_DEBUG ("pass:      %s", m_password.data());
    LOG_DEBUG ("rig-id     %s", m_rigId.data());
    LOG_DEBUG ("algo:      %s", m_algorithm.name());
    LOG_DEBUG ("nicehash:  %d", static_cast<int>(m_flags.test(FLAG_NICEHASH)));
    LOG_DEBUG ("keepAlive: %d", m_keepAlive);
}
#endif


bool xmrig::Pool::parseIPv6(const char *addr)
{
    const char *end = strchr(addr, ']');
    if (!end) {
        return false;
    }

    const char *port = strchr(end, ':');
    if (!port) {
        return false;
    }

    const size_t size = static_cast<size_t>(end - addr);
    char *host        = new char[size]();
    memcpy(host, addr + 1, size - 1);

    m_host = host;
    m_port = static_cast<uint16_t>(strtol(port + 1, nullptr, 10));

    return true;
}
