/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#include "base/io/Json.h"
#include "base/net/Pool.h"
#include "rapidjson/document.h"


#ifdef APP_DEBUG
#   include "common/log/Log.h"
#endif


#ifdef _MSC_VER
#   define strncasecmp _strnicmp
#   define strcasecmp  _stricmp
#endif


namespace xmrig {

static const char *kEnabled     = "enabled";
static const char *kFingerprint = "tls-fingerprint";
static const char *kKeepalive   = "keepalive";
static const char *kNicehash    = "nicehash";
static const char *kPass        = "pass";
static const char *kRigId       = "rig-id";
static const char *kTls         = "tls";
static const char *kUrl         = "url";
static const char *kUser        = "user";
static const char *kVariant     = "variant";

}


xmrig::Pool::Pool() :
    m_enabled(true),
    m_nicehash(false),
    m_tls(false),
    m_keepAlive(0),
    m_port(kDefaultPort)
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
    m_enabled(true),
    m_nicehash(false),
    m_tls(false),
    m_keepAlive(0),
    m_port(kDefaultPort)
{
    parse(url);
}


xmrig::Pool::Pool(const rapidjson::Value &object) :
    m_enabled(true),
    m_nicehash(false),
    m_tls(false),
    m_keepAlive(0),
    m_port(kDefaultPort)
{
    if (!parse(Json::getString(object, kUrl))) {
        return;
    }

    setUser(Json::getString(object, kUser));
    setPassword(Json::getString(object, kPass));
    setRigId(Json::getString(object, kRigId));
    setNicehash(Json::getBool(object, kNicehash));

    const rapidjson::Value &keepalive = object[kKeepalive];
    if (keepalive.IsInt()) {
        setKeepAlive(keepalive.GetInt());
    }
    else if (keepalive.IsBool()) {
        setKeepAlive(keepalive.GetBool());
    }

    const rapidjson::Value &variant = object[kVariant];
    if (variant.IsString()) {
        algorithm().parseVariant(variant.GetString());
    }
    else if (variant.IsInt()) {
        algorithm().parseVariant(variant.GetInt());
    }

    m_enabled     = Json::getBool(object, kEnabled, true);
    m_tls         = Json::getBool(object, kTls);
    m_fingerprint = Json::getString(object, kFingerprint);
}


xmrig::Pool::Pool(const char *host, uint16_t port, const char *user, const char *password, int keepAlive, bool nicehash, bool tls) :
    m_enabled(true),
    m_nicehash(nicehash),
    m_tls(tls),
    m_keepAlive(keepAlive),
    m_host(host),
    m_password(password),
    m_user(user),
    m_port(port)
{
    const size_t size = m_host.size() + 8;
    assert(size > 8);

    char *url = new char[size]();
    snprintf(url, size - 1, "%s:%d", m_host.data(), m_port);

    m_url = url;
}


bool xmrig::Pool::isCompatible(const Algorithm &algorithm) const
{
    if (m_algorithms.empty()) {
        return true;
    }

    for (const auto &a : m_algorithms) {
        if (algorithm == a) {
            return true;
        }
    }

#   ifdef XMRIG_PROXY_PROJECT
    if (m_algorithm.algo() == xmrig::CRYPTONIGHT && algorithm.algo() == xmrig::CRYPTONIGHT) {
        return m_algorithm.variant() == xmrig::VARIANT_XTL || m_algorithm.variant() == xmrig::VARIANT_MSR;
    }
#   endif

    return false;
}


bool xmrig::Pool::isEnabled() const
{
#   ifdef XMRIG_NO_TLS
    if (isTLS()) {
        return false;
    }
#   endif

    return m_enabled && isValid() && algorithm().isValid();
}


bool xmrig::Pool::isEqual(const Pool &other) const
{
    return (m_nicehash       == other.m_nicehash
            && m_enabled     == other.m_enabled
            && m_tls         == other.m_tls
            && m_keepAlive   == other.m_keepAlive
            && m_port        == other.m_port
            && m_algorithm   == other.m_algorithm
            && m_fingerprint == other.m_fingerprint
            && m_host        == other.m_host
            && m_password    == other.m_password
            && m_rigId       == other.m_rigId
            && m_url         == other.m_url
            && m_user        == other.m_user);
}


bool xmrig::Pool::parse(const char *url)
{
    assert(url != nullptr);

    const char *p = strstr(url, "://");
    const char *base = url;

    if (p) {
        if (strncasecmp(url, "stratum+tcp://", 14) == 0) {
            m_tls = false;
        }
        else if (strncasecmp(url, "stratum+ssl://", 14) == 0) {
            m_tls = true;
        }
        else {
            return false;
        }

        base = url + 14;
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


bool xmrig::Pool::setUserpass(const char *userpass)
{
    const char *p = strchr(userpass, ':');
    if (!p) {
        return false;
    }

    char *user = new char[p - userpass + 1]();
    strncpy(user, userpass, static_cast<size_t>(p - userpass));

    m_user     = user;
    m_password = p + 1;

    return true;
}


rapidjson::Value xmrig::Pool::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kUrl),   m_url.toJSON(), allocator);
    obj.AddMember(StringRef(kUser),  m_user.toJSON(), allocator);
    obj.AddMember(StringRef(kPass),  m_password.toJSON(), allocator);
    obj.AddMember(StringRef(kRigId), m_rigId.toJSON(), allocator);

#   ifndef XMRIG_PROXY_PROJECT
    obj.AddMember(StringRef(kNicehash), isNicehash(), allocator);
#   endif

    if (m_keepAlive == 0 || m_keepAlive == kKeepAliveTimeout) {
        obj.AddMember(StringRef(kKeepalive), m_keepAlive > 0, allocator);
    }
    else {
        obj.AddMember(StringRef(kKeepalive), m_keepAlive, allocator);
    }

    obj.AddMember(StringRef(kVariant), StringRef(m_algorithm.variantName()), allocator);

    obj.AddMember(StringRef(kEnabled),     m_enabled, allocator);
    obj.AddMember(StringRef(kTls),         isTLS(), allocator);
    obj.AddMember(StringRef(kFingerprint), m_fingerprint.toJSON(), allocator);

    return obj;
}


void xmrig::Pool::adjust(const Algorithm &algorithm)
{
    if (!isValid()) {
        return;
    }

    if (!m_algorithm.isValid()) {
        m_algorithm.setAlgo(algorithm.algo());
        adjustVariant(algorithm.variant());
    }

    rebuild();
}


void xmrig::Pool::setAlgo(const xmrig::Algorithm &algorithm)
{
    m_algorithm = algorithm;

    rebuild();
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
    LOG_DEBUG ("nicehash:  %d", static_cast<int>(m_nicehash));
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


void xmrig::Pool::addVariant(xmrig::Variant variant)
{
    const xmrig::Algorithm algorithm(m_algorithm.algo(), variant);
    if (!algorithm.isValid() || m_algorithm == algorithm) {
        return;
    }

    m_algorithms.push_back(algorithm);
}


void xmrig::Pool::adjustVariant(const xmrig::Variant variantHint)
{
#   ifndef XMRIG_PROXY_PROJECT
    using namespace xmrig;

    if (variantHint != VARIANT_AUTO) {
        m_algorithm.setVariant(variantHint);
        return;
    }

    if (m_algorithm.variant() != VARIANT_AUTO) {
        return;
    }
#   endif
}


void xmrig::Pool::rebuild()
{
    m_algorithms.clear();

    if (!m_algorithm.isValid()) {
        return;
    }

    m_algorithms.push_back(m_algorithm);

#   ifndef XMRIG_PROXY_PROJECT
    addVariant(VARIANT_AUTO);
    addVariant(VARIANT_CHUKWA);
    addVariant(VARIANT_CHUKWA_LITE);
#   endif
}
