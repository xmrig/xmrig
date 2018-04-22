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
#include <stdlib.h>
#include <stdio.h>


#include "net/Url.h"
#include "xmrig.h"


#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


Url::Url() :
    m_keepAlive(false),
    m_nicehash(false),
    m_host(nullptr),
    m_password(nullptr),
    m_user(nullptr),
    m_algo(xmrig::ALGO_CRYPTONIGHT),
    m_variant(xmrig::VARIANT_AUTO),
    m_url(nullptr),
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
Url::Url(const char *url) :
    m_keepAlive(false),
    m_nicehash(false),
    m_host(nullptr),
    m_password(nullptr),
    m_user(nullptr),
    m_algo(xmrig::ALGO_CRYPTONIGHT),
    m_variant(xmrig::VARIANT_AUTO),
    m_url(nullptr),
    m_port(kDefaultPort)
{
    parse(url);
}


Url::Url(const char *host, uint16_t port, const char *user, const char *password, bool keepAlive, bool nicehash, int variant) :
    m_keepAlive(keepAlive),
    m_nicehash(nicehash),
    m_password(password ? strdup(password) : nullptr),
    m_user(user ? strdup(user) : nullptr),
    m_algo(xmrig::ALGO_CRYPTONIGHT),
    m_variant(variant),
    m_url(nullptr),
    m_port(port)
{
    m_host = strdup(host);
}


Url::~Url()
{
    free(m_host);
    free(m_password);
    free(m_user);

    if (m_url) {
        delete [] m_url;
    }
}


bool Url::parse(const char *url)
{
    const char *p = strstr(url, "://");
    const char *base = url;

    if (p) {
        if (strncasecmp(url, "stratum+tcp://", 14)) {
            return false;
        }

        base = url + 14;
    }

    if (!strlen(base) || *base == '/') {
        return false;
    }

    if (base[0] == '[') {
        return parseIPv6(base);
    }

    const char *port = strchr(base, ':');
    if (!port) {
        m_host = strdup(base);
        return false;
    }

    const size_t size = port++ - base + 1;
    m_host = new char[size]();
    memcpy(m_host, base, size - 1);

    m_port = (uint16_t) strtol(port, nullptr, 10);
    return true;
}


bool Url::setUserpass(const char *userpass)
{
    const char *p = strchr(userpass, ':');
    if (!p) {
        return false;
    }

    free(m_user);
    free(m_password);

    m_user = static_cast<char*>(calloc(p - userpass + 1, 1));
    strncpy(m_user, userpass, p - userpass);
    m_password = strdup(p + 1);

    return true;
}


const char *Url::url() const
{
    if (!m_url) {
        const size_t size = strlen(m_host) + 8;
        m_url = new char[size];

        snprintf(m_url, size - 1, "%s:%d", m_host, m_port);
    }

    return m_url;
}


void Url::adjust(int algo)
{
    if (!isValid()) {
        return;
    }

    m_algo = algo;

    if (strstr(m_host, ".nicehash.com")) {
        m_keepAlive = false;
        m_nicehash  = true;

        if (strstr(m_host, "cryptonightv7.")) {
            m_variant = xmrig::VARIANT_V1;
        }
    }

    if (strstr(m_host, ".minergate.com")) {
        m_keepAlive = false;
    }
}


void Url::setPassword(const char *password)
{
    if (!password) {
        return;
    }

    free(m_password);
    m_password = strdup(password);
}


void Url::setUser(const char *user)
{
    if (!user) {
        return;
    }

    free(m_user);
    m_user = strdup(user);
}


void Url::setVariant(int variant)
{
   switch (variant) {
   case xmrig::VARIANT_AUTO:
   case xmrig::VARIANT_NONE:
   case xmrig::VARIANT_V1:
       m_variant = variant;
       break;

   default:
       break;
   }
}


bool Url::operator==(const Url &other) const
{
    if (m_port != other.m_port || m_keepAlive != other.m_keepAlive || m_nicehash != other.m_nicehash) {
        return false;
    }

    if (strcmp(host(), other.host()) != 0 || strcmp(user(), other.user()) != 0 || strcmp(password(), other.password()) != 0) {
        return false;
    }

    return true;
}


Url &Url::operator=(const Url *other)
{
    m_keepAlive = other->m_keepAlive;
    m_algo      = other->m_algo;
    m_variant   = other->m_variant;
    m_nicehash  = other->m_nicehash;
    m_port      = other->m_port;

    free(m_host);
    m_host = strdup(other->m_host);

    setPassword(other->m_password);
    setUser(other->m_user);

    if (m_url) {
        delete [] m_url;
        m_url = nullptr;
    }

    return *this;
}


bool Url::parseIPv6(const char *addr)
{
    const char *end = strchr(addr, ']');
    if (!end) {
        return false;
    }

    const char *port = strchr(end, ':');
    if (!port) {
        return false;
    }

    const size_t size = end - addr;
    m_host = new char[size]();
    memcpy(m_host, addr + 1, size - 1);

    m_port = (uint16_t) strtol(port + 1, nullptr, 10);

    return true;
}
