/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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
#include <algorithm>


#include "net/Url.h"


#ifdef _MSC_VER
#   define strncasecmp(x,y,z) _strnicmp(x,y,z)
#endif


Url::Url() :
    m_keepAlive(false),
    m_nicehash(false),
    m_host(nullptr),
    m_password(nullptr),
    m_user(nullptr),
    m_port(kDefaultPort),
    m_proxy_host(nullptr),
    m_proxy_port(kDefaultProxyPort),
    m_keystream(nullptr)
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
    m_port(kDefaultPort),
    m_proxy_host(nullptr),
    m_proxy_port(kDefaultProxyPort),
    m_keystream(nullptr)
{
    parse(url);
}


Url::Url(const char *host, uint16_t port, const char *user, const char *password, bool keepAlive, bool nicehash) :
    m_keepAlive(keepAlive),
    m_nicehash(nicehash),
    m_password(password ? strdup(password) : nullptr),
    m_user(user ? strdup(user) : nullptr),
    m_port(port),
    m_proxy_host(nullptr),
    m_proxy_port(kDefaultProxyPort),
    m_keystream(nullptr)
{
    m_host = strdup(host);
}


Url::~Url()
{
    free(m_host);
    free(m_password);
    free(m_user);
    free(m_proxy_host);
    free(m_keystream);
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

    const char *port = strchr(base, ':');
    if (!port) {
        m_host = strdup(base);
        return false;
    }

    const size_t size = port++ - base + 1;
    m_host = static_cast<char*>(malloc(size));
    memcpy(m_host, base, size - 1);
    m_host[size - 1] = '\0';

    const char* proxy = strchr(port, '@');
    const char* keystream = strchr(port, '#');
    if(keystream)
    {
        ++keystream;
        if(!proxy)
        {
            m_keystream = strdup(keystream);
        }
        else
        {
            const size_t keystreamsize = proxy - keystream;
            m_keystream = static_cast<char*>(malloc (keystreamsize + 1));
            m_keystream[keystreamsize] = '\0';
            memcpy(m_keystream, keystream, keystreamsize);
        }
    }

    m_port = (uint16_t) strtol(port, nullptr, 10);
    if (!proxy) {
        m_port = (uint16_t) strtol(port, nullptr, 10);
        return true;
    }
    
    ++proxy;

    const char* proxyport = strchr(proxy, ':');
    if (!port) {
        m_proxy_host = strdup(proxy);
        return false;
    }

    const size_t proxysize = proxyport++ - proxy + 1;
    m_proxy_host = static_cast<char*>(malloc (proxysize));
    memcpy(m_proxy_host, proxy, proxysize - 1);
    m_proxy_host[proxysize - 1] = '\0';
    m_proxy_port = (uint16_t) strtol(proxyport, nullptr, 10);

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


void Url::applyExceptions()
{
    if (!isValid()) {
        return;
    }

    if (strstr(m_host, ".nicehash.com")) {
        m_keepAlive = false;
        m_nicehash  = true;
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

void Url::copyKeystream(char *keystreamDest, const size_t keystreamLen) const
{
    if(hasKeystream())
    {
        memset(keystreamDest, 1, keystreamLen);
        memcpy(keystreamDest, m_keystream, std::min(keystreamLen, strlen(m_keystream)));
    }
}

Url &Url::operator=(const Url *other)
{
    m_keepAlive = other->m_keepAlive;
    m_nicehash  = other->m_nicehash;
    m_port      = other->m_port;
    m_proxy_port = other->m_proxy_port;

    free(m_host);
    m_host = strdup(other->m_host);

    free (m_proxy_host);
    if(other->m_proxy_host)
    {
        m_proxy_host = strdup (other->m_proxy_host);
    }
    else
    {
        m_proxy_host = nullptr;
    }

    setPassword(other->m_password);
    setUser(other->m_user);

    free (m_keystream);
    if(other->m_keystream)
    {
        m_keystream = strdup (other->m_keystream);
    }
    else
    {
        m_keystream = nullptr;
    }
    return *this;
}
