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


#include "net/Url.h"


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
    m_host(nullptr),
    m_port(3333)
{
    const char *p = strstr(url, "://");
    const char *base = url;

    if (p) {
        if (strncasecmp(url, "stratum+tcp://", 14)) {
            return;
        }

        base = url + 14;
    }

    if (!strlen(base) || *base == '/') {
        return;
    }

    char *port = strchr(base, ':');
    if (!port) {
        m_host = strdup(base);
        return;
    }

    const size_t size = port++ - base + 1;
    m_host = static_cast<char*>(malloc(size));
    memcpy(m_host, base, size - 1);
    m_host[size - 1] = '\0';

    m_port = strtol(port, nullptr, 10);
}


Url::Url(const char *host, uint16_t port) :
    m_port(port)
{
    m_host = strdup(host);
}


Url::~Url()
{
    free(m_host);
}


bool Url::isNicehash() const
{
    return isValid() && strstr(m_host, ".nicehash.com");
}
