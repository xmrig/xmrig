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

#ifndef XMRIG_DNS_H
#define XMRIG_DNS_H


#include <vector>
#include <uv.h>


#include "base/net/dns/DnsRecord.h"
#include "base/net/tools/Storage.h"
#include "base/tools/String.h"


namespace xmrig {


class IDnsListener;


class Dns
{
public:
    Dns(IDnsListener *listener);
    ~Dns();

    inline bool isEmpty() const       { return m_ipv4.empty() && m_ipv6.empty(); }
    inline const String &host() const { return m_host; }
    inline int status() const         { return m_status; }

    bool resolve(const String &host);
    const char *error() const;
    const DnsRecord &get(DnsRecord::Type prefered = DnsRecord::A) const;
    size_t count(DnsRecord::Type type = DnsRecord::Unknown) const;

private:
    void clear();
    void onResolved(int status, addrinfo *res);

    static void onResolved(uv_getaddrinfo_t *req, int status, addrinfo *res);

    addrinfo m_hints;
    IDnsListener *m_listener;
    int m_status;
    std::vector<DnsRecord> m_ipv4;
    std::vector<DnsRecord> m_ipv6;
    String m_host;
    uintptr_t m_key;
    uv_getaddrinfo_t *m_resolver;

    static Storage<Dns> m_storage;
};


} /* namespace xmrig */


#endif /* XMRIG_DNS_H */
