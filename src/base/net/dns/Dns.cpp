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


#include "base/kernel/interfaces/IDnsListener.h"
#include "base/net/dns/Dns.h"
#include "base/tools/Handle.h"


namespace xmrig {
    Storage<Dns> Dns::m_storage;
    static const DnsRecord defaultRecord;
}


xmrig::Dns::Dns(IDnsListener *listener) :
    m_hints(),
    m_listener(listener),
    m_status(0),
    m_resolver(nullptr)
{
    m_key = m_storage.add(this);

    m_resolver = new uv_getaddrinfo_t;
    m_resolver->data = m_storage.ptr(m_key);

    m_hints.ai_family   = AF_UNSPEC;
    m_hints.ai_socktype = SOCK_STREAM;
    m_hints.ai_protocol = IPPROTO_TCP;
}


xmrig::Dns::~Dns()
{
    m_storage.release(m_key);

    delete m_resolver;
}


bool xmrig::Dns::resolve(const String &host)
{
    if (m_host != host) {
        m_host = host;

        clear();
    }

    m_status = uv_getaddrinfo(uv_default_loop(), m_resolver, Dns::onResolved, m_host.data(), nullptr, &m_hints);

    return m_status == 0;
}


const char *xmrig::Dns::error() const
{
    return uv_strerror(m_status);
}


const xmrig::DnsRecord &xmrig::Dns::get(DnsRecord::Type prefered) const
{
    if (count() == 0) {
        return defaultRecord;
    }

    const size_t ipv4 = m_ipv4.size();
    const size_t ipv6 = m_ipv6.size();

    if (ipv6 && (prefered == DnsRecord::AAAA || !ipv4)) {
        return m_ipv6[ipv6 == 1 ? 0 : static_cast<size_t>(rand()) % ipv6];
    }

    if (ipv4) {
        return m_ipv4[ipv4 == 1 ? 0 : static_cast<size_t>(rand()) % ipv4];
    }

    return defaultRecord;
}


size_t xmrig::Dns::count(DnsRecord::Type type) const
{
    if (type == DnsRecord::A) {
        return m_ipv4.size();
    }

    if (type == DnsRecord::AAAA) {
        return m_ipv6.size();
    }

    return m_ipv4.size() + m_ipv6.size();
}


void xmrig::Dns::clear()
{
    m_ipv4.clear();
    m_ipv6.clear();
}


void xmrig::Dns::onResolved(int status, addrinfo *res)
{
    m_status = status;

    if (m_status < 0) {
        return m_listener->onResolved(*this, status);
    }

    clear();

    addrinfo *ptr = res;
    while (ptr != nullptr) {
        if (ptr->ai_family == AF_INET) {
            m_ipv4.push_back(ptr);
        }

        if (ptr->ai_family == AF_INET6) {
            m_ipv6.push_back(ptr);
        }

        ptr = ptr->ai_next;
    }

    if (isEmpty()) {
        m_status = UV_EAI_NONAME;
    }

    m_listener->onResolved(*this, m_status);
}


void xmrig::Dns::onResolved(uv_getaddrinfo_t *req, int status, addrinfo *res)
{
    Dns *dns = m_storage.get(req->data);
    if (dns) {
        dns->onResolved(status, res);
    }

    uv_freeaddrinfo(res);
}
