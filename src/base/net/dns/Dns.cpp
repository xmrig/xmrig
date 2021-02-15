/* xmlcore
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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


#include "base/net/dns/Dns.h"
#include "base/kernel/interfaces/IDnsListener.h"


namespace xmlcore {
    Storage<Dns> Dns::m_storage;
    static const DnsRecord defaultRecord;
}


xmlcore::Dns::Dns(IDnsListener *listener) :
    m_listener(listener)
{
    m_key = m_storage.add(this);

    m_resolver = new uv_getaddrinfo_t;
    m_resolver->data = m_storage.ptr(m_key);

    m_hints.ai_family   = AF_UNSPEC;
    m_hints.ai_socktype = SOCK_STREAM;
    m_hints.ai_protocol = IPPROTO_TCP;
}


xmlcore::Dns::~Dns()
{
    m_storage.release(m_key);

    delete m_resolver;
}


bool xmlcore::Dns::resolve(const String &host)
{
    if (m_host != host) {
        m_host = host;

        clear();
    }

    m_status = uv_getaddrinfo(uv_default_loop(), m_resolver, Dns::onResolved, m_host.data(), nullptr, &m_hints);

    return m_status == 0;
}


const char *xmlcore::Dns::error() const
{
    return uv_strerror(m_status);
}


const xmlcore::DnsRecord &xmlcore::Dns::get(DnsRecord::Type prefered) const
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


size_t xmlcore::Dns::count(DnsRecord::Type type) const
{
    if (type == DnsRecord::A) {
        return m_ipv4.size();
    }

    if (type == DnsRecord::AAAA) {
        return m_ipv6.size();
    }

    return m_ipv4.size() + m_ipv6.size();
}


void xmlcore::Dns::clear()
{
    m_ipv4.clear();
    m_ipv6.clear();
}


void xmlcore::Dns::onResolved(int status, addrinfo *res)
{
    m_status = status;

    if (m_status < 0) {
        return m_listener->onResolved(*this, status);
    }

    clear();

    addrinfo *ptr = res;
    while (ptr != nullptr) {
        if (ptr->ai_family == AF_INET) {
            m_ipv4.emplace_back(ptr);
        }

        if (ptr->ai_family == AF_INET6) {
            m_ipv6.emplace_back(ptr);
        }

        ptr = ptr->ai_next;
    }

    if (isEmpty()) {
        m_status = UV_EAI_NONAME;
    }

    m_listener->onResolved(*this, m_status);
}


void xmlcore::Dns::onResolved(uv_getaddrinfo_t *req, int status, addrinfo *res)
{
    Dns *dns = m_storage.get(req->data);
    if (dns) {
        dns->onResolved(status, res);
    }

    uv_freeaddrinfo(res);
}
