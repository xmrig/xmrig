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


#include "base/net/dns/Dns.h"
#include "base/kernel/interfaces/IDnsListener.h"


namespace xmrig {
    Storage<Dns> Dns::m_storage;
}


xmrig::Dns::Dns(IDnsListener *listener) :
    m_listener(listener)
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

        m_records.clear();
    }

    m_status = uv_getaddrinfo(uv_default_loop(), m_resolver, Dns::onResolved, m_host.data(), nullptr, &m_hints);

    return m_status == 0;
}


void xmrig::Dns::onResolved(int status, addrinfo *res)
{
    m_status = status;

    if (m_status < 0) {
        return m_listener->onResolved(m_records, status);
    }

    m_records.parse(res);

    if (m_records.isEmpty()) {
        m_status = UV_EAI_NONAME;
    }

    m_listener->onResolved(m_records, m_status);
}


void xmrig::Dns::onResolved(uv_getaddrinfo_t *req, int status, addrinfo *res)
{
    Dns *dns = m_storage.get(req->data);
    if (dns) {
        dns->onResolved(status, res);
    }

    uv_freeaddrinfo(res);
}
