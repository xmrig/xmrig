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


#include <uv.h>


#include "base/net/dns/DnsUvBackend.h"
#include "base/kernel/interfaces/IDnsListener.h"
#include "base/net/dns/DnsRequest.h"
#include "base/tools/Chrono.h"


namespace xmrig {


Storage<DnsUvBackend>& DnsUvBackend::getStorage()
{
    static Storage<DnsUvBackend>* storage = new Storage<DnsUvBackend>();
    return *storage;
}

static addrinfo hints{};


} // namespace xmrig


xmrig::DnsUvBackend::DnsUvBackend()
{
    if (!hints.ai_protocol) {
        hints.ai_family     = AF_UNSPEC;
        hints.ai_socktype   = SOCK_STREAM;
        hints.ai_protocol   = IPPROTO_TCP;
    }

    m_key = getStorage().add(this);
}


xmrig::DnsUvBackend::~DnsUvBackend()
{
    getStorage().release(m_key);
}


std::shared_ptr<xmrig::DnsRequest> xmrig::DnsUvBackend::resolve(const String &host, IDnsListener *listener, uint64_t ttl)
{
    auto req = std::make_shared<DnsRequest>(listener);

    if (Chrono::currentMSecsSinceEpoch() - m_ts <= ttl && !m_records.isEmpty()) {
        req->listener->onResolved(m_records, 0, nullptr);
    } else {
        m_queue.emplace(req);
    }

    if (m_queue.size() == 1 && !resolve(host)) {
        done();
    }

    return req;
}


bool xmrig::DnsUvBackend::resolve(const String &host)
{
    m_req = std::make_shared<uv_getaddrinfo_t>();
    m_req->data = getStorage().ptr(m_key);

    m_status = uv_getaddrinfo(uv_default_loop(), m_req.get(), DnsUvBackend::onResolved, host.data(), nullptr, &hints);

    return m_status == 0;
}


void xmrig::DnsUvBackend::done()
{
    const char *error = m_status < 0 ? uv_strerror(m_status) : nullptr;

    while (!m_queue.empty()) {
        auto req = std::move(m_queue.front()).lock();
        if (req) {
            req->listener->onResolved(m_records, m_status, error);
        }

        m_queue.pop();
    }

    m_req.reset();
}


void xmrig::DnsUvBackend::onResolved(int status, addrinfo *res)
{
    m_ts = Chrono::currentMSecsSinceEpoch();

    if ((m_status = status) < 0) {
        return done();
    }

    m_records.parse(res);

    if (m_records.isEmpty()) {
        m_status = UV_EAI_NONAME;
    }

    done();
}


void xmrig::DnsUvBackend::onResolved(uv_getaddrinfo_t *req, int status, addrinfo *res)
{
    auto backend = getStorage().get(req->data);
    if (backend) {
        backend->onResolved(status, res);
    }

    uv_freeaddrinfo(res);
}
