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


#include "base/net/dns/DnsRecords.h"
#include "base/net/dns/Dns.h"


const xmrig::DnsRecord &xmrig::DnsRecords::get(DnsRecord::Type prefered) const
{
    static const DnsRecord defaultRecord;

    if (isEmpty()) {
        return defaultRecord;
    }

    const size_t ipv4 = m_ipv4.size();
    const size_t ipv6 = m_ipv6.size();

    if (ipv6 && (prefered == DnsRecord::AAAA || Dns::config().isIPv6() || !ipv4)) {
        return m_ipv6[ipv6 == 1 ? 0 : static_cast<size_t>(rand()) % ipv6]; // NOLINT(concurrency-mt-unsafe, cert-msc30-c, cert-msc50-cpp)
    }

    if (ipv4) {
        return m_ipv4[ipv4 == 1 ? 0 : static_cast<size_t>(rand()) % ipv4]; // NOLINT(concurrency-mt-unsafe, cert-msc30-c, cert-msc50-cpp)
    }

    return defaultRecord;
}


size_t xmrig::DnsRecords::count(DnsRecord::Type type) const
{
    if (type == DnsRecord::A) {
        return m_ipv4.size();
    }

    if (type == DnsRecord::AAAA) {
        return m_ipv6.size();
    }

    return m_ipv4.size() + m_ipv6.size();
}


void xmrig::DnsRecords::clear()
{
    m_ipv4.clear();
    m_ipv6.clear();
}


void xmrig::DnsRecords::parse(addrinfo *res)
{
    clear();

    addrinfo *ptr = res;
    size_t ipv4   = 0;
    size_t ipv6   = 0;

    while (ptr != nullptr) {
        if (ptr->ai_family == AF_INET) {
            ++ipv4;
        }
        else if (ptr->ai_family == AF_INET6) {
            ++ipv6;
        }

        ptr = ptr->ai_next;
    }

    if (ipv4 == 0 && ipv6 == 0) {
        return;
    }

    m_ipv4.reserve(ipv4);
    m_ipv6.reserve(ipv6);

    ptr = res;
    while (ptr != nullptr) {
        if (ptr->ai_family == AF_INET) {
            m_ipv4.emplace_back(ptr);
        }
        else if (ptr->ai_family == AF_INET6) {
            m_ipv6.emplace_back(ptr);
        }

        ptr = ptr->ai_next;
    }
}
