/* XMRig
 * Copyright (c) 2018-2025 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


namespace {


static size_t dns_records_count(const addrinfo *res, int &ai_family)
{
    size_t ipv4 = 0;
    size_t ipv6 = 0;

    while (res != nullptr) {
        if (res->ai_family == AF_INET) {
            ++ipv4;
        }

        if (res->ai_family == AF_INET6) {
            ++ipv6;
        }

        res = res->ai_next;
    }

    if (ai_family == AF_INET6 && !ipv6) {
        ai_family = AF_INET;
    }

    switch (ai_family) {
    case AF_UNSPEC:
        return ipv4 + ipv6;

    case AF_INET:
        return ipv4;

    case AF_INET6:
        return ipv6;

    default:
        break;
    }

    return 0;
}


} // namespace


xmrig::DnsRecords::DnsRecords(const addrinfo *res, int ai_family)
{
    size_t size = dns_records_count(res, ai_family);
    if (!size) {
        return;
    }

    m_records.reserve(size);

    if (ai_family == AF_UNSPEC) {
        while (res != nullptr) {
            if (res->ai_family == AF_INET || res->ai_family == AF_INET6) {
                m_records.emplace_back(res);
            }

            res = res->ai_next;
        };
    } else {
        while (res != nullptr) {
            if (res->ai_family == ai_family) {
                m_records.emplace_back(res);
            }

            res = res->ai_next;
        };
    }

    size = m_records.size();
    if (size > 1) {
        m_index = static_cast<size_t>(rand()) % size; // NOLINT(concurrency-mt-unsafe, cert-msc30-c, cert-msc50-cpp)
    }
}


const xmrig::DnsRecord &xmrig::DnsRecords::get() const
{
    static const DnsRecord defaultRecord;

    const size_t size = m_records.size();
    if (size > 0) {
        return m_records[m_index++ % size];
    }

    return defaultRecord;
}
