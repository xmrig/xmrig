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

#include "base/net/dns/DnsRecord.h"


xmrig::DnsRecord::DnsRecord(const addrinfo *addr)
{
    static_assert(sizeof(m_data) >= sizeof(sockaddr_in6), "Not enough storage for IPv6 address.");

    memcpy(m_data, addr->ai_addr, addr->ai_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in));
}


const sockaddr *xmrig::DnsRecord::addr(uint16_t port) const
{
    reinterpret_cast<sockaddr_in*>(m_data)->sin_port = htons(port);

    return reinterpret_cast<const sockaddr *>(m_data);
}


xmrig::String xmrig::DnsRecord::ip() const
{
    char *buf = nullptr;

    if (reinterpret_cast<const sockaddr &>(m_data).sa_family == AF_INET6) {
        buf = new char[45]();
        uv_ip6_name(reinterpret_cast<const sockaddr_in6*>(m_data), buf, 45);
    }
    else {
        buf = new char[16]();
        uv_ip4_name(reinterpret_cast<const sockaddr_in*>(m_data), buf, 16);
    }

    return buf;
}
