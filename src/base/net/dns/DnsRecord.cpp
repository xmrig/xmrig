/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


xmrig::DnsRecord::DnsRecord(const addrinfo *addr) :
    m_type(addr->ai_family == AF_INET6 ? AAAA : A)
{
    char *buf = nullptr;

    if (m_type == AAAA) {
        buf = new char[45]();
        uv_ip6_name(reinterpret_cast<sockaddr_in6*>(addr->ai_addr), buf, 45);
    }
    else {
        buf = new char[16]();
        uv_ip4_name(reinterpret_cast<sockaddr_in*>(addr->ai_addr), buf, 16);
    }

    m_ip = buf;
}


sockaddr *xmrig::DnsRecord::addr(uint16_t port) const
{
    if (m_type == A) {
        auto addr = new sockaddr_in();
        uv_ip4_addr(m_ip.data(), port, addr);

        return reinterpret_cast<sockaddr *>(addr);
    }

    if (m_type == AAAA) {
        auto addr = new sockaddr_in6();
        uv_ip6_addr(m_ip.data(), port, addr);

        return reinterpret_cast<sockaddr *>(addr);
    }

    return nullptr;
}
