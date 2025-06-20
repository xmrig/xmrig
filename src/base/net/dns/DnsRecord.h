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

#pragma once

struct addrinfo;
struct sockaddr;

#include "base/tools/String.h"


namespace xmrig {


class DnsRecord
{
public:
    DnsRecord() {}
    DnsRecord(const addrinfo *addr);

    const sockaddr *addr(uint16_t port = 0) const;
    String ip() const;

private:
    mutable uint8_t m_data[28]{};
};


} // namespace xmrig
