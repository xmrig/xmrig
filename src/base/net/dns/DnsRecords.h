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

#include "base/net/dns/DnsRecord.h"


namespace xmrig {


class DnsRecords
{
public:
    DnsRecords() = default;
    DnsRecords(const addrinfo *res, int ai_family);

    inline bool isEmpty() const                             { return m_records.empty(); }
    inline const std::vector<DnsRecord> &records() const    { return m_records; }
    inline size_t size() const                              { return m_records.size(); }

    const DnsRecord &get() const;

private:
    mutable size_t m_index = 0;
    std::vector<DnsRecord> m_records;
};


} // namespace xmrig
