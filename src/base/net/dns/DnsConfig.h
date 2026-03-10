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

#include "3rdparty/rapidjson/fwd.h"


namespace xmrig {


class DnsConfig
{
public:
    static const char *kField;
    static const char *kIPv;
    static const char *kTTL;

    DnsConfig() = default;
    DnsConfig(const rapidjson::Value &value);

    inline uint32_t ipv() const { return m_ipv; }
    inline uint32_t ttl() const { return m_ttl * 1000U; }

    int ai_family() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;

private:
    uint32_t m_ttl = 30U;
    uint32_t m_ipv = 0U;
};


} // namespace xmrig
