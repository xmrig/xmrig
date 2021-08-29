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

#ifndef XMRIG_DNSCONFIG_H
#define XMRIG_DNSCONFIG_H


#include "3rdparty/rapidjson/fwd.h"


namespace xmrig {


class DnsConfig
{
public:
    static const char *kField;
    static const char *kIPv6;
    static const char *kTTL;

    DnsConfig() = default;
    DnsConfig(const rapidjson::Value &value);

    inline bool isIPv6() const  { return m_ipv6; }
    inline uint32_t ttl() const { return m_ttl * 1000U; }

    rapidjson::Value toJSON(rapidjson::Document &doc) const;


private:
    bool m_ipv6     = false;
    uint32_t m_ttl  = 30U;
};


} /* namespace xmrig */


#endif /* XMRIG_DNSCONFIG_H */
