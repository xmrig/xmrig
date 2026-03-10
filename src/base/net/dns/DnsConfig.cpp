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

#include "base/net/dns/DnsConfig.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"

#include <algorithm>
#include <uv.h>


namespace xmrig {


const char *DnsConfig::kField   = "dns";
const char *DnsConfig::kIPv     = "ip_version";
const char *DnsConfig::kTTL     = "ttl";


} // namespace xmrig


xmrig::DnsConfig::DnsConfig(const rapidjson::Value &value)
{
    const uint32_t ipv = Json::getUint(value, kIPv, m_ipv);
    if (ipv == 0 || ipv == 4 || ipv == 6) {
        m_ipv = ipv;
    }

    m_ttl = std::max(Json::getUint(value, kTTL, m_ttl), 1U);
}


int xmrig::DnsConfig::ai_family() const
{
    if (m_ipv == 4) {
        return AF_INET;
    }

    if (m_ipv == 6) {
        return AF_INET6;
    }

    return AF_UNSPEC;
}


rapidjson::Value xmrig::DnsConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();
    Value obj(kObjectType);

    obj.AddMember(StringRef(kIPv), m_ipv, allocator);
    obj.AddMember(StringRef(kTTL), m_ttl, allocator);

    return obj;
}
