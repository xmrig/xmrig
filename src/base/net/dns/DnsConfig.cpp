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

#include "base/net/dns/DnsConfig.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"


#include <algorithm>


namespace xmrig {


const char *DnsConfig::kField   = "dns";
const char *DnsConfig::kIPv6    = "ipv6";
const char *DnsConfig::kTTL     = "ttl";


} // namespace xmrig


xmrig::DnsConfig::DnsConfig(const rapidjson::Value &value)
{
    m_ipv6  = Json::getBool(value, kIPv6, m_ipv6);
    m_ttl   = std::max(Json::getUint(value, kTTL, m_ttl), 1U);
}


rapidjson::Value xmrig::DnsConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();
    Value obj(kObjectType);

    obj.AddMember(StringRef(kIPv6), m_ipv6, allocator);
    obj.AddMember(StringRef(kTTL),  m_ttl, allocator);

    return obj;
}
