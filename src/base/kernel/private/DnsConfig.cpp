/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/kernel/private/DnsConfig.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/tools/Arguments.h"


#if defined(APP_DEBUG) && defined(XMRIG_FEATURE_EVENTS)
#   include "base/io/log/Log.h"
#   include "base/kernel/Config.h"
#endif


#include <algorithm>


namespace xmrig {


const char *DnsConfig::kField   = "dns";
const char *DnsConfig::kIPv6    = "ipv6";
const char *DnsConfig::kTTL     = "ttl";


#ifndef XMRIG_FEATURE_EVENTS
static DnsConfig config;
#endif


} // namespace xmrig


xmrig::DnsConfig::DnsConfig(const Arguments &arguments) :
    m_ipv6(arguments.contains("--dns-ipv6")),
    m_ttl(std::max(arguments.value("--dns-ttl").toUint(kDefaultTTL), 1U))
{
}


xmrig::DnsConfig::DnsConfig(const rapidjson::Value &value, const DnsConfig &current)
{
    m_ipv6  = Json::getBool(value, kIPv6, current.m_ipv6);
    m_ttl   = std::max(Json::getUint(value, kTTL, current.m_ttl), 1U);
}


bool xmrig::DnsConfig::isEqual(const DnsConfig &other) const
{
    return other.m_ipv6 == m_ipv6 && other.m_ttl == m_ttl;
}


rapidjson::Value xmrig::DnsConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();
    Value obj(kObjectType);

    obj.AddMember(StringRef(kIPv6), m_ipv6 ? Value(m_ipv6) : Value(kNullType), allocator);
    obj.AddMember(StringRef(kTTL),  m_ttl == kDefaultTTL ? Value(kNullType) : Value(m_ttl), allocator);

    return obj;
}


void xmrig::DnsConfig::print() const
{
#if defined(APP_DEBUG) && defined(XMRIG_FEATURE_EVENTS)
    LOG_DEBUG("%s " MAGENTA_BOLD("DNS")
              MAGENTA("<ipv6=") CYAN("%d")
              MAGENTA(", ttl=") CYAN("%u")
              MAGENTA(">"),
              Config::tag(), m_ipv6, m_ttl);
#   endif
}


#ifndef XMRIG_FEATURE_EVENTS
const xmrig::DnsConfig &xmrig::DnsConfig::current()
{
    return config;
}


void xmrig::DnsConfig::set(const rapidjson::Value &value)
{
    config = { value, config };
}
#endif
