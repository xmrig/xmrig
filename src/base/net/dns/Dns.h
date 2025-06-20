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

#ifndef XMRIG_DNS_H
#define XMRIG_DNS_H


#include "base/net/dns/DnsConfig.h"
#include "base/tools/String.h"


#include <map>
#include <memory>


namespace xmrig {


class DnsConfig;
class DnsRequest;
class IDnsBackend;
class IDnsListener;


class Dns
{
public:
    inline static const DnsConfig &config()             { return m_config; }
    inline static void set(const DnsConfig &config)     { m_config = config; }

    static std::shared_ptr<DnsRequest> resolve(const String &host, IDnsListener *listener);

private:
    static DnsConfig m_config;
    static std::map<String, std::shared_ptr<IDnsBackend> > m_backends;
};


} /* namespace xmrig */


#endif /* XMRIG_DNS_H */
