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


#include "base/net/dns/Dns.h"
#include "base/net/dns/DnsUvBackend.h"


namespace xmrig {


DnsConfig Dns::m_config;
std::map<String, std::shared_ptr<IDnsBackend> > Dns::m_backends;


} // namespace xmrig


std::shared_ptr<xmrig::DnsRequest> xmrig::Dns::resolve(const String &host, IDnsListener *listener, uint64_t ttl)
{
    if (m_backends.find(host) == m_backends.end()) {
        m_backends.insert({ host, std::make_shared<DnsUvBackend>() });
    }

    return m_backends.at(host)->resolve(host, listener, ttl == 0 ? m_config.ttl() : ttl);
}
