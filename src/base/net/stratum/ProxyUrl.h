/* XMRig
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

#ifndef XMRIG_PROXYURL_H
#define XMRIG_PROXYURL_H


#include "base/net/stratum/Url.h"


namespace xmrig {


class ProxyUrl : public Url
{
public:
    inline ProxyUrl() { m_port = 0; }

    ProxyUrl(const rapidjson::Value &value);

    inline bool isValid() const { return m_port > 0 && (m_scheme == UNSPECIFIED || m_scheme == SOCKS5); }

    const String &host() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
};


} /* namespace xmrig */


#endif /* XMRIG_PROXYURL_H */
