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


#include "base/net/stratum/ProxyUrl.h"
#include "3rdparty/rapidjson/document.h"


namespace xmrig {

static const String kLocalhost = "127.0.0.1";

} // namespace xmrig


xmrig::ProxyUrl::ProxyUrl(const rapidjson::Value &value)
{
    m_port = 0;

    if (value.IsString()) {
        parse(value.GetString());
    }
    else if (value.IsUint()) {
        m_port = value.GetUint();
    }
}


const xmrig::String &xmrig::ProxyUrl::host() const
{
    return m_host.isNull() && isValid() ? kLocalhost : m_host;
}


rapidjson::Value xmrig::ProxyUrl::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    if (!isValid()) {
        return Value(kNullType);
    }

    if (!m_host.isNull()) {
        return m_url.toJSON(doc);
    }

    return Value(m_port);
}
