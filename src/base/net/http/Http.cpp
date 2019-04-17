/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/net/http/Http.h"


namespace xmrig {

static const char *kEnabled    = "enabled";
static const char *kHost       = "host";
static const char *kLocalhost  = "127.0.0.1";
static const char *kPort       = "port";
static const char *kRestricted = "restricted";
static const char *kToken      = "access-token";

}


xmrig::Http::Http() :
    m_enabled(false),
    m_restricted(true),
    m_host(kLocalhost),
    m_port(0)
{
}


bool xmrig::Http::isEqual(const Http &other) const
{
    return other.m_enabled    == m_enabled &&
           other.m_restricted == m_restricted &&
           other.m_host       == m_host &&
           other.m_token      == m_token &&
           other.m_port       == m_port;
}


rapidjson::Value xmrig::Http::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kEnabled),    m_enabled, allocator);
    obj.AddMember(StringRef(kHost),       m_host.toJSON(), allocator);
    obj.AddMember(StringRef(kPort),       m_port, allocator);
    obj.AddMember(StringRef(kToken),      m_token.toJSON(), allocator);
    obj.AddMember(StringRef(kRestricted), m_restricted, allocator);

    return obj;
}


void xmrig::Http::load(const rapidjson::Value &http)
{
    if (!http.IsObject()) {
        return;
    }

    m_enabled    = Json::getBool(http, kEnabled);
    m_restricted = Json::getBool(http, kRestricted, true);
    m_host       = Json::getString(http, kHost, kLocalhost);
    m_token      = Json::getString(http, kToken);

    setPort(Json::getInt(http, kPort));
}


void xmrig::Http::setPort(int port)
{
    if (port >= 0 && port <= 65536) {
        m_port = static_cast<uint16_t>(port);
    }
}
