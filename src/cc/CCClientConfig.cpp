/* XMRigCC
 * Copyright 2017-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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

#include "base/io/json/Json.h"
#include "cc/CCClientConfig.h"
#include "rapidjson/document.h"


namespace xmrig {

static const char *kEnabled                 = "enabled";
static const char *kUseTLS                  = "use-tls";
static const char *kUseRemoteLog            = "use-remote-logging";
static const char *kUploadConfigOnStartup   = "upload-config-on-start";

static const char *kUrl                     = "url";
static const char *kAccessToken             = "access-token";
static const char *kWorkerId                = "worker-id";
static const char *kRebootCmd               = "reboot-cmd";

static const char *kUpdateInterval          = "update-interval-s";

}


rapidjson::Value xmrig::CCClientConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kEnabled), m_enabled, allocator);
    obj.AddMember(StringRef(kUseTLS), m_useTls, allocator);
    obj.AddMember(StringRef(kUseRemoteLog), m_useRemoteLogging, allocator);
    obj.AddMember(StringRef(kUploadConfigOnStartup), m_uploadConfigOnStartup, allocator);

    obj.AddMember(StringRef(kUrl), m_url.toJSON(), allocator);
    obj.AddMember(StringRef(kAccessToken), m_token.toJSON(), allocator);
    obj.AddMember(StringRef(kWorkerId), m_workerId.toJSON(), allocator);
    obj.AddMember(StringRef(kRebootCmd), m_rebootCmd.toJSON(), allocator);

    obj.AddMember(StringRef(kUpdateInterval), m_updateInterval, allocator);

    return obj;
}


bool xmrig::CCClientConfig::read(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_enabled = Json::getBool(value, kEnabled, m_enabled);
        m_useTls = Json::getBool(value, kUseTLS, m_useTls);
        m_useRemoteLogging = Json::getBool(value, kUseRemoteLog, m_useRemoteLogging);
        m_uploadConfigOnStartup = Json::getBool(value, kUploadConfigOnStartup, m_uploadConfigOnStartup);

        m_url = Json::getString(value, kUrl, m_url);
        m_token = Json::getString(value, kAccessToken, m_token);
        m_workerId = Json::getString(value, kWorkerId, m_workerId);
        m_rebootCmd = Json::getString(value, kRebootCmd, m_rebootCmd);

        m_updateInterval   = Json::getInt(value, kUpdateInterval, m_updateInterval);

        parseCCUrl(m_url);

        return true;
    }

    return false;
}

bool xmrig::CCClientConfig::parseCCUrl(const char* url)
{
    const char *base = url;
    if (!base || !strlen(base) || *base == '/') {
        return false;
    }

    const char *port = strchr(base, ':');
    if (!port) {
        m_host = base;
        return true;
    }

    const size_t size = port++ - base + 1;
    auto *host        = new char[size]();
    memcpy(host, base, size - 1);

    m_host = host;
    m_port = static_cast<uint16_t>(strtol(port, nullptr, 10));

    return true;
}

bool xmrig::CCClientConfig::isEqual(const CCClientConfig &other) const
{
    return other.m_enabled                  == m_enabled &&
           other.m_useTls                   == m_useTls &&
           other.m_useRemoteLogging         == m_useRemoteLogging &&
           other.m_uploadConfigOnStartup    == m_uploadConfigOnStartup &&
           other.m_url                      == m_url &&
           other.m_host                     == m_host &&
           other.m_token                    == m_token &&
           other.m_port                     == m_port &&
           other.m_workerId                 == m_workerId &&
           other.m_rebootCmd                == m_rebootCmd &&
           other.m_updateInterval           == m_updateInterval;
}
