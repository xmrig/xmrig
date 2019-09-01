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

#ifndef XMRIG_CC_CLIENT_CONFIG_H
#define XMRIG_CC_CLIENT_CONFIG_H

#include "base/tools/String.h"
#include "rapidjson/fwd.h"

namespace xmrig {

class CCClientConfig
{
public:
    bool read(const rapidjson::Value &value);
    rapidjson::Value toJSON(rapidjson::Document &doc) const;

    inline bool enabled() const                  { return m_enabled; }
    inline bool useTLS() const                   { return m_useTls; }
    inline bool useRemoteLogging() const         { return m_useRemoteLogging; }
    inline bool uploadConfigOnStartup() const    { return m_uploadConfigOnStartup; }

    inline const char *url() const               { return m_url.data(); }
    inline const char *host() const              { return m_host.data(); }
    inline const char *token() const             { return m_token.data(); }
    inline const char *workerId() const          { return m_workerId.data(); }
    inline const char *rebootCmd() const         { return m_rebootCmd.data(); }

    inline int updateInterval() const            { return m_updateInterval; }
    inline int port() const                      { return m_port; }

    inline bool operator!=(const CCClientConfig &other) const    { return !isEqual(other); }
    inline bool operator==(const CCClientConfig &other) const    { return isEqual(other); }

    bool isEqual(const CCClientConfig &other) const;

private:
    bool parseCCUrl(const char* url);

    bool m_enabled = true;
    bool m_useTls = false;
    bool m_useRemoteLogging = true;
    bool m_uploadConfigOnStartup = true;

    int m_updateInterval = 10;
    int m_port = 3344;

    String m_url;
    String m_host;
    String m_token;
    String m_workerId;
    String m_rebootCmd;
};


} /* namespace xmrig */


#endif /* XMRIG_CC_CLIENT_CONFIG_H */
