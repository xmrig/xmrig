/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      Howard Chu  <https://github.com/hyc>
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

#ifndef XMRIG_POOL_H
#define XMRIG_POOL_H


#include <bitset>
#include <vector>


#include "base/tools/String.h"
#include "crypto/common/Algorithm.h"
#include "rapidjson/fwd.h"


namespace xmrig {


class Pool
{
public:
    enum Flags {
        FLAG_ENABLED,
        FLAG_NICEHASH,
        FLAG_TLS,
        FLAG_DAEMON,
        FLAG_MAX
    };

    static const String kDefaultPassword;
    static const String kDefaultUser;

    constexpr static int kKeepAliveTimeout         = 60;
    constexpr static uint16_t kDefaultPort         = 3333;
    constexpr static uint64_t kDefaultPollInterval = 1000;

    Pool();
    Pool(const char *url);
    Pool(const rapidjson::Value &object);
    Pool(const char *host,
         uint16_t port,
         const char *user       = nullptr,
         const char *password   = nullptr,
         int keepAlive          = 0,
         bool nicehash          = false,
         bool tls               = false
       );

    inline bool isDaemon() const                        { return m_flags.test(FLAG_DAEMON); }
    inline bool isNicehash() const                      { return m_flags.test(FLAG_NICEHASH); }
    inline bool isTLS() const                           { return m_flags.test(FLAG_TLS); }
    inline bool isValid() const                         { return !m_host.isNull() && m_port > 0; }
    inline const Algorithm &algorithm() const           { return m_algorithm; }
    inline const String &fingerprint() const            { return m_fingerprint; }
    inline const String &host() const                   { return m_host; }
    inline const String &password() const               { return !m_password.isNull() ? m_password : kDefaultPassword; }
    inline const String &rigId() const                  { return m_rigId; }
    inline const String &url() const                    { return m_url; }
    inline const String &user() const                   { return !m_user.isNull() ? m_user : kDefaultUser; }
    inline int keepAlive() const                        { return m_keepAlive; }
    inline uint16_t port() const                        { return m_port; }
    inline uint64_t pollInterval() const                { return m_pollInterval; }
    inline void setAlgo(const Algorithm &algorithm)     { m_algorithm = algorithm; }
    inline void setPassword(const String &password)     { m_password = password; }
    inline void setRigId(const String &rigId)           { m_rigId = rigId; }
    inline void setUser(const String &user)             { m_user = user; }

    inline bool operator!=(const Pool &other) const     { return !isEqual(other); }
    inline bool operator==(const Pool &other) const     { return isEqual(other); }

    bool isEnabled() const;
    bool isEqual(const Pool &other) const;
    bool parse(const char *url);
    rapidjson::Value toJSON(rapidjson::Document &doc) const;

#   ifdef APP_DEBUG
    void print() const;
#   endif

private:
    inline void setKeepAlive(bool enable)               { setKeepAlive(enable ? kKeepAliveTimeout : 0); }
    inline void setKeepAlive(int keepAlive)             { m_keepAlive = keepAlive >= 0 ? keepAlive : 0; }

    bool parseIPv6(const char *addr);

    Algorithm m_algorithm;
    int m_keepAlive;
    std::bitset<FLAG_MAX> m_flags;
    String m_fingerprint;
    String m_host;
    String m_password;
    String m_rigId;
    String m_url;
    String m_user;
    uint16_t m_port;
    uint64_t m_pollInterval;
};


} /* namespace xmrig */


#endif /* XMRIG_POOL_H */
