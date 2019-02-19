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

#ifndef XMRIG_POOL_H
#define XMRIG_POOL_H


#include <vector>


#include "base/tools/String.h"
#include "common/crypto/Algorithm.h"
#include "rapidjson/fwd.h"


namespace xmrig {


class Pool
{
public:
    constexpr static const char *kDefaultPassword = "x";
    constexpr static const char *kDefaultUser     = "x";
    constexpr static uint16_t kDefaultPort        = 3333;
    constexpr static int kKeepAliveTimeout        = 60;

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

    inline bool isNicehash() const                      { return m_nicehash; }
    inline bool isTLS() const                           { return m_tls; }
    inline bool isValid() const                         { return !m_host.isNull() && m_port > 0; }
    inline const char *fingerprint() const              { return m_fingerprint.data(); }
    inline const char *host() const                     { return m_host.data(); }
    inline const char *password() const                 { return !m_password.isNull() ? m_password.data() : kDefaultPassword; }
    inline const char *rigId() const                    { return m_rigId.data(); }
    inline const char *url() const                      { return m_url.data(); }
    inline const char *user() const                     { return !m_user.isNull() ? m_user.data() : kDefaultUser; }
    inline const Algorithm &algorithm() const           { return m_algorithm; }
    inline const Algorithms &algorithms() const         { return m_algorithms; }
    inline int keepAlive() const                        { return m_keepAlive; }
    inline uint16_t port() const                        { return m_port; }
    inline void setFingerprint(const char *fingerprint) { m_fingerprint = fingerprint; }
    inline void setKeepAlive(int keepAlive)             { m_keepAlive = keepAlive >= 0 ? keepAlive : 0; }
    inline void setKeepAlive(bool enable)               { setKeepAlive(enable ? kKeepAliveTimeout : 0); }
    inline void setNicehash(bool nicehash)              { m_nicehash = nicehash; }
    inline void setPassword(const char *password)       { m_password = password; }
    inline void setRigId(const char *rigId)             { m_rigId = rigId; }
    inline void setTLS(bool tls)                        { m_tls = tls; }
    inline void setUser(const char *user)               { m_user = user; }
    inline Algorithm &algorithm()                       { return m_algorithm; }

    inline bool operator!=(const Pool &other) const  { return !isEqual(other); }
    inline bool operator==(const Pool &other) const  { return isEqual(other); }

    bool isCompatible(const Algorithm &algorithm) const;
    bool isEnabled() const;
    bool isEqual(const Pool &other) const;
    bool parse(const char *url);
    bool setUserpass(const char *userpass);
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    void adjust(const Algorithm &algorithm);
    void setAlgo(const Algorithm &algorithm);

#   ifdef APP_DEBUG
    void print() const;
#   endif

private:
    bool parseIPv6(const char *addr);
    void addVariant(Variant variant);
    void adjustVariant(const Variant variantHint);
    void rebuild();

    Algorithm m_algorithm;
    Algorithms m_algorithms;
    bool m_enabled;
    bool m_nicehash;
    bool m_tls;
    int m_keepAlive;
    String m_fingerprint;
    String m_host;
    String m_password;
    String m_rigId;
    String m_url;
    String m_user;
    uint16_t m_port;
};


} /* namespace xmrig */


#endif /* XMRIG_POOL_H */
