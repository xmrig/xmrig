/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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

#ifndef __URL_H__
#define __URL_H__


#include <stdint.h>


class Url
{
public:
    constexpr static const char *kDefaultPassword = "x";
    constexpr static const char *kDefaultUser     = "x";
    constexpr static uint16_t kDefaultPort        = 3333;

    Url();
    Url(const char *url);
    Url(const char *host, uint16_t port, const char *user = nullptr, const char *password = nullptr, bool keepAlive = false, bool nicehash = false  );
    ~Url();

    inline bool isKeepAlive() const          { return m_keepAlive; }
    inline bool isNicehash() const           { return m_nicehash; }
    inline bool isValid() const              { return m_host && m_port > 0; }
    inline const char *host() const          { return m_host; }
    inline const char *password() const      { return m_password ? m_password : kDefaultPassword; }
    inline const char *user() const          { return m_user ? m_user : kDefaultUser; }
    inline uint16_t port() const             { return m_port; }
    inline void setKeepAlive(bool keepAlive) { m_keepAlive = keepAlive; }
    inline void setNicehash(bool nicehash)   { m_nicehash = nicehash; }

    bool parse(const char *url);
    bool setUserpass(const char *userpass);
    void applyExceptions();
    void setPassword(const char *password);
    void setUser(const char *user);

    Url &operator=(const Url *other);

private:
    bool m_keepAlive;
    bool m_nicehash;
    char *m_host;
    char *m_password;
    char *m_user;
    uint16_t m_port;
};

#endif /* __URL_H__ */
