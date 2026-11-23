/* XMRig
 * Copyright (c) 2018-2026 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2026 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_SOCKS5HANDSHAKE_H
#define XMRIG_SOCKS5HANDSHAKE_H

#include "base/tools/Object.h"
#include "base/tools/String.h"
#include <cstdint>
#include <string>

namespace xmrig {

class Socks5Handshake
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Socks5Handshake)

    Socks5Handshake(const String &host, uint16_t port);

    inline bool isReady() const         { return m_state == READY; }
    inline const char *error() const    { return m_error; }

    static std::string greeting();

    bool read(const char *&data, size_t &size, std::string &out);

private:
    enum State {
        METHOD_REPLY,
        CONNECT_REPLY,
        READY,
        FAILED
    };

    bool connectRequest(std::string &out);
    void fail(const char *error);
    void parse(std::string &out);

    const char *m_error = nullptr;
    const String m_host;
    const uint16_t m_port;
    size_t m_size       = 2;    // size of the current reply as far as it is known yet
    State m_state       = METHOD_REPLY;
    std::string m_reply;
};

} // namespace xmrig

#endif // XMRIG_SOCKS5HANDSHAKE_H
