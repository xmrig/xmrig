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


#include "base/net/tools/Socks5Handshake.h"
#include <algorithm>
#include <uv.h>

namespace xmrig {

static const char kGreeting[]       = { 0x05, 0x01, 0x00 }; // version 5, 1 method: no authentication
static const char kConnect[]        = { 0x05, 0x01, 0x00 }; // version 5, command CONNECT, reserved

static const char *replyError(uint8_t code)
{
    switch (code) {
    case 0x01:
        return "general SOCKS server failure";

    case 0x02:
        return "connection not allowed by ruleset";

    case 0x03:
        return "network unreachable";

    case 0x04:
        return "host unreachable";

    case 0x05:
        return "connection refused";

    case 0x06:
        return "TTL expired";

    case 0x07:
        return "command not supported";

    case 0x08:
        return "address type not supported";

    default:
        break;
    }

    return "unknown error";
}

} // namespace xmrig

xmrig::Socks5Handshake::Socks5Handshake(const String &host, uint16_t port)
    : m_host(host)
    , m_port(port)
{
}

std::string xmrig::Socks5Handshake::greeting()
{
    return { kGreeting, sizeof(kGreeting) };
}

bool xmrig::Socks5Handshake::read(const char *&data, size_t &size, std::string &out)
{
    while (size > 0 && m_state != READY && m_state != FAILED) {
        // Take only the bytes of the current reply: anything after the CONNECT reply comes from the destination.
        const size_t n = std::min(size, m_size - m_reply.size());

        m_reply.append(data, n);
        data += n;
        size -= n;

        if (m_reply.size() == m_size) {
            parse(out);
        }
    }

    return m_state != FAILED;
}

bool xmrig::Socks5Handshake::connectRequest(std::string &out)
{
    if (m_host.isEmpty() || m_host.size() > 255) {
        return false;
    }

    out.append(kConnect, sizeof(kConnect));

    sockaddr_storage addr{};

    if (uv_ip4_addr(m_host.data(), 0, reinterpret_cast<sockaddr_in *>(&addr)) == 0) {
        out.push_back(0x01);
        out.append(reinterpret_cast<const char *>(&reinterpret_cast<const sockaddr_in *>(&addr)->sin_addr), 4);
    }
    else if (uv_ip6_addr(m_host.data(), 0, reinterpret_cast<sockaddr_in6 *>(&addr)) == 0) {
        out.push_back(0x04);
        out.append(reinterpret_cast<const char *>(&reinterpret_cast<const sockaddr_in6 *>(&addr)->sin6_addr), 16);
    }
    else {
        out.push_back(0x03);
        out.push_back(static_cast<char>(m_host.size()));
        out.append(m_host.data(), m_host.size());
    }

    out.push_back(static_cast<char>(m_port >> 8));
    out.push_back(static_cast<char>(m_port & 0xFF));

    return true;
}

void xmrig::Socks5Handshake::fail(const char *error)
{
    m_error = error;
    m_state = FAILED;
}

void xmrig::Socks5Handshake::parse(std::string &out)
{
    const auto *reply = reinterpret_cast<const uint8_t *>(m_reply.data());

    if (reply[0] != 0x05) {
        return fail("invalid reply");
    }

    if (m_state == METHOD_REPLY) {
        if (reply[1] != 0x00) {
            return fail("proxy requires authentication");
        }

        if (!connectRequest(out)) {
            return fail("invalid host name");
        }

        m_state = CONNECT_REPLY;
        m_size  = 5; // enough to know the size of the whole reply
        m_reply.clear();

        return;
    }

    // CONNECT reply: version, status, reserved, address type, bound address, bound port
    if (reply[1] != 0x00) {
        return fail(replyError(reply[1]));
    }

    switch (reply[3]) {
    case 0x01:
        m_size = 4 + 4 + 2;
        break;

    case 0x03:
        m_size = 4 + 1 + reply[4] + 2;
        break;

    case 0x04:
        m_size = 4 + 16 + 2;
        break;

    default:
        return fail("invalid reply");
    }

    if (m_reply.size() == m_size) {
        m_state = READY;
        m_reply.clear();
    }
}
