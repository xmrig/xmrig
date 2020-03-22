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


#include "base/net/stratum/Socks5.h"


xmrig::Client::Socks5::Socks5(Client *client) :
    m_client(client)
{
}


bool xmrig::Client::Socks5::read(const char *data, size_t size)
{
    if (size < m_nextSize) {
        return false;
    }

    if (data[0] == 0x05 && data[1] == 0x00) {
        if (m_state == SentInitialHandshake) {
            connect();
        }
        else {
            m_state = Ready;
        }
    }
    else {
        m_client->close();
    }

    return true;
}


void xmrig::Client::Socks5::handshake()
{
    m_nextSize  = 2;
    m_state     = SentInitialHandshake;

    char buf[3] = { 0x05, 0x01, 0x00 };

    m_client->write(uv_buf_init(buf, sizeof (buf)));
}


bool xmrig::Client::Socks5::isIPv4(const String &host, sockaddr_storage *addr) const
{
    return uv_ip4_addr(host.data(), 0, reinterpret_cast<sockaddr_in *>(addr)) == 0;
}


bool xmrig::Client::Socks5::isIPv6(const String &host, sockaddr_storage *addr) const
{
   return uv_ip6_addr(host.data(), 0, reinterpret_cast<sockaddr_in6 *>(addr)) == 0;
}


void xmrig::Client::Socks5::connect()
{
    m_nextSize  = 5;
    m_state     = SentFinalHandshake;

    const auto &host = m_client->pool().host();
    std::vector<uint8_t> buf;
    sockaddr_storage addr{};

    if (isIPv4(host, &addr)) {
        buf.resize(10);
        buf[3] = 0x01;
        memcpy(buf.data() + 4, &reinterpret_cast<sockaddr_in *>(&addr)->sin_addr, 4);
    }
    else if (isIPv6(host, &addr)) {
        buf.resize(22);
        buf[3] = 0x04;
        memcpy(buf.data() + 4, &reinterpret_cast<sockaddr_in6 *>(&addr)->sin6_addr, 16);
    }
    else {
        buf.resize(host.size() + 7);
        buf[3] = 0x03;
        buf[4] = static_cast<uint8_t>(host.size());
        memcpy(buf.data() + 5, host.data(), host.size());
    }

    buf[0] = 0x05;
    buf[1] = 0x01;
    buf[2] = 0x00;

    const uint16_t port = htons(m_client->pool().port());
    memcpy(buf.data() + (buf.size() - sizeof(port)), &port, sizeof(port));

    m_client->write(uv_buf_init(reinterpret_cast<char *>(buf.data()), buf.size()));
}
