/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/kernel/interfaces/ITcpServerListener.h"
#include "base/net/tools/TcpServer.h"
#include "base/tools/Handle.h"
#include "base/tools/String.h"


static const xmrig::String kLocalHost("127.0.0.1");


xmrig::TcpServer::TcpServer(const String &host, uint16_t port, ITcpServerListener *listener) :
    m_host(host.isNull() ? kLocalHost : host),
    m_listener(listener),
    m_port(port)
{
    assert(m_listener != nullptr);

    m_tcp = new uv_tcp_t;
    uv_tcp_init(uv_default_loop(), m_tcp);
    m_tcp->data = this;

    uv_tcp_nodelay(m_tcp, 1);

    if (m_host.contains(":") && uv_ip6_addr(m_host.data(), m_port, reinterpret_cast<sockaddr_in6 *>(&m_addr)) == 0) {
        m_version = 6;
    }
    else if (uv_ip4_addr(m_host.data(), m_port, reinterpret_cast<sockaddr_in *>(&m_addr)) == 0) {
        m_version = 4;
    }
}


xmrig::TcpServer::~TcpServer()
{
    Handle::close(m_tcp);
}


int xmrig::TcpServer::bind()
{
    if (!m_version) {
        return UV_EAI_ADDRFAMILY;
    }

    uv_tcp_bind(m_tcp, reinterpret_cast<const sockaddr*>(&m_addr), 0);

    const int rc = uv_listen(reinterpret_cast<uv_stream_t*>(m_tcp), 511, TcpServer::onConnection);
    if (rc != 0) {
        return rc;
    }

    if (!m_port) {
        sockaddr_storage storage = {};
        int size = sizeof(storage);

        uv_tcp_getsockname(m_tcp, reinterpret_cast<sockaddr*>(&storage), &size);

        m_port = ntohs(reinterpret_cast<sockaddr_in *>(&storage)->sin_port);
    }

    return m_port;
}


void xmrig::TcpServer::create(uv_stream_t *stream, int status)
{
    if (status < 0) {
        return;
    }

    m_listener->onConnection(stream, m_port);
}


void xmrig::TcpServer::onConnection(uv_stream_t *stream, int status)
{
    static_cast<TcpServer *>(stream->data)->create(stream, status);
}
