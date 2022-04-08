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

#ifndef XMRIG_TCPSERVER_H
#define XMRIG_TCPSERVER_H


#include <uv.h>


#include "base/tools/Object.h"


namespace xmrig {


class ITcpServerListener;
class String;


class TcpServer
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(TcpServer)

    TcpServer(const String &host, uint16_t port, ITcpServerListener *listener);
    ~TcpServer();

    int bind();

private:
    void create(uv_stream_t *stream, int status);

    static void onConnection(uv_stream_t *stream, int status);

    const String &m_host;
    int m_version   = 0;
    ITcpServerListener *m_listener;
    sockaddr_storage m_addr{};
    uint16_t m_port;
    uv_tcp_t *m_tcp;
};


} /* namespace xmrig */


#endif /* XMRIG_TCPSERVER_H */
