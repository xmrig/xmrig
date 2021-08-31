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

#ifndef XMRIG_HTTPSSERVER_H
#define XMRIG_HTTPSSERVER_H


using uv_tcp_t  = struct uv_tcp_s;

struct http_parser;
struct http_parser_settings;
struct uv_buf_t;


#include "base/kernel/interfaces/ITcpServerListener.h"
#include "base/tools/Object.h"


#include <memory>


namespace xmrig {


class IHttpListener;
class TlsContext;
class TlsConfig;


class HttpsServer : public ITcpServerListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(HttpsServer)

    HttpsServer(const std::shared_ptr<IHttpListener> &listener);
    ~HttpsServer() override;

    bool setTls(const TlsConfig &config);

protected:
    void onConnection(uv_stream_t *stream, uint16_t port) override;

private:
    static void onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);

    std::weak_ptr<IHttpListener> m_listener;
    TlsContext *m_tls   = nullptr;
};


} // namespace xmrig


#endif // XMRIG_HTTPSSERVER_H

