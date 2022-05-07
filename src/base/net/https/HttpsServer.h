/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
 *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
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

    std::shared_ptr<TlsContext> m_tls;
    std::weak_ptr<IHttpListener> m_listener;
};


} // namespace xmrig


#endif // XMRIG_HTTPSSERVER_H

