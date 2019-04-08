/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2014-2019 heapwolf    <https://github.com/heapwolf>
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


#include <functional>
#include <uv.h>


#include "3rdparty/http-parser/http_parser.h"
#include "base/kernel/interfaces/IHttpListener.h"
#include "base/net/http/HttpContext.h"
#include "base/net/http/HttpResponse.h"
#include "base/net/http/HttpServer.h"


xmrig::HttpServer::HttpServer(IHttpListener *listener) :
    m_listener(listener)
{
}


xmrig::HttpServer::~HttpServer()
{
    HttpContext::closeAll();
}


void xmrig::HttpServer::onConnection(uv_stream_t *stream, uint16_t)
{
    HttpContext *ctx = new HttpContext(HTTP_REQUEST, m_listener);
    uv_accept(stream, ctx->stream());

    uv_read_start(ctx->stream(),
        [](uv_handle_t *, size_t suggested_size, uv_buf_t *buf)
        {
            buf->base = new char[suggested_size];

#           ifdef _WIN32
            buf->len = static_cast<unsigned int>(suggested_size);
#           else
            buf->len = suggested_size;
#           endif
        },
        [](uv_stream_t *tcp, ssize_t nread, const uv_buf_t *buf)
        {
            HttpContext *ctx = static_cast<HttpContext*>(tcp->data);

            if (nread >= 0) {
                const size_t size   = static_cast<size_t>(nread);
                const size_t parsed = ctx->parse(buf->base, size);

                if (parsed < size) {
                    ctx->close();
                }
            } else {
                ctx->close();
            }

            delete [] buf->base;
        });
}
