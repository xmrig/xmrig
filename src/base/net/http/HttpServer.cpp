/* XMRig
 * Copyright (c) 2014-2019 heapwolf    <https://github.com/heapwolf>
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


#include <functional>
#include <uv.h>


#include "base/net/http/HttpServer.h"
#include "3rdparty/llhttp/llhttp.h"
#include "base/net/http/HttpContext.h"
#include "base/net/tools/NetBuffer.h"


xmrig::HttpServer::HttpServer(const std::shared_ptr<IHttpListener> &listener) :
    m_listener(listener)
{
}


xmrig::HttpServer::~HttpServer()
{
    HttpContext::closeAll();
}


void xmrig::HttpServer::onConnection(uv_stream_t *stream, uint16_t)
{
    auto ctx = new HttpContext(HTTP_REQUEST, m_listener);
    uv_accept(stream, ctx->stream());

    uv_read_start(ctx->stream(), NetBuffer::onAlloc,
        [](uv_stream_t *tcp, ssize_t nread, const uv_buf_t *buf)
        {
            auto ctx = static_cast<HttpContext*>(tcp->data);

            if (nread >= 0) {
                const auto size   = static_cast<size_t>(nread);
                const auto parsed = ctx->parse(buf->base, size);

                if (parsed < size) {
                    ctx->close();
                }
            } else {
                ctx->close();
            }

            NetBuffer::release(buf);
        });
}
