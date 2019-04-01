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


#include <sstream>
#include <string.h>
#include <uv.h>


#include "3rdparty/http-parser/http_parser.h"
#include "base/io/log/Log.h"
#include "base/net/http/HttpContext.h"
#include "base/net/http/HttpResponse.h"


namespace xmrig {

static const char *kCRLF      = "\r\n";
static const char *kUserAgent = "user-agent";

} // namespace xmrig


xmrig::HttpResponse::HttpResponse(uint64_t id, int statusCode) :
    m_id(id),
    m_statusCode(statusCode)
{
}


bool xmrig::HttpResponse::isAlive() const
{
    HttpContext *ctx = HttpContext::get(m_id);

    return ctx && uv_is_writable(ctx->stream());
}


void xmrig::HttpResponse::end(const char *data, size_t size)
{
    if (!isAlive()) {
        return;
    }

    if (data && !size) {
        size = strlen(data);
    }

    if (size) {
        setHeader("Content-Length", std::to_string(size));
    }

    setHeader("Connection", "close");

    std::stringstream ss;
    ss << "HTTP/1.1 " << statusCode() << " " << http_status_str(static_cast<http_status>(statusCode())) << kCRLF;

    for (auto &header : m_headers) {
        ss << header.first << ": " << header.second << kCRLF;
    }

    ss << kCRLF;
    const std::string header = ss.str();

    uv_buf_t bufs[2];
    bufs[0].base = const_cast<char *>(header.c_str());

#   ifdef _WIN32
    bufs[0].len = static_cast<unsigned int>(header.size());
#   else
    bufs[0].len = header.size();
#   endif

    if (data) {
        bufs[1].base = const_cast<char *>(data);

#       ifdef _WIN32
        bufs[1].len = static_cast<unsigned int>(size);
#       else
        bufs[1].len = size;
#       endif
    }

    HttpContext *ctx = HttpContext::get(m_id);

#   ifndef APP_DEBUG
    if (statusCode() >= 400)
#   endif
    {
        const bool err        = statusCode() >= 400;
        char ip[46]           = {};
        sockaddr_storage addr = {};
        int aSize             = sizeof(addr);

        uv_tcp_getpeername(ctx->tcp, reinterpret_cast<sockaddr*>(&addr), &aSize);
        if (reinterpret_cast<sockaddr_in *>(&addr)->sin_family == AF_INET6) {
            uv_ip6_name(reinterpret_cast<sockaddr_in6*>(&addr), ip, 45);
        }
        else {
            uv_ip4_name(reinterpret_cast<sockaddr_in*>(&addr), ip, 16);
        }

        Log::print(err ? Log::ERR : Log::INFO, CYAN("%s ") CLEAR MAGENTA_BOLD("%s") WHITE_BOLD(" %s ") CSI "1;%dm%d " CLEAR WHITE_BOLD("%zu ") BLACK_BOLD("\"%s\""),
                   ip,
                   http_method_str(static_cast<http_method>(ctx->method)),
                   ctx->url.c_str(),
                   err ? 31 : 32,
                   statusCode(),
                   header.size() + size,
                   ctx->headers.count(kUserAgent) ? ctx->headers.at(kUserAgent).c_str() : nullptr
                   );
    }

    uv_try_write(ctx->stream(), bufs, data ? 2 : 1);

    ctx->close();
}
