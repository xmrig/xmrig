/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2014-2019 heapwolf    <https://github.com/heapwolf>
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


#include "base/net/http/HttpResponse.h"
#include "3rdparty/http-parser/http_parser.h"
#include "base/io/log/Log.h"
#include "base/net/http/HttpContext.h"


#include <cinttypes>
#include <cstring>
#include <sstream>
#include <uv.h>


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
    auto ctx = HttpContext::get(m_id);

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

    auto ctx         = HttpContext::get(m_id);
    std::string body = data ? (ss.str() + std::string(data, size)) : ss.str();

#   ifndef APP_DEBUG
    if (statusCode() >= 400)
#   endif
    {
        const bool err = statusCode() >= 400;

        Log::print(err ? Log::ERR : Log::INFO, CYAN("%s ") CLEAR MAGENTA_BOLD("%s") WHITE_BOLD(" %s ") CSI "1;%dm%d " CLEAR WHITE_BOLD("%zu ") CYAN_BOLD("%" PRIu64 "ms ") BLACK_BOLD("\"%s\""),
                   ctx->ip().c_str(),
                   http_method_str(static_cast<http_method>(ctx->method)),
                   ctx->url.c_str(),
                   err ? 31 : 32,
                   statusCode(),
                   body.size(),
                   ctx->elapsed(),
                   ctx->headers.count(kUserAgent) ? ctx->headers.at(kUserAgent).c_str() : nullptr
                   );
    }

    ctx->write(std::move(body), true);
}
