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


#include "base/net/http/HttpResponse.h"
#include "3rdparty/llhttp/llhttp.h"
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
    ss << "HTTP/1.1 " << statusCode() << " " << HttpData::statusName(statusCode()) << kCRLF;

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
                   llhttp_method_name(static_cast<llhttp_method>(ctx->method)),
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
