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
#include "base/tools/Baton.h"


namespace xmrig {

static const char *kCRLF      = "\r\n";
static const char *kUserAgent = "user-agent";


class WriteBaton : public Baton<uv_write_t>
{
public:
    inline WriteBaton(const std::stringstream &ss, const char *data, size_t size, HttpContext *ctx) :
        m_ctx(ctx),
        m_header(ss.str())
    {
        bufs[0].len  = m_header.size();
        bufs[0].base = const_cast<char *>(m_header.c_str());

        if (data) {
            bufs[1].len  = size;
            bufs[1].base = new char[size];
            memcpy(bufs[1].base, data, size);
        }
        else {
            bufs[1].base = nullptr;
            bufs[1].len  = 0;
        }
    }


    inline ~WriteBaton()
    {
        if (count() == 2) {
            delete [] bufs[1].base;
        }

        m_ctx->close();
    }


    inline size_t count() const                      { return bufs[1].base == nullptr ? 1 : 2; }
    inline size_t size() const                       { return bufs[0].len + bufs[1].len; }
    inline static void onWrite(uv_write_t *req, int) { delete reinterpret_cast<WriteBaton *>(req->data); }


    uv_buf_t bufs[2];

private:
    HttpContext *m_ctx;
    std::string m_header;
};

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

    HttpContext *ctx  = HttpContext::get(m_id);
    WriteBaton *baton = new WriteBaton(ss, data, size, ctx);

#   ifndef APP_DEBUG
    if (statusCode() >= 400)
#   endif
    {
        const bool err = statusCode() >= 400;

        Log::print(err ? Log::ERR : Log::INFO, CYAN("%s ") CLEAR MAGENTA_BOLD("%s") WHITE_BOLD(" %s ") CSI "1;%dm%d " CLEAR WHITE_BOLD("%zu ") BLACK_BOLD("\"%s\""),
                   ctx->ip().c_str(),
                   http_method_str(static_cast<http_method>(ctx->method)),
                   ctx->url.c_str(),
                   err ? 31 : 32,
                   statusCode(),
                   baton->size(),
                   ctx->headers.count(kUserAgent) ? ctx->headers.at(kUserAgent).c_str() : nullptr
                   );
    }

    uv_write(&baton->req, ctx->stream(), baton->bufs, baton->count(), WriteBaton::onWrite);
}
