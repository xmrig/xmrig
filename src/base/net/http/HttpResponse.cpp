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


#include <uv.h>


#include "3rdparty/http-parser/http_parser.h"
#include "base/net/http/HttpContext.h"
#include "base/net/http/HttpResponse.h"


namespace xmrig {

static const char *kCRLF             = "\r\n";
static const char *kTransferEncoding = "Transfer-Encoding";

} // namespace xmrig


xmrig::HttpResponse::HttpResponse() :
    parser(nullptr),
    statusCode(HTTP_STATUS_OK),
    body(""),
    statusAdjective("OK"), // FIXME
    m_writtenOrEnded(false)
{
}


void xmrig::HttpResponse::writeOrEnd(const std::string &str, bool end)
{
    std::stringstream ss;

    if (!m_writtenOrEnded) {
        ss << "HTTP/1.1 " << statusCode << " " << statusAdjective << kCRLF;

        for (auto &header : headers) {
            ss << header.first << ": " << header.second << kCRLF;
        }

        ss << kCRLF;
        m_writtenOrEnded = true;
    }

    if (headers.count(kTransferEncoding) && headers[kTransferEncoding] == "chunked") {
        ss << std::hex << str.size() << std::dec << kCRLF << str << kCRLF;

        if (end) {
            ss << "0" << kCRLF << kCRLF;
        }
    }
    else {
        ss << str;
    }

    const std::string out = ss.str();

#   ifdef _WIN32
    uv_buf_t resbuf = uv_buf_init(const_cast<char *>(out.c_str()), static_cast<unsigned int>(out.size()));
#   else
    uv_buf_t resbuf = uv_buf_init(const_cast<char *>(out.c_str()), out.size());
#   endif

    HttpContext* context = static_cast<HttpContext*>(parser->data);

    uv_try_write(context->stream(), &resbuf, 1);

    if (end) {
        if (!uv_is_closing(context->handle())) {
            uv_close(context->handle(), HttpContext::close);
        }
    }
}

