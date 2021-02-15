/* xmlcore
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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


#include "base/net/https/HttpsContext.h"
#include "3rdparty/http-parser/http_parser.h"
#include "base/net/tls/TlsContext.h"


#include <openssl/bio.h>
#include <uv.h>


xmlcore::HttpsContext::HttpsContext(TlsContext *tls, const std::weak_ptr<IHttpListener> &listener) :
    HttpContext(HTTP_REQUEST, listener),
    ServerTls(tls ? tls->ctx() : nullptr)
{
    if (!tls) {
        m_mode = TLS_OFF;
    }
}


xmlcore::HttpsContext::~HttpsContext() = default;


void xmlcore::HttpsContext::append(char *data, size_t size)
{
    if (m_mode == TLS_AUTO) {
        m_mode = isTLS(data, size) ? TLS_ON : TLS_OFF;
    }

    if (m_mode == TLS_ON) {
        read(data, size);
    }
    else {
        parse(data, size);
    }
}


bool xmlcore::HttpsContext::write(BIO *bio)
{
    if (uv_is_writable(stream()) != 1) {
        return false;
    }

    char *data        = nullptr;
    const size_t size = BIO_get_mem_data(bio, &data);
    std::string body(data, size);

    (void) BIO_reset(bio);

    HttpContext::write(std::move(body), m_close);

    return true;
}


void xmlcore::HttpsContext::parse(char *data, size_t size)
{
    if (HttpContext::parse(data, size) < size) {
        close();
    }
}


void xmlcore::HttpsContext::shutdown()
{
    close();
}


void xmlcore::HttpsContext::write(std::string &&data, bool close)
{
    m_close = close;

    if (m_mode == TLS_ON) {
        send(data.data(), data.size());
    }
    else {
        HttpContext::write(std::move(data), close);
    }
}
