/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <microhttpd.h>
#include <string.h>

#include "common/api/HttpBody.h"
#include "common/api/HttpRequest.h"
#include "common/api/HttpReply.h"


#ifndef MHD_HTTP_PAYLOAD_TOO_LARGE
#   define MHD_HTTP_PAYLOAD_TOO_LARGE 413
#endif


xmrig::HttpRequest::HttpRequest(MHD_Connection *connection, const char *url, const char *method, const char *uploadData, size_t *uploadSize, void **cls) :
    m_fulfilled(true),
    m_restricted(true),
    m_uploadData(uploadData),
    m_url(url),
    m_body(static_cast<HttpBody*>(*cls)),
    m_method(Unsupported),
    m_connection(connection),
    m_uploadSize(uploadSize),
    m_cls(cls)
{
    if (strcmp(method, MHD_HTTP_METHOD_OPTIONS) == 0) {
        m_method = Options;
    }
    else if (strcmp(method, MHD_HTTP_METHOD_GET) == 0) {
        m_method = Get;
    }
    else if (strcmp(method, MHD_HTTP_METHOD_PUT) == 0) {
        m_method = Put;
    }
}


xmrig::HttpRequest::~HttpRequest()
{
    if (m_fulfilled) {
        delete m_body;
    }
}


bool xmrig::HttpRequest::match(const char *path) const
{
    return strcmp(m_url, path) == 0;
}


bool xmrig::HttpRequest::process(const char *accessToken, bool restricted, xmrig::HttpReply &reply)
{
    m_restricted = restricted || !accessToken;

    if (m_body) {
        if (*m_uploadSize != 0) {
            if (!m_body->write(m_uploadData, *m_uploadSize)) {
                *m_cls       = nullptr;
                m_fulfilled  = true;
                reply.status = MHD_HTTP_PAYLOAD_TOO_LARGE;
                return false;
            }

            *m_uploadSize = 0;
            m_fulfilled   = false;
            return true;
        }

        m_fulfilled = true;
        return true;
    }

    reply.status = auth(accessToken);
    if (reply.status != MHD_HTTP_OK) {
        return false;
    }

    if (m_restricted && m_method != Get) {
        reply.status = MHD_HTTP_FORBIDDEN;
        return false;
    }

    if (m_method == Get) {
        return true;
    }

    const char *contentType = MHD_lookup_connection_value(m_connection, MHD_HEADER_KIND, "Content-Type");
    if (!contentType || strcmp(contentType, "application/json") != 0) {
        reply.status = MHD_HTTP_UNSUPPORTED_MEDIA_TYPE;
        return false;
    }

    m_body      = new xmrig::HttpBody();
    m_fulfilled = false;
    *m_cls      = m_body;

    return true;
}


const char *xmrig::HttpRequest::body() const
{
    return m_body ? m_body->data() : nullptr;
}


int xmrig::HttpRequest::end(const HttpReply &reply)
{
    if (reply.buf) {
        return end(reply.status, MHD_create_response_from_buffer(reply.size ? reply.size : strlen(reply.buf), (void*) reply.buf, MHD_RESPMEM_MUST_FREE));
    }

    return end(reply.status, nullptr);
}


int xmrig::HttpRequest::end(int status, MHD_Response *rsp)
{
    if (!rsp) {
        rsp = MHD_create_response_from_buffer(0, nullptr, MHD_RESPMEM_PERSISTENT);
    }

    MHD_add_response_header(rsp, "Content-Type", "application/json");
    MHD_add_response_header(rsp, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(rsp, "Access-Control-Allow-Methods", "GET, PUT");
    MHD_add_response_header(rsp, "Access-Control-Allow-Headers", "Authorization, Content-Type");

    const int ret = MHD_queue_response(m_connection, status, rsp);
    MHD_destroy_response(rsp);
    return ret;
}


int xmrig::HttpRequest::auth(const char *accessToken)
{
    if (!accessToken) {
        return MHD_HTTP_OK;
    }

    const char *header = MHD_lookup_connection_value(m_connection, MHD_HEADER_KIND, "Authorization");
    if (accessToken && !header) {
        return MHD_HTTP_UNAUTHORIZED;
    }

    const size_t size = strlen(header);
    if (size < 8 || strlen(accessToken) != size - 7 || memcmp("Bearer ", header, 7) != 0) {
        return MHD_HTTP_FORBIDDEN;
    }

    return strncmp(accessToken, header + 7, strlen(accessToken)) == 0 ? MHD_HTTP_OK : MHD_HTTP_FORBIDDEN;
}
