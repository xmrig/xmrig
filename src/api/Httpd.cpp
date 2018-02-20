/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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


#include "api/Api.h"
#include "api/Httpd.h"
#include "log/Log.h"


Httpd::Httpd(int port, const char *accessToken) :
    m_accessToken(accessToken),
    m_port(port),
    m_daemon(nullptr)
{
}


bool Httpd::start()
{
    if (!m_port) {
        return false;
    }

    m_daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, m_port, nullptr, nullptr, &Httpd::handler, this, MHD_OPTION_END);
    if (!m_daemon) {
        LOG_ERR("HTTP Daemon failed to start.");
        return false;
    }

    return true;
}


int Httpd::auth(const char *header)
{
    if (!m_accessToken) {
        return MHD_HTTP_OK;
    }

    if (m_accessToken && !header) {
        return MHD_HTTP_UNAUTHORIZED;
    }

    const size_t size = strlen(header);
    if (size < 8 || strlen(m_accessToken) != size - 7 || memcmp("Bearer ", header, 7) != 0) {
        return MHD_HTTP_FORBIDDEN;
    }

    return strncmp(m_accessToken, header + 7, strlen(m_accessToken)) == 0 ? MHD_HTTP_OK : MHD_HTTP_FORBIDDEN;
}


int Httpd::done(MHD_Connection *connection, int status, MHD_Response *rsp)
{
    if (!rsp) {
        rsp = MHD_create_response_from_buffer(0, nullptr, MHD_RESPMEM_PERSISTENT);
    }

    MHD_add_response_header(rsp, "Content-Type", "application/json");
    MHD_add_response_header(rsp, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(rsp, "Access-Control-Allow-Methods", "GET");
    MHD_add_response_header(rsp, "Access-Control-Allow-Headers", "Authorization");

    const int ret = MHD_queue_response(connection, status, rsp);
    MHD_destroy_response(rsp);
    return ret;
}


int Httpd::handler(void *cls, struct MHD_Connection *connection, const char *url, const char *method, const char *version, const char *upload_data, size_t *upload_data_size, void **con_cls)
{
    if (strcmp(method, "OPTIONS") == 0) {
        return done(connection, MHD_HTTP_OK, nullptr);
    }

    if (strcmp(method, "GET") != 0) {
        return MHD_NO;
    }

    int status = static_cast<Httpd*>(cls)->auth(MHD_lookup_connection_value(connection, MHD_HEADER_KIND, "Authorization"));
    if (status != MHD_HTTP_OK) {
        return done(connection, status, nullptr);
    }

    char *buf = Api::get(url, &status);
    if (buf == nullptr) {
        return MHD_NO;
    }

    MHD_Response *rsp = MHD_create_response_from_buffer(strlen(buf), (void*) buf, MHD_RESPMEM_MUST_FREE);
    return done(connection, status, rsp);
}
