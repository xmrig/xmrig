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


#include "api/Httpd.h"
#include "log/Log.h"


static const char kNotFound []    = "{\"error\":\"NOT_FOUND\"}";
static const size_t kNotFoundSize = sizeof(kNotFound) - 1;


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

    m_daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, 4455, nullptr, nullptr, &Httpd::handler, this, MHD_OPTION_END);
    if (!m_daemon) {
        LOG_ERR("HTTP Daemon failed to start.");
        return false;
    }

    return true;
}


int Httpd::handler(void *cls, struct MHD_Connection *connection, const char *url, const char *method, const char *version, const char *upload_data, size_t *upload_data_size, void **con_cls)
{
    struct MHD_Response *rsp = MHD_create_response_from_buffer(kNotFoundSize, (void*)kNotFound, MHD_RESPMEM_PERSISTENT);

    MHD_add_response_header(rsp, "Content-Type", "application/json");

    const int ret = MHD_queue_response(connection, MHD_HTTP_NOT_FOUND, rsp);
    MHD_destroy_response(rsp);
    return ret;
}
