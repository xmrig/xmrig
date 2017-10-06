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

#ifndef __HTTPD_H__
#define __HTTPD_H__


#include <uv.h>


struct MHD_Connection;
struct MHD_Daemon;
struct MHD_Response;


class Httpd
{
public:
    Httpd(int port, const char *accessToken);
    bool start();

private:
    int auth(const char *header);

    static int done(MHD_Connection *connection, int status, MHD_Response *rsp);
    static int handler(void *cls, MHD_Connection *connection, const char *url, const char *method, const char *version, const char *upload_data, size_t *upload_data_size, void **con_cls);

    const char *m_accessToken;
    const int m_port;
    MHD_Daemon *m_daemon;
};

#endif /* __HTTPD_H__ */
