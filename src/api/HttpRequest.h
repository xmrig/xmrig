/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef __HTTPREQUEST_H__
#define __HTTPREQUEST_H__


#include <stdint.h>


struct MHD_Connection;
struct MHD_Response;


namespace xmrig {


class HttpBody;
class HttpReply;


class HttpRequest
{
public:
    enum Method {
        Unsupported,
        Options,
        Get,
        Put
    };

    HttpRequest(MHD_Connection *connection, const char *url, const char *method, const char *uploadData, size_t *uploadSize, void **cls);
    ~HttpRequest();

    inline bool isFulfilled() const  { return m_fulfilled; }
    inline bool isRestricted() const { return m_restricted; }
    inline Method method() const     { return m_method; }

    bool match(const char *path) const;
    bool process(const char *accessToken, bool restricted, xmrig::HttpReply &reply);
    const char *body() const;
    int end(const HttpReply &reply);
    int end(int status, MHD_Response *rsp);

private:
    int auth(const char *accessToken);

    bool m_fulfilled;
    bool m_restricted;
    const char *m_uploadData;
    const char *m_url;
    HttpBody *m_body;
    Method m_method;
    MHD_Connection *m_connection;
    size_t *m_uploadSize;
    void **m_cls;
};


} /* namespace xmrig */


#endif /* __HTTPREQUEST_H__ */
