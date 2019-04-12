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


#ifndef XMRIG_HTTPCLIENT_H
#define XMRIG_HTTPCLIENT_H


#include "base/net/http/HttpContext.h"
#include "base/kernel/interfaces/IDnsListener.h"


namespace xmrig {


class String;


class HttpClient : public HttpContext, public IDnsListener
{
public:
    HttpClient(int method, const String &url, IHttpListener *listener, const char *data = nullptr, size_t size = 0);
    ~HttpClient() override;

    inline uint16_t port() const     { return m_port; }
    inline void setQuiet(bool quiet) { m_quiet = quiet; }

    bool connect(const String &host, uint16_t port);
    const String &host() const;

protected:
    void onResolved(const Dns &dns, int status) override;

    virtual void handshake();
    virtual void read(const char *data, size_t size);
    virtual void write(const std::string &header);

    bool m_quiet;

private:
    static void onConnect(uv_connect_t *req, int status);

    Dns *m_dns;
    uint16_t m_port;
};


} // namespace xmrig


#endif // XMRIG_HTTPCLIENT_H

