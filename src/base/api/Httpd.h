/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_HTTPD_H
#define XMRIG_HTTPD_H


#include "base/kernel/interfaces/IBaseListener.h"
#include "base/net/http/HttpListener.h"
#include "base/tools/Object.h"


#include <cstdint>
#include <memory>


namespace xmrig {


class Base;
class HttpServer;
class HttpsServer;
class TcpServer;


class Httpd : public IBaseListener, public IHttpListener
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Httpd)

    Httpd(Base *base);
    ~Httpd() override;

    bool start();
    void stop();

protected:
    void onConfigChanged(Config *config, Config *previousConfig) override;
    void onHttpData(const HttpData &data) override;

private:
    int auth(const HttpData &req) const;

    const Base *m_base;
    std::shared_ptr<IHttpListener> m_httpListener;
    TcpServer *m_server     = nullptr;
    uint16_t m_port         = 0;

#   ifdef XMRIG_FEATURE_TLS
    HttpsServer *m_http     = nullptr;
#   else
    HttpServer *m_http      = nullptr;
#   endif
};


} /* namespace xmrig */


#endif /* XMRIG_HTTPD_H */
