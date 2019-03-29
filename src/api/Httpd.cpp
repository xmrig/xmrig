/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "api/Httpd.h"
#include "base/io/log/Log.h"
#include "base/net/http/HttpRequest.h"
#include "base/net/http/HttpResponse.h"
#include "base/net/http/HttpServer.h"
#include "base/net/tools/TcpServer.h"
#include "core/Config.h"
#include "core/Controller.h"


xmrig::Httpd::Httpd(Controller *controller) :
    m_controller(controller),
    m_http(nullptr),
    m_server(nullptr),
    m_port(0)
{
    controller->addListener(this);
}


xmrig::Httpd::~Httpd()
{
}


bool xmrig::Httpd::start()
{
    const Http &config = m_controller->config()->http();

    if (!config.isEnabled()) {
        return true;
    }

    m_http   = new HttpServer(this);
    m_server = new TcpServer(config.host(), config.port(), m_http);

    const int rc = m_server->bind();
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") BLUE_BOLD("%s:%d") " " RED_BOLD("%s"),
               "HTTP API",
               config.host().data(),
               rc < 0 ? config.port() : rc,
               rc < 0 ? uv_strerror(rc) : ""
               );

    if (rc < 0) {
        stop();

        return false;
    }

    m_port = static_cast<uint16_t>(rc);

    return true;
}


void xmrig::Httpd::stop()
{
    delete m_server;
    delete m_http;

    m_server = nullptr;
    m_http   = nullptr;
    m_port   = 0;
}



void xmrig::Httpd::onConfigChanged(Config *config, Config *previousConfig)
{
    if (config->http() == previousConfig->http()) {
        return;
    }

    stop();
    start();
}


void xmrig::Httpd::onHttpRequest(const HttpRequest &req)
{
    HttpResponse res(req.id());

    LOG_INFO(GREEN_BOLD_S "OK");
    res.end();
}
