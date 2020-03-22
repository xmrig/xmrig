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


#include "base/api/Httpd.h"
#include "3rdparty/http-parser/http_parser.h"
#include "base/api/Api.h"
#include "base/io/log/Log.h"
#include "base/net/http/HttpApiResponse.h"
#include "base/net/http/HttpData.h"
#include "base/net/tools/TcpServer.h"
#include "core/config/Config.h"
#include "core/Controller.h"


#ifdef XMRIG_FEATURE_TLS
#   include "base/net/https/HttpsServer.h"
#else
#   include "base/net/http/HttpServer.h"
#endif


namespace xmrig {

static const char *kAuthorization = "authorization";

#ifdef _WIN32
static const char *favicon = nullptr;
static size_t faviconSize  = 0;
#endif

} // namespace xmrig


xmrig::Httpd::Httpd(Base *base) :
    m_base(base)
{
    m_httpListener = std::make_shared<HttpListener>(this);

    base->addListener(this);
}


xmrig::Httpd::~Httpd() = default;


bool xmrig::Httpd::start()
{
    const auto &config = m_base->config()->http();

    if (!config.isEnabled()) {
        return true;
    }

    bool tls = false;

#   ifdef XMRIG_FEATURE_TLS
    m_http = new HttpsServer(m_httpListener);
    tls = m_http->setTls(m_base->config()->tls());
#   else
    m_http = new HttpServer(m_httpListener);
#   endif

    m_server = new TcpServer(config.host(), config.port(), m_http);

    const int rc = m_server->bind();
    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CSI "1;%dm%s:%d" " " RED_BOLD("%s"),
               "HTTP API",
               tls ? 32 : 36,
               config.host().data(),
               rc < 0 ? config.port() : rc,
               rc < 0 ? uv_strerror(rc) : ""
               );

    if (rc < 0) {
        stop();

        return false;
    }

    m_port = static_cast<uint16_t>(rc);

#   ifdef _WIN32
    HRSRC src = FindResource(nullptr, MAKEINTRESOURCE(1), RT_ICON);
    if (src != nullptr) {
        HGLOBAL res = LoadResource(nullptr, src);
        if (res != nullptr) {
            favicon     = static_cast<const char *>(LockResource(res));
            faviconSize = SizeofResource(nullptr, src);
        }
    }
#   endif

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


void xmrig::Httpd::onHttpData(const HttpData &data)
{
    if (data.method == HTTP_OPTIONS) {
        return HttpApiResponse(data.id()).end();
    }

    if (data.method == HTTP_GET && data.url == "/favicon.ico") {
#       ifdef _WIN32
        if (favicon != nullptr) {
            HttpResponse response(data.id());
            response.setHeader(HttpData::kContentType, "image/x-icon");

            return response.end(favicon, faviconSize);
        }
#       endif

        return HttpResponse(data.id(), HTTP_STATUS_NOT_FOUND).end();
    }

    if (data.method > 4) {
        return HttpApiResponse(data.id(), HTTP_STATUS_METHOD_NOT_ALLOWED).end();
    }

    const int status = auth(data);
    if (status != HTTP_STATUS_OK) {
        return HttpApiResponse(data.id(), status).end();
    }

    if (data.method != HTTP_GET) {
        if (m_base->config()->http().isRestricted()) {
            return HttpApiResponse(data.id(), HTTP_STATUS_FORBIDDEN).end();
        }

        if (!data.headers.count(HttpData::kContentTypeL) || data.headers.at(HttpData::kContentTypeL) != HttpData::kApplicationJson) {
            return HttpApiResponse(data.id(), HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE).end();
        }
    }

    m_base->api()->request(data);
}


int xmrig::Httpd::auth(const HttpData &req) const
{
    const Http &config = m_base->config()->http();

    if (!req.headers.count(kAuthorization)) {
        return config.isAuthRequired() ? HTTP_STATUS_UNAUTHORIZED : HTTP_STATUS_OK;
    }

    if (config.token().isNull()) {
        return HTTP_STATUS_UNAUTHORIZED;
    }

    const std::string &token = req.headers.at(kAuthorization);
    const size_t size        = token.size();

    if (token.size() < 8 || config.token().size() != size - 7 || memcmp("Bearer ", token.c_str(), 7) != 0) {
        return HTTP_STATUS_FORBIDDEN;
    }

    return strncmp(config.token().data(), token.c_str() + 7, config.token().size()) == 0 ? HTTP_STATUS_OK : HTTP_STATUS_FORBIDDEN;
}
