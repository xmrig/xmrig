/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2017-     BenDr0id    <ben@graef.in>
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

#include <cstring>
#include <microhttpd.h>
#include <memory>

#include "version.h"
#include "server/Service.h"
#include "server/Httpd.h"
#include "log/Log.h"

Httpd::Httpd(int port, const char *accessToken) :
        m_accessToken(accessToken),
        m_adminUser("admin"),
        m_adminPassword("passw0rd"),
        m_port(port),
        m_daemon(nullptr)
{
}

bool Httpd::start()
{
    if (!m_port) {
        return false;
    }

    m_daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, m_port, nullptr, nullptr, &Httpd::handler,
                                this, MHD_OPTION_END);

    if (!m_daemon) {
        LOG_ERR("HTTP Daemon failed to start.");
        return false;
    } else {
        LOG_INFO("%s Server started on Port: %d", APP_NAME, m_port);
    }

    return true;
}

unsigned Httpd::tokenAuth(struct MHD_Connection *connection)
{
    if (!m_accessToken) {
        LOG_WARN("AccessToken not set. Access Granted!");
        return MHD_HTTP_OK;
    }

    const char *header = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, MHD_HTTP_HEADER_AUTHORIZATION);
    if (m_accessToken && !header) {
        return MHD_HTTP_UNAUTHORIZED;
    }

    const size_t size = strlen(header);
    if (size < 8 || strlen(m_accessToken) != size - 7 || memcmp("Bearer ", header, 7) != 0) {
        LOG_WARN("AccessToken wrong. Access Forbidden!");
        return MHD_HTTP_FORBIDDEN;
    }

    return strncmp(m_accessToken, header + 7, strlen(m_accessToken)) == 0 ? MHD_HTTP_OK : MHD_HTTP_FORBIDDEN;
}

unsigned Httpd::basicAuth(struct MHD_Connection *connection, std::string &resp)
{
    if (!m_adminUser || !m_adminPassword) {
        resp = std::string("<html><body\\>"
                           "Please configure adminUser and adminPass to view this Page."
                           "</body><html\\>");

        LOG_WARN("AdminUser/AdminPassword not set. Access Forbidden!");
        return MHD_HTTP_FORBIDDEN;
    }

    const char *header = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, MHD_HTTP_HEADER_AUTHORIZATION);
    if (!header) {
        return MHD_HTTP_UNAUTHORIZED;
    }

    char* user;
    char* pass;

    user = MHD_basic_auth_get_username_password(connection, &pass);

    if (user == nullptr || strcmp(user, m_adminUser) != 0 ||
        pass == nullptr || strcmp(pass, m_adminPassword) != 0) {

        LOG_WARN("AdminUser/AdminPassword wrong. Access Unauthorized!");
        return MHD_HTTP_UNAUTHORIZED;
    }

    return MHD_HTTP_OK;
}

int Httpd::sendJSONResponse(MHD_Connection *connection, unsigned status, MHD_Response *rsp)
{
    return sendResponse(connection, status, rsp, "application/json");
}

int Httpd::sendHTMLResponse(MHD_Connection *connection, unsigned status, MHD_Response *rsp)
{
    return sendResponse(connection, status, rsp, "text/html");
}

int Httpd::sendResponse(MHD_Connection *connection, unsigned status, MHD_Response *rsp, const char *contentType)
{
    if (!rsp) {
        rsp = MHD_create_response_from_buffer(0, nullptr, MHD_RESPMEM_PERSISTENT);
    }

    MHD_add_response_header(rsp, "Content-Type", contentType);
    MHD_add_response_header(rsp, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(rsp, "Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    MHD_add_response_header(rsp, "Access-Control-Allow-Headers", "Authorization");
    MHD_add_response_header(rsp, "WWW-Authenticate", "Basic");
    MHD_add_response_header(rsp, "WWW-Authenticate", "Bearer");

    int ret = MHD_queue_response(connection, status, rsp);

    MHD_destroy_response(rsp);

    return ret;
}


int Httpd::handler(void *cls, MHD_Connection *connection, const char *url, const char *method,
                   const char *version, const char *upload_data, size_t *upload_data_size, void **con_cls)
{
    if (strcmp(method, MHD_HTTP_METHOD_OPTIONS) == 0) {
        LOG_INFO("OPTIONS Requested");
        return sendHTMLResponse(connection, MHD_HTTP_OK, nullptr);
    }

    if (strcmp(method, MHD_HTTP_METHOD_GET) != 0 && strcmp(method, MHD_HTTP_METHOD_POST) != 0) {
        LOG_ERR("HTTP_METHOD_NOT_ALLOWED");
        return sendHTMLResponse(connection, MHD_HTTP_METHOD_NOT_ALLOWED, nullptr);
    }

    if (strstr(url, "/client/")) {
        unsigned status = static_cast<Httpd *>(cls)->tokenAuth(connection);
        if (status != MHD_HTTP_OK) {
            return sendJSONResponse(connection, status, nullptr);
        }
    } else {
        std::string resp;
        unsigned status = static_cast<Httpd *>(cls)->basicAuth(connection, resp);
        if (status != MHD_HTTP_OK) {
            MHD_Response *rsp = nullptr;
            if (!resp.empty()) {
                rsp = MHD_create_response_from_buffer(resp.length(), (void *)resp.c_str(), MHD_RESPMEM_PERSISTENT);
            }
            return sendHTMLResponse(connection, status, rsp);
        }
    }

    if (strcmp(method, MHD_HTTP_METHOD_GET) == 0) {
        return handleGET(connection, url);
    } else {
        return handlePOST(connection, url, upload_data, upload_data_size, con_cls);
    }

    return MHD_NO;
}

int Httpd::handleGET(struct MHD_Connection *connection, const char *urlPtr)
{
    LOG_INFO("HANDLE GET REQUEST");

    std::string resp;
    std::string url(urlPtr, strlen(urlPtr));

    unsigned status = Service::get(url, resp);

    MHD_Response *rsp = nullptr;
    if (!resp.empty()) {
        rsp = MHD_create_response_from_buffer(resp.length(), (void *) resp.c_str(), MHD_RESPMEM_PERSISTENT);
    }

    return sendJSONResponse(connection, status, rsp);
}

int Httpd::handlePOST(struct MHD_Connection *connection, const char* urlPtr, const char *upload_data,
                      size_t *upload_data_size, void **con_cls)
{
    LOG_INFO("HANDLE POST REQUEST");

    auto *cc = (ConnectionContext*) *con_cls;
    if (cc == nullptr) {
        cc = new ConnectionContext();
        *con_cls = (void *) cc;
    } else {
        if (*upload_data_size != 0) {
            cc->data << std::string(upload_data, *upload_data_size);

            *upload_data_size = 0;
        } else {

            std::string resp;
            std::string url(urlPtr, strlen(urlPtr));

            unsigned status = Service::post(url, cc->data.str(), resp);

            MHD_Response *rsp = nullptr;
            if (!resp.empty()) {
                rsp = MHD_create_response_from_buffer(resp.length(), (void *) resp.c_str(), MHD_RESPMEM_PERSISTENT);
            }

            delete cc;
            *con_cls = nullptr;

            return sendJSONResponse(connection, status, rsp);
        }
    }

    return MHD_YES;
}
