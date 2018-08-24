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
#include <sstream>
#include <fstream>
#include <microhttpd.h>
#include <memory>

#include "version.h"
#include "Service.h"
#include "Httpd.h"
#include "log/Log.h"

Httpd::Httpd(const Options *options)
        : m_options(options)
        , m_daemon(nullptr)
{
}

bool Httpd::start()
{
    if (!m_options->ccPort()) {
        return false;
    }

#   ifndef XMRIG_NO_TLS
    if (m_options->ccUseTls()) {

        m_keyPem = readFile(m_options->ccKeyFile());
        m_certPem = readFile(m_options->ccCertFile());

        if (m_keyPem.empty() || m_certPem.empty()) {
            LOG_ERR("HTTPS Daemon failed to start. Unable to load Key/Cert.");
            return false;
        }

        m_daemon = MHD_start_daemon (MHD_USE_SELECT_INTERNALLY | MHD_USE_SSL,
                                   static_cast<uint16_t>(m_options->ccPort()), nullptr, nullptr, &Httpd::handler,
                                   this, MHD_OPTION_CONNECTION_TIMEOUT, (unsigned int) 25,
                                   MHD_OPTION_HTTPS_MEM_KEY, m_keyPem.c_str(),
                                   MHD_OPTION_HTTPS_MEM_CERT, m_certPem.c_str(),
                                   MHD_OPTION_END);
    } else {
#   endif
        m_daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, static_cast<uint16_t>(m_options->ccPort()), nullptr,
                                    nullptr, &Httpd::handler,
                                    this, MHD_OPTION_CONNECTION_TIMEOUT, (unsigned int) 25, MHD_OPTION_END);
#   ifndef XMRIG_NO_TLS
    }
#   endif

    if (!m_daemon) {
        LOG_ERR("HTTP Daemon failed to start.");
        return false;
    } else {
        LOG_INFO("%s Server started on Port: %d %s", APP_NAME, m_options->ccPort(), m_options->ccUseTls() ? "with TLS" : "");
    }

    return true;
}

std::string Httpd::readFile(const std::string &fileName)
{
    std::stringstream data;
    std::ifstream file(fileName);
    if (file) {
        data << file.rdbuf();
        file.close();
    }

    return data.str();
}

unsigned Httpd::tokenAuth(struct MHD_Connection* connection, const std::string& clientIp)
{
    if (!m_options->ccToken()) {
        LOG_WARN("[%s] 200 OK - WARNING AccessToken not set!", clientIp.c_str());
        return MHD_HTTP_OK;
    }

    const char* header = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, MHD_HTTP_HEADER_AUTHORIZATION);
    if (m_options->ccToken() && !header) {
        LOG_WARN("[%s] 401 UNAUTHORIZED", clientIp.c_str());
        return MHD_HTTP_UNAUTHORIZED;
    }

    const size_t size = strlen(header);
    if (size < 8 || strlen(m_options->ccToken()) != size - 7 || memcmp("Bearer ", header, 7) != 0) {
        LOG_ERR("[%s] 403 FORBIDDEN - AccessToken wrong!", clientIp.c_str());
        return MHD_HTTP_FORBIDDEN;
    }

    return strncmp(m_options->ccToken(), header + 7, strlen(m_options->ccToken())) == 0 ? MHD_HTTP_OK : MHD_HTTP_FORBIDDEN;
}

unsigned Httpd::basicAuth(struct MHD_Connection* connection, const std::string& clientIp, std::string& resp)
{
    unsigned result = MHD_HTTP_OK;

    if (!m_options->ccAdminUser() || !m_options->ccAdminPass()) {
        resp = std::string("<html><body\\>"
                           "Please configure admin user and pass to view this Page."
                           "</body><html\\>");

        LOG_ERR("[%s] 403 FORBIDDEN - Admin user/password not set!",  clientIp.c_str());
        result = MHD_HTTP_FORBIDDEN;
    }
    else {
        const char* header = MHD_lookup_connection_value(connection, MHD_HEADER_KIND, MHD_HTTP_HEADER_AUTHORIZATION);
        if (header) {
            char* user = nullptr;
            char* pass = nullptr;

            user = MHD_basic_auth_get_username_password(connection, &pass);
            if (user == nullptr || strcmp(user, m_options->ccAdminUser()) != 0 ||
                pass == nullptr || strcmp(pass, m_options->ccAdminPass()) != 0) {

                LOG_ERR("[%s] 403 FORBIDDEN - Admin user/password wrong!",  clientIp.c_str());
                result = MHD_HTTP_UNAUTHORIZED;
            }

            if (user) {
                free(user);
            }

            if (pass) {
                free(pass);
            }
        } else {
            LOG_WARN("[%s] 401 UNAUTHORIZED",  clientIp.c_str());
            result = MHD_HTTP_UNAUTHORIZED;
        }
    }

    return result;
}

int Httpd::sendResponse(MHD_Connection* connection, unsigned status, MHD_Response* rsp, const char* contentType)
{
    if (!rsp) {
        rsp = MHD_create_response_from_buffer(0, nullptr, MHD_RESPMEM_MUST_COPY);
    }

    MHD_add_response_header(rsp, "Content-Type", contentType);
    MHD_add_response_header(rsp, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(rsp, "Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    MHD_add_response_header(rsp, "Access-Control-Allow-Headers", "Content-Type, Authorization");
    MHD_add_response_header(rsp, "WWW-Authenticate", "Basic");
    MHD_add_response_header(rsp, "WWW-Authenticate", "Bearer");

    int ret = MHD_queue_response(connection, status, rsp);

    MHD_destroy_response(rsp);

    return ret;
}


int Httpd::handler(void* httpd, MHD_Connection* connection, const char* url, const char* method,
                   const char* version, const char* upload_data, size_t* upload_data_size, void** con_cls)
{
    std::string clientIp;
    const MHD_ConnectionInfo *info = MHD_get_connection_info(connection, MHD_CONNECTION_INFO_CLIENT_ADDRESS);
    if (info) {
        char clientHost[NI_MAXHOST];
        int ec = getnameinfo(info->client_addr, sizeof(*info->client_addr), clientHost, sizeof(clientHost),
                             0, 0, NI_NUMERICHOST|NI_NUMERICSERV);

        if (ec == 0) {
            clientIp = std::string(clientHost);
        }
    }

    if (strcmp(method, MHD_HTTP_METHOD_OPTIONS) == 0) {
        LOG_INFO("[%s] OPTIONS Requested",  clientIp.c_str());
        return sendResponse(connection, MHD_HTTP_OK, nullptr, CONTENT_TYPE_HTML);
    }

    if (strcmp(method, MHD_HTTP_METHOD_GET) != 0 && strcmp(method, MHD_HTTP_METHOD_POST) != 0) {
        LOG_ERR("[%s] 405 METHOD NOT ALLOWED (%s)",  clientIp.c_str(), method);
        return sendResponse(connection, MHD_HTTP_METHOD_NOT_ALLOWED, nullptr, CONTENT_TYPE_HTML);
    }

    if (strstr(url, "/client/")) {
        unsigned status = static_cast<Httpd*>(httpd)->tokenAuth(connection, clientIp);
        if (status != MHD_HTTP_OK) {
            return sendResponse(connection, status, nullptr, CONTENT_TYPE_JSON);
        }
    } else {
        std::string resp;
        unsigned status = static_cast<Httpd*>(httpd)->basicAuth(connection, clientIp, resp);
        if (status != MHD_HTTP_OK) {
            MHD_Response* rsp = nullptr;
            if (!resp.empty()) {
                rsp = MHD_create_response_from_buffer(resp.length(), (void*)resp.c_str(), MHD_RESPMEM_MUST_COPY);
            }
            return sendResponse(connection, status, rsp, CONTENT_TYPE_HTML);
        }
    }

    if (strcmp(method, MHD_HTTP_METHOD_GET) == 0) {
        return handleGET(static_cast<Httpd*>(httpd), connection, clientIp, url);
    } else {
        return handlePOST(static_cast<Httpd*>(httpd), connection, clientIp, url, upload_data, upload_data_size, con_cls);
    }
}

int Httpd::handleGET(const Httpd* httpd, struct MHD_Connection* connection, const std::string& clientIp, const char* urlPtr)
{
    std::string resp;
    std::string url(urlPtr);
    std::string clientId;

    const char* clientIdPtr = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "clientId");
    if (clientIdPtr)
    {
        clientId = std::string(clientIdPtr);
    }

    unsigned status = Service::handleGET(httpd->m_options, url, clientIp, clientId, resp);

    MHD_Response* rsp = nullptr;
    if (!resp.empty()) {
        rsp = MHD_create_response_from_buffer(resp.length(), (void*) resp.c_str(), MHD_RESPMEM_MUST_COPY);
    }

    char* contentType;
    if (url == "/") {
        contentType = const_cast<char*>(CONTENT_TYPE_HTML);
    } else {
        contentType = const_cast<char*>(CONTENT_TYPE_JSON);
    }

    return sendResponse(connection, status, rsp, contentType);
}

int Httpd::handlePOST(const Httpd* httpd, struct MHD_Connection* connection, const std::string& clientIp, const char* urlPtr, const char* upload_data,
                      size_t* upload_data_size, void** con_cls)
{
    auto* cc = (ConnectionContext*)* con_cls;
    if (cc == nullptr) {
        cc = new ConnectionContext();
        *con_cls = (void*) cc;
    } else {
        if (*upload_data_size != 0) {
            cc->data << std::string(upload_data, *upload_data_size);

            *upload_data_size = 0;
        } else {
            std::string resp;
            std::string url(urlPtr);
            std::string clientId;

            const char* clientIdPtr = MHD_lookup_connection_value(connection, MHD_GET_ARGUMENT_KIND, "clientId");
            if (clientIdPtr) {
                clientId = std::string(clientIdPtr);
            }

            unsigned status = Service::handlePOST(httpd->m_options, url, clientIp, clientId, cc->data.str(), resp);

            MHD_Response* rsp = nullptr;
            if (!resp.empty()) {
                rsp = MHD_create_response_from_buffer(resp.length(), (void*) resp.c_str(), MHD_RESPMEM_MUST_COPY);
            }

            delete cc;
            *con_cls = nullptr;

            return sendResponse(connection, status, rsp, CONTENT_TYPE_JSON);
        }
    }

    return MHD_YES;
}
