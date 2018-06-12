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

#include <chrono>
#include <cstring>
#include <sstream>
#include <fstream>
#include <3rdparty/rapidjson/document.h>
#include <3rdparty/rapidjson/stringbuffer.h>
#include <3rdparty/rapidjson/writer.h>
#include <3rdparty/rapidjson/filewritestream.h>
#include <3rdparty/rapidjson/filereadstream.h>
#include <3rdparty/rapidjson/error/en.h>
#include <3rdparty/rapidjson/prettywriter.h>
#include <version.h>
#include "log/Log.h"
#include "Service.h"

uv_mutex_t Service::m_mutex;
std::map<std::string, ControlCommand> Service::m_clientCommand;
std::map<std::string, ClientStatus> Service::m_clientStatus;
int Service::m_currentServerTime = 0;

bool Service::start()
{
    uv_mutex_init(&m_mutex);

    return true;
}

void Service::release()
{
    uv_mutex_lock(&m_mutex);

    m_clientCommand.clear();
    m_clientStatus.clear();

    uv_mutex_unlock(&m_mutex);
}

unsigned Service::handleGET(const Options* options, const std::string& url, const std::string& clientId, std::string& resp)
{
    uv_mutex_lock(&m_mutex);

    unsigned resultCode = MHD_HTTP_NOT_FOUND;

    LOG_INFO("GET(url='%s', clientId='%s')", url.c_str(), clientId.c_str());

    if (url == "/") {
        resultCode = getAdminPage(options, resp);
    } else if (url.rfind("/admin/getClientStatusList", 0) == 0) {
        resultCode = getClientStatusList(resp);
    } else {
        if (!clientId.empty()) {
            if (url.rfind("/client/getConfig", 0) == 0 || url.rfind("/admin/getClientConfig", 0) == 0) {
                resultCode = getClientConfig(options, clientId, resp);
            } else if (url.rfind("/admin/getClientCommand", 0) == 0) {
                resultCode = getClientCommand(clientId, resp);
            }
        }
        else {
            LOG_ERR("Request does not contain clientId: %s", url.c_str());
        }
    }

    uv_mutex_unlock(&m_mutex);

    return resultCode;
}

unsigned Service::handlePOST(const Options* options, const std::string& url, const std::string& clientIp,
                             const std::string& clientId, const std::string& data, std::string& resp)
{
    uv_mutex_lock(&m_mutex);

    unsigned resultCode = MHD_HTTP_NOT_FOUND;

    LOG_INFO("POST(url='%s', clientIp='%s', clientId='%s', dataLen='%d')",
             url.c_str(), clientIp.c_str(), clientId.c_str(), data.length());

    if (url.rfind("/client/setClientStatus", 0) == 0) {
        resultCode = setClientStatus(clientIp, clientId, data, resp);
    } else if (url.rfind("/admin/setClientConfig", 0) == 0 || url.rfind("/client/setClientConfig", 0) == 0) {
        resultCode = setClientConfig(options, clientId, data, resp);
    } else if (url.rfind("/admin/setClientCommand", 0) == 0) {
        resultCode = setClientCommand(clientId, data, resp);
    } else if (url.rfind("/admin/resetClientStatusList", 0) == 0) {
        resultCode = resetClientStatusList(data, resp);
    }

    uv_mutex_unlock(&m_mutex);

    return resultCode;
}

unsigned Service::getClientConfig(const Options* options, const std::string& clientId, std::string& resp)
{
    unsigned resultCode = MHD_HTTP_INTERNAL_SERVER_ERROR;

    std::string clientConfigFileName = getClientConfigFileName(options, clientId);

    std::stringstream data;
    std::ifstream clientConfig(clientConfigFileName);
    if (clientConfig) {
        data << clientConfig.rdbuf();
        clientConfig.close();
    } else {
        std::ifstream defaultConfig("default_config.json");
        if (defaultConfig) {
            data << defaultConfig.rdbuf();
            defaultConfig.close();
        }
    }

    if (data.tellp() > 0) {
        rapidjson::Document document;
        document.Parse(data.str().c_str());

        if (!document.HasParseError()) {
            rapidjson::StringBuffer buffer(0, 4096);
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            writer.SetMaxDecimalPlaces(10);
            document.Accept(writer);

            resp = buffer.GetString();

            resultCode = MHD_HTTP_OK;
        } else {
            LOG_ERR("Not able to send client config. Client config %s is broken!", clientConfigFileName.c_str());
        }
    } else{
        LOG_ERR("Not able to load a client config. Please check your configuration!");
    }

    return resultCode;
}

unsigned Service::setClientConfig(const Options* options, const std::string &clientId, const std::string &data, std::string &resp)
{
    unsigned resultCode = MHD_HTTP_BAD_REQUEST;

    rapidjson::Document document;
    if (!document.Parse(data.c_str()).HasParseError()) {
        std::string clientConfigFileName = getClientConfigFileName(options, clientId);
        std::ofstream clientConfigFile(clientConfigFileName);

        if (clientConfigFile){
            rapidjson::StringBuffer buffer(0, 4096);
            rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
            writer.SetMaxDecimalPlaces(10);
            document.Accept(writer);

            clientConfigFile << buffer.GetString();
            clientConfigFile.close();

            resultCode = MHD_HTTP_OK;
        } else {
            LOG_ERR("Not able to store client config to file %s.", clientConfigFileName.c_str());
        }
    } else{
        LOG_ERR("Not able to store client config. The received client config for client %s is broken!", clientId.c_str());
    }

    return resultCode;
}

unsigned Service::getClientStatusList(std::string& resp)
{
    rapidjson::Document document;
    document.SetObject();

    auto& allocator = document.GetAllocator();

    rapidjson::Value clientStatusList(rapidjson::kArrayType);
    for (auto& clientStatus : m_clientStatus) {
        rapidjson::Value clientStatusEntry(rapidjson::kObjectType);
        clientStatusEntry.AddMember("client_status", clientStatus.second.toJson(allocator), allocator);
        clientStatusList.PushBack(clientStatusEntry, allocator);
    }

    auto time_point = std::chrono::system_clock::now();
    m_currentServerTime = std::chrono::system_clock::to_time_t(time_point);

    document.AddMember("current_server_time", m_currentServerTime, allocator);
    document.AddMember("current_version", rapidjson::StringRef(Version::string().c_str()), allocator);
    document.AddMember("client_status_list", clientStatusList, allocator);

    rapidjson::StringBuffer buffer(0, 4096);
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.SetMaxDecimalPlaces(10);
    document.Accept(writer);

    resp = buffer.GetString();

    return MHD_HTTP_OK;
}

unsigned Service::setClientStatus(const std::string& clientIp, const std::string& clientId, const std::string& data, std::string& resp)
{
    int resultCode = MHD_HTTP_BAD_REQUEST;

    rapidjson::Document document;
    if (!document.Parse(data.c_str()).HasParseError()) {
        LOG_INFO("Status from client: %s", clientId.c_str());

        ClientStatus clientStatus;
        clientStatus.parseFromJson(document);
        clientStatus.setExternalIp(clientIp);

        m_clientStatus[clientId] = clientStatus;

        resultCode = getClientCommand(clientId, resp);

        if (m_clientCommand[clientId].isOneTimeCommand()) {
            m_clientCommand.erase(clientId);
        }
    } else {
        LOG_ERR("Parse Error Occured: %d", document.GetParseError());
    }

    return resultCode;
}

unsigned Service::getClientCommand(const std::string& clientId, std::string& resp)
{
    if (m_clientCommand.find(clientId) == m_clientCommand.end()) {
        m_clientCommand[clientId] = ControlCommand();
    }

    rapidjson::Document respDocument;
    respDocument.SetObject();

    auto& allocator = respDocument.GetAllocator();

    rapidjson::Value controlCommand = m_clientCommand[clientId].toJson(allocator);
    respDocument.AddMember("control_command", controlCommand, allocator);

    rapidjson::StringBuffer buffer(0, 4096);
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.SetMaxDecimalPlaces(10);
    respDocument.Accept(writer);

    resp = buffer.GetString();

    return MHD_HTTP_OK;
}

unsigned Service::setClientCommand(const std::string& clientId, const std::string& data, std::string& resp)
{
    ControlCommand controlCommand;

    rapidjson::Document document;
    if (!document.Parse(data.c_str()).HasParseError()) {
        controlCommand.parseFromJson(document);

        m_clientCommand[clientId] = controlCommand;

        return MHD_HTTP_OK;
    } else {
        return MHD_HTTP_BAD_REQUEST;
    }
}

unsigned Service::resetClientStatusList(const std::string& data, std::string& resp)
{
    m_clientStatus.clear();

    return MHD_HTTP_OK;
}

unsigned Service::getAdminPage(const Options* options, std::string& resp)
{
    std::stringstream data;

    if (options->ccCustomDashboard() != nullptr) {
        std::ifstream customDashboard(options->ccCustomDashboard());
        if (customDashboard)
        {
            data << customDashboard.rdbuf();
            customDashboard.close();
            resp = data.str();
        }
    }

    if (resp.empty()) {
        data << "<!DOCTYPE html>";
        data << "<html lang=\"en\">";
        data << "<head>";
        data << "<meta charset=\"utf-8\">";
        data << "<title>XMRigCC Dashboard</title>";
        data << "</head>";
        data << "<body>";
        data << "    <div style=\"text-align: center;\">";
        data << "       <h1>Please configure a Dashboard</h1>";
        data << "    </div>";
        data << "</body>";
        data << "</html>";
    }

    resp = data.str();

    return MHD_HTTP_OK;
}

std::string Service::getClientConfigFileName(const Options* options, const std::string& clientId)
{
    std::string clientConfigFileName;

    if (options->ccClientConfigFolder() != nullptr) {
        clientConfigFileName += options->ccClientConfigFolder();
#       ifdef WIN32
        clientConfigFileName += '\\';
#       else
        clientConfigFileName += '/';
#       endif
    }

    clientConfigFileName += clientId + std::string("_config.json");

    return clientConfigFileName;
}
