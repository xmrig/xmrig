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

#include <zconf.h>
#include <fstream>
#include <3rdparty/rapidjson/stringbuffer.h>
#include <3rdparty/rapidjson/prettywriter.h>

#include "CCClient.h"
#include "App.h"
#include "ControlCommand.h"

#include "api/NetworkState.h"
#include "log/Log.h"
#include "workers/Workers.h"
#include "workers/Hashrate.h"

CCClient *CCClient::m_self = nullptr;
uv_mutex_t CCClient::m_mutex;

CCClient::CCClient(const Options *options)
    : m_options(options)
{
    m_self = this;

    std::string clientId;
    if (m_options->ccWorkerId()){
        clientId = m_options->ccWorkerId();
    } else{
        char hostname[128];
        memset(hostname, 0, sizeof(hostname));
        gethostname(hostname, sizeof(hostname)-1);
        clientId = std::string(hostname);
    }

    m_clientStatus.setClientId(clientId);
    m_serverURL = std::string("http://") + options->ccUrl();

    if (m_options->ccToken() != nullptr) {
        m_authorization = std::string("Authorization: Bearer ") + m_self->m_options->ccToken();
    }

    uv_timer_init(uv_default_loop(), &m_timer);
    uv_timer_start(&m_timer, CCClient::onReport, kTickInterval, kTickInterval);
}

CCClient::~CCClient()
{
    uv_timer_stop(&m_timer);
    m_self = nullptr;
}

void CCClient::updateHashrate(const Hashrate *hashrate)
{
    uv_mutex_lock(&m_mutex);

    m_self->m_clientStatus.setHashrateShort(hashrate->calc(Hashrate::ShortInterval));
    m_self->m_clientStatus.setHashrateMedium(hashrate->calc(Hashrate::MediumInterval));
    m_self->m_clientStatus.setHashrateLong(hashrate->calc(Hashrate::LargeInterval));
    m_self->m_clientStatus.setHashrateHighest(hashrate->highest());

    uv_mutex_unlock(&m_mutex);
}


void CCClient::updateNetworkState(const NetworkState &network)
{
    uv_mutex_lock(&m_mutex);

    m_self->m_clientStatus.setCurrentStatus(Workers::isEnabled() ? "mining" : "paused");
    m_self->m_clientStatus.setCurrentPool(network.pool);
    m_self->m_clientStatus.setSharesGood(network.accepted);
    m_self->m_clientStatus.setSharesTotal(network.accepted + network.rejected);
    m_self->m_clientStatus.setHashesTotal(network.total);
    m_self->m_clientStatus.setAvgTime(network.avgTime());

    uv_mutex_unlock(&m_mutex);
}

void CCClient::publishClientStatusReport()
{
    std::string requestUrl = m_self->m_serverURL + "/client/setClientStatus?clientId=" + m_self->m_clientStatus.getClientId();
    std::string requestBuffer = m_self->m_clientStatus.toJsonString();
    std::string responseBuffer;

    CURLcode res = performCurl(requestUrl, requestBuffer, "POST", responseBuffer);
    if (res != CURLE_OK) {
        LOG_ERR("CCClient error: %s", curl_easy_strerror(res));
    } else {
        ControlCommand controlCommand;
        if (controlCommand.parseFromJsonString(responseBuffer)) {
            if (controlCommand.getCommand() == ControlCommand::START) {
                if (!Workers::isEnabled()) {
                    LOG_INFO("Command: START received -> Resuming");
                    Workers::setEnabled(true);
                }
            } else if (controlCommand.getCommand() == ControlCommand::STOP) {
                if (!Workers::isEnabled()) {
                    LOG_INFO("Command: STOP received -> Pausing");
                    Workers::setEnabled(true);
                }
            } else if (controlCommand.getCommand() == ControlCommand::UPDATE_CONFIG) {
                LOG_INFO("Command: UPDATE_CONFIG received -> Updating config");
                updateConfig();
            } else if (controlCommand.getCommand() == ControlCommand::RELOAD) {
                LOG_INFO("Command: RELOAD received -> Reload");
                App::reloadConfig();
            } else {
                LOG_ERR("Command: GET_CONFIG received -> NOT IMPLEMENTED YET!");
            }
        } else {
            LOG_ERR("Unknown Command received from CC Server.");
        }
    }
}

void CCClient::updateConfig()
{
    std::string requestUrl = m_self->m_serverURL + "/client/getConfig?clientId=" + m_self->m_clientStatus.getClientId();
    std::string requestBuffer;
    std::string responseBuffer;

    CURLcode res = performCurl(requestUrl, requestBuffer, "GET", responseBuffer);
    if (res != CURLE_OK) {
        LOG_ERR("CCClient error: %s", curl_easy_strerror(res));
    } else {
        rapidjson::Document document;
        if (!document.Parse(responseBuffer.c_str()).HasParseError()) {
            std::ofstream clientConfigFile(m_self->m_options->configFile());
            if (clientConfigFile) {
                rapidjson::StringBuffer buffer(0, 65536);
                rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
                writer.SetMaxDecimalPlaces(10);
                document.Accept(writer);

                clientConfigFile << buffer.GetString();
                clientConfigFile.close();

                LOG_INFO("Config update done. Reload.");
                App::reloadConfig();
            } else {
                LOG_ERR("Not able to store client config to file %s.", m_self->m_options->configFile());
            }
        } else{
            LOG_ERR("Not able to store client config. The received client config is broken!");
        }
    }
}

CURLcode CCClient::performCurl(const std::string& requestUrl, const std::string& requestBuffer,
                               const std::string& operation, std::string& responseBuffer)
{
    curl_global_init(CURL_GLOBAL_ALL);
    CURL *curl = curl_easy_init();

    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, "Content-Type: application/json");

    if (!m_self->m_authorization.empty()) {
        headers = curl_slist_append(headers, m_self->m_authorization.c_str());
    }

    if (!requestBuffer.empty()) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestBuffer.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, requestUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, operation.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &CCClient::onResponse);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);

    CURLcode res = curl_easy_perform(curl);

    curl_easy_cleanup(curl);
    curl_global_cleanup();

    return res;
}

void CCClient::onReport(uv_timer_t *handle)
{
    m_self->publishClientStatusReport();
}

int CCClient::onResponse(char* data, size_t size, size_t nmemb, std::string* responseBuffer)
{
    int result = 0;

    if (responseBuffer != nullptr) {
        responseBuffer->append(data, size * nmemb);
        result = size * nmemb;
    }

    return result;
}

