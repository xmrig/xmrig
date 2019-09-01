/* XMRigCC
 * Copyright 2017-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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

#ifndef __SERVICE_H__
#define __SERVICE_H__

#define CONTENT_TYPE_HTML "text/html"
#define CONTENT_TYPE_JSON "application/json"

#include <string>
#include <uv.h>
#include <microhttpd.h>
#include <map>
#include "Options.h"
#include "ClientStatus.h"
#include "ControlCommand.h"

#define TIMER_INTERVAL 10000
#define OFFLINE_TRESHOLD_IN_MS 60000
#define STATUS_UPDATE_INTERVAL 3600000

class Service
{
public:
    static bool start();
    static void release();

    static unsigned handleGET(const Options* options, const std::string& url, const std::string& clientIp, const std::string& clientId, std::string& resp);
    static unsigned handlePOST(const Options* options, const std::string& url, const std::string& clientIp, const std::string& clientId, const std::string& data, std::string& resp);

private:
    static unsigned getClientConfig(const Options* options, const std::string& clientId, std::string& resp);
    static unsigned getClientCommand(const std::string& clientId, std::string& resp);
    static unsigned getClientLog(const std::string& clientId, std::string& resp);
    static unsigned getClientStatusList(std::string& resp);
    static unsigned getClientConfigTemplates(const Options* options, std::string& resp);
    static unsigned getAdminPage(const Options* options, std::string& resp);

    static unsigned setClientStatus(const Options* options, const std::string& clientIp, const std::string& clientId, const std::string& data, std::string& resp);
    static unsigned setClientCommand(const std::string& clientId, const std::string& data, std::string& resp);
    static unsigned setClientConfig(const Options* options, const std::string &clientId, const std::string &data, std::string &resp);
    static unsigned deleteClientConfig(const Options* options, const std::string& clientId, std::string& resp);
    static unsigned resetClientStatusList(const std::string& data, std::string& resp);

    static void setClientLog(size_t maxRows, const std::string& clientId, const std::string& log);

    static std::string getClientConfigFileName(const Options *options, const std::string &clientId);

    static void onPushTimer(uv_timer_t* handle);
    static void sendServerStatusPush(uint64_t now);
    static void sendMinerOfflinePush(uint64_t now);
    static void sendMinerZeroHashratePush(uint64_t now);
    static void triggerPush(const std::string& title, const std::string& message);

private:
    static uint64_t m_currentServerTime;
    static uint64_t m_lastStatusUpdateTime;

    static std::map<std::string, ClientStatus> m_clientStatus;
    static std::map<std::string, ControlCommand> m_clientCommand;
    static std::map<std::string, std::list<std::string>> m_clientLog;

    static std::list<std::string> m_offlineNotified;
    static std::list<std::string> m_zeroHashNotified;

    static uv_mutex_t m_mutex;
    static uv_timer_t m_timer;

    static void sendViaPushover(const std::string &title, const std::string &message);

    static void sendViaTelegram(const std::string &title, const std::string &message);
};

#endif /* __SERVICE_H__ */
