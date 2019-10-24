/* XMRigCC
 * Copyright 2019-     BenDr0id    <https://github.com/BenDr0id>, <ben@graef.in>
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

#include <memory>
#include <string>
#include <map>
#include <3rdparty/cpp-httplib/httplib.h>

#include "CCServerConfig.h"
#include "ClientStatus.h"
#include "ControlCommand.h"
#include "Timer.h"

constexpr static char CONTENT_TYPE_HTML[] = "text/html";
constexpr static char CONTENT_TYPE_JSON[] = "application/json";

constexpr static int HTTP_OK = 200;
constexpr static int HTTP_BAD_REQUEST = 400;
constexpr static int HTTP_UNAUTHORIZED = 401;
constexpr static int HTTP_FORBIDDEN = 403;
constexpr static int HTTP_NOT_FOUND = 404;
constexpr static int HTTP_INTERNAL_ERROR = 500;

constexpr static int TIMER_INTERVAL = 10000;
constexpr static int OFFLINE_TRESHOLD_IN_MS = 60000;
constexpr static int STATUS_UPDATE_INTERVAL = 3600000;

class Service
{
public:
  explicit Service(const std::shared_ptr<CCServerConfig>& config);
  ~Service();

public:
  bool start();
  void stop();

  int handleGET(const httplib::Request& req, httplib::Response& res);
  int handlePOST(const httplib::Request& req, httplib::Response& res);

private:
  int getAdminPage(httplib::Response& res);

  int getClientStatusList(httplib::Response& res);
  int getClientCommand(const std::string& clientId, httplib::Response& res);
  int getClientConfigTemplates(httplib::Response& res);
  int getClientConfig(const std::string& clientId, httplib::Response& res);
  int getClientLog(const std::string& clientId, httplib::Response& res);

  int setClientStatus(const httplib::Request& req, const std::string& clientId, httplib::Response& res);
  int setClientCommand(const httplib::Request& req, const std::string& clientId, httplib::Response& res);
  int setClientConfig(const httplib::Request& req, const std::string& clientId, httplib::Response& res);
  int deleteClientConfig(const std::string& clientId);
  int resetClientStatusList();

  std::string getClientConfigFileName(const std::string& clientId);

  void setClientLog(size_t maxRows, const std::string& clientId, const std::string& log);

  void sendServerStatusPush(uint64_t now);
  void sendMinerOfflinePush(uint64_t now);
  void sendMinerZeroHashratePush(uint64_t now);
  void triggerPush(const std::string& title, const std::string& message);
  void sendViaPushover(const std::string& title, const std::string& message);
  void sendViaTelegram(const std::string& title, const std::string& message);

private:
  std::shared_ptr<CCServerConfig> m_config;
  std::shared_ptr<Timer> m_timer;

  uint64_t m_currentServerTime = 0;
  uint64_t m_lastStatusUpdateTime = 0;

  std::map<std::string, ClientStatus> m_clientStatus;
  std::map<std::string, ControlCommand> m_clientCommand;
  std::map<std::string, std::list<std::string>> m_clientLog;

  std::list<std::string> m_offlineNotified;
  std::list<std::string> m_zeroHashNotified;

  std::mutex m_mutex;
};

#endif /* __SERVICE_H__ */
