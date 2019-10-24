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

#ifndef XMRIG_CC_SERVER_CONFIG_H
#define XMRIG_CC_SERVER_CONFIG_H

#include <string>
#include <rapidjson/fwd.h>
#include <cxxopts/cxxopts.hpp>

#include "base/kernel/interfaces/IJsonReader.h"

class CCServerConfig
{
public:
  explicit CCServerConfig(cxxopts::ParseResult& parseResult);

public:
  bool read(const xmrig::IJsonReader &reader);

  inline bool colors() const                      { return m_colors; }
  inline bool background() const                  { return m_background; }
  inline bool syslog() const                      { return m_syslog; }
  inline bool useTLS() const                      { return m_useTLS; }
  inline bool usePushover() const                 { return !m_pushoverUserKey.empty() && !m_pushoverApiToken.empty(); }
  inline bool useTelegram() const                 { return !m_telegramBotToken.empty() && !m_telegramChatId.empty(); }
  inline bool pushOfflineMiners() const           { return m_pushOfflineMiners; }
  inline bool pushZeroHashrateMiners() const      { return m_pushZeroHashrateMiners; }
  inline bool pushPeriodicStatus() const          { return m_pushPeriodicStatus; }

  inline std::string bindIp() const               { return m_bindIp; }
  inline std::string adminUser() const            { return m_adminUser; }
  inline std::string adminPass() const            { return m_adminPass; }
  inline std::string token() const                { return m_token; }
  inline std::string customDashboard() const      { return m_customDashboard; }
  inline std::string clientConfigFolder() const   { return m_clientConfigFolder; }
  inline std::string logFile() const              { return m_logFile; }
  inline std::string keyFile() const              { return m_keyFile; }
  inline std::string certFile() const             { return m_certFile; }
  inline std::string pushoverApiToken() const     { return m_pushoverApiToken; }
  inline std::string pushoverUserKey() const      { return m_pushoverUserKey; }
  inline std::string telegramBotToken() const     { return m_telegramBotToken; }
  inline std::string telegramChatId() const       { return m_telegramChatId; }

  inline int port() const                         { return m_port; }
  inline int clientLogHistory() const             { return m_clientLogHistory; }

  inline bool isValid() const                     { return !m_bindIp.empty() &&
                                                            m_port > 0 && m_port < 65535; }

private:
  bool m_colors = true;
  bool m_background = false;
  bool m_syslog = false;
  bool m_useTLS = false;
  bool m_pushOfflineMiners = true;
  bool m_pushZeroHashrateMiners = true;
  bool m_pushPeriodicStatus = true;

  int m_clientLogHistory = 100;
  int m_port = 3344;

  std::string m_bindIp = "0.0.0.0";
  std::string m_adminUser = "";
  std::string m_adminPass = "";
  std::string m_token = "";

  std::string m_customDashboard = "index.html";
  std::string m_clientConfigFolder;
  std::string m_logFile;

  std::string m_keyFile = "server.key";
  std::string m_certFile = "server.pem";

  std::string m_pushoverApiToken;
  std::string m_pushoverUserKey;
  std::string m_telegramBotToken;
  std::string m_telegramChatId;
};

#endif /* XMRIG_CC_SERVER_CONFIG_H */
