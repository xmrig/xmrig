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

#include <rapidjson/document.h>

#include "base/io/json/JsonChain.h"
#include "base/io/log/Log.h"
#include "CCServerConfig.h"


template<class T>
T getParseResult(const cxxopts::ParseResult &parseResult, const std::string &propery, T defaultValue)
{
  if (parseResult.count(propery) > 0)
  {
    return parseResult[propery].as<T>();
  }
  else
  {
    return defaultValue;
  }
}

CCServerConfig::CCServerConfig(cxxopts::ParseResult& parseResult)
{
  try
  {
    xmrig::JsonChain chain;

    auto m_configFile = getParseResult(parseResult, "config", std::string());
    if (!m_configFile.empty())
    {
     chain.addFile(m_configFile.c_str());
    }
    else
    {
     chain.addFile("config_cc.json");
    }

    read(chain);

    m_bindIp = getParseResult(parseResult, "bind", m_bindIp);
    m_port = getParseResult(parseResult, "port", m_port);
    m_adminUser = getParseResult(parseResult, "user", m_adminUser);
    m_adminPass = getParseResult(parseResult, "pass", m_adminPass);
    m_token = getParseResult(parseResult, "token", m_token);
    m_useTLS = getParseResult(parseResult, "tls", m_useTLS);
    m_keyFile = getParseResult(parseResult, "key-file", m_keyFile);
    m_certFile = getParseResult(parseResult, "cert-file", m_certFile);

    m_colors = !getParseResult(parseResult, "no-colors", !m_colors);
    m_background = getParseResult(parseResult, "background", m_background);
    m_syslog = getParseResult(parseResult, "syslog", m_syslog);

    m_clientLogHistory = getParseResult(parseResult, "client-log-lines-history", m_clientLogHistory);
    m_customDashboard = getParseResult(parseResult, "custom-dashboard", m_customDashboard);
    m_clientConfigFolder = getParseResult(parseResult, "client-config-folder", m_clientConfigFolder);
    m_logFile = getParseResult(parseResult, "log-file", m_logFile);

    m_pushoverApiToken = getParseResult(parseResult, "pushover-api-token", m_pushoverApiToken);
    m_pushoverUserKey = getParseResult(parseResult, "pushover-user-key", m_pushoverUserKey);
    m_telegramBotToken = getParseResult(parseResult, "telegram-bot-token", m_telegramBotToken);
    m_telegramChatId = getParseResult(parseResult, "telegram-chat-id", m_telegramChatId);
    m_pushOfflineMiners = getParseResult(parseResult, "push-miner-offline-info", m_pushOfflineMiners);
    m_pushZeroHashrateMiners = getParseResult(parseResult, "push-miner-zero-hash-info", m_pushZeroHashrateMiners);
    m_pushPeriodicStatus = getParseResult(parseResult, "push-periodic-mining-status", m_pushPeriodicStatus);
  }
  catch (const cxxopts::OptionException& e)
  {
    LOG_WARN("Failed to parse params. Error: %s", e.what());
  }
}

bool CCServerConfig::read(const xmrig::IJsonReader& reader)
{
  if (reader.isEmpty())
  {
    return false;
  }

  m_port = reader.getInt("port", m_port);
  m_bindIp = reader.getString("bind-ip", m_bindIp.c_str());
  m_adminUser = reader.getString("user", m_adminUser.c_str());
  m_adminPass = reader.getString("pass", m_adminPass.c_str());
  m_token = reader.getString("access-token", m_token.c_str());
  m_useTLS = reader.getBool("use-tls", m_useTLS);
  m_keyFile = reader.getString("key-file", m_keyFile.c_str());
  m_certFile = reader.getString("cert-file", m_certFile.c_str());

  m_colors = reader.getBool("colors", m_colors);
  m_background = reader.getBool("background", m_background);
  m_syslog = reader.getBool("syslog", m_syslog);

  m_clientLogHistory = reader.getInt("client-log-lines-history", m_clientLogHistory);
  m_customDashboard = reader.getString("custom-dashboard", m_customDashboard.c_str());
  m_clientConfigFolder = reader.getString("client-config-folder", m_clientConfigFolder.c_str());
  m_logFile = reader.getString("log-file", m_logFile.c_str());

  m_pushoverApiToken = reader.getString("pushover-api-token", m_pushoverApiToken.c_str());
  m_pushoverUserKey = reader.getString("pushover-user-key", m_pushoverUserKey.c_str());
  m_telegramBotToken = reader.getString("telegram-bot-token", m_telegramBotToken.c_str());
  m_telegramChatId = reader.getString("telegram-chat-id", m_telegramChatId.c_str());
  m_pushOfflineMiners = reader.getBool("push-miner-offline-info", m_pushOfflineMiners);
  m_pushZeroHashrateMiners = reader.getBool("push-miner-zero-hash-info", m_pushZeroHashrateMiners);
  m_pushPeriodicStatus = reader.getBool("push-periodic-mining-status", m_pushPeriodicStatus);

  return true;
}
