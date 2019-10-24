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

#include <iostream>
#include <cxxopts/cxxopts.hpp>
#include "CCServer.h"
#include "version.h"

int main(int argc, char** argv)
{
  int ret = 0;

  try
  {
    cxxopts::Options options(argv[0] ,APP_NAME "Server " APP_VERSION);
    options.positional_help("[optional args]");
    options.show_positional_help();

    options.add_options()
        ("b, bind", "The CC Server bind ip", cxxopts::value<std::string>()->default_value("0.0.0.0"))
        ("p, port", "The CC Server port", cxxopts::value<int>(), "N")
        ("U, user", "The CC Server admin user", cxxopts::value<std::string>())
        ("P, pass", "The CC Server admin pass", cxxopts::value<std::string>())
        ("T, token", "The CC Server access token for the CC Client", cxxopts::value<std::string>())

        ("t, tls", "Enable SSL/TLS support", cxxopts::value<bool>()->default_value("false"))
        ("K, key-file", "The private key file to use when TLS is ON", cxxopts::value<std::string>()->default_value("server.key"), "FILE")
        ("C, cert-file", "The cert file to use when TLS is ON", cxxopts::value<std::string>()->default_value("server.pem"), "FILE")

        ("B, background", "Run the Server in the background", cxxopts::value<bool>()->default_value("false"))
        ("S, syslog", "Log to the syslog", cxxopts::value<bool>()->default_value("false"))
        ("no-colors", "Disable colored output", cxxopts::value<bool>()->default_value("false"))

        ("pushover-user-key", "The user key for pushover notifications", cxxopts::value<std::string>())
        ("pushover-api-token", "The api token/keytoken of the application for pushover notification", cxxopts::value<std::string>())
        ("telegram-bot-token", "The bot token for telegram notifications", cxxopts::value<std::string>())
        ("telegram-chat-id", "The chat-id for telegram notifications", cxxopts::value<std::string>())
        ("push-miner-offline-info", "Push notification for offline miners and recovery", cxxopts::value<bool>()->default_value("true"))
        ("push-miner-zero-hash-info", "Push notification when miner reports 0 hashrate and recovers", cxxopts::value<bool>()->default_value("true"))
        ("push-periodic-mining-status", "Push every hour a status notification", cxxopts::value<bool>()->default_value("true"))

        ("custom-dashboard", "The custom dashboard to use", cxxopts::value<std::string>()->default_value("index.html"), "FILE")
        ("client-config-folder", "The CC Server access token for the CC Client", cxxopts::value<std::string>(), "FOLDER")
        ("log-file", "The log file to write", cxxopts::value<std::string>(), "FILE")
        ("client-log-lines-history", "Maximum lines of log history kept per miner", cxxopts::value<int>()->default_value("100"), "N")

        ("c, config", "The JSON-format configuration file to use", cxxopts::value<std::string>(), "FILE")
        ("h, help", "Print this help")
    ;

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
      std::cout << options.help({""}) << std::endl;
    }
    else
    {
      CCServer server(result);
      ret = server.start();
    }
  }
  catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    ret = EINVAL;
  }

  return ret;
}
