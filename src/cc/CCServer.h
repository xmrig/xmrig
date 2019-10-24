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

#ifndef __CC_SERVER_H__
#define __CC_SERVER_H__

#include <memory>
#include <cxxopts/cxxopts.hpp>

#include "base/kernel/interfaces/IConsoleListener.h"
#include "base/kernel/interfaces/ISignalListener.h"
#include "base/kernel/Signals.h"
#include "base/io/Console.h"

#include "CCServerConfig.h"
#include "Httpd.h"

class CCServer : public xmrig::IConsoleListener, public xmrig::ISignalListener
{
public:
  CCServer(cxxopts::ParseResult& parseResult);
  ~CCServer();

  int start();

protected:
  void onConsoleCommand(char command) override;
  void onSignal(int signum) override;

private:
  void stop();
  void moveToBackground();

  std::shared_ptr<xmrig::Console> m_console;
  std::shared_ptr<xmrig::Signals> m_signals;
  std::shared_ptr<CCServerConfig> m_config;
  std::shared_ptr<Httpd> m_httpd;

  void startUvLoopThread() const;
};


#endif /* __CC_SERVER_H__ */
