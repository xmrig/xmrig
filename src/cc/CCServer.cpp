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

#include <uv.h>
#include <memory>

#include "base/io/log/backends/ConsoleLog.h"
#include "base/io/log/backends/FileLog.h"
#include "base/io/json/JsonChain.h"
#include "base/io/log/Log.h"

#ifdef HAVE_SYSLOG_H
#include "base/io/log/backends/SysLog.h"
#endif

#include "CCServerConfig.h"
#include "Httpd.h"
#include "CCServer.h"
#include "Summary.h"

CCServer::CCServer(cxxopts::ParseResult& parseResult)
{
  m_config = std::make_shared<CCServerConfig>(parseResult);

  if (!m_config->background())
  {
    xmrig::Log::colors = m_config->colors();
    xmrig::Log::add(new xmrig::ConsoleLog());
    m_console = std::make_shared<xmrig::Console>(this);
  }

  if (!m_config->logFile().empty())
  {
    xmrig::Log::add(new xmrig::FileLog(m_config->logFile().c_str()));
  }

#ifdef HAVE_SYSLOG_H
  if (m_config->syslog())
  {
    xmrig::Log::add(new xmrig::SysLog());
  }
#endif
}

CCServer::~CCServer()
{
  m_signals.reset();
  m_console.reset();
  m_httpd.reset();
  m_config.reset();
}

int CCServer::start()
{
  if (!m_config->isValid())
  {
    LOG_ERR("Invalid config provided");
    return EINVAL;
  }

  m_signals = std::make_shared<xmrig::Signals>(this);

  if (m_config->background())
  {
    moveToBackground();
  }

  Summary::print(m_config);

  startUvLoopThread();

  m_httpd = std::make_shared<Httpd>(m_config);
  int retVal = m_httpd->start();
  if (retVal > 0)
  {
    LOG_ERR("Failed to bind %sServer to %s:%d", m_config->useTLS() ? "TLS " : "", m_config->bindIp().c_str(),
            m_config->port());
  }
  else if (retVal < 0)
  {
    LOG_ERR("Invalid config. %s", m_config->useTLS() ? "Check bindIp, port and the certificate/key file"
                                                     : "Check bindIp and port");
  }
  else
  {
    LOG_INFO("Server stopped. Exit.");
  }

  return retVal;
}

void CCServer::startUvLoopThread() const
{
  std::thread([]()
              {
                uv_run(uv_default_loop(), UV_RUN_DEFAULT);
                uv_loop_close(uv_default_loop());
              }).detach();
}

void CCServer::onConsoleCommand(char command)
{
  switch (command)
  {
    case 'q':
    case 'Q':
      stop();
      break;

    case 3:
      LOG_WARN("Ctrl+C received, exiting");
      stop();
      break;

    default:
      break;
  }
}

void CCServer::stop()
{
  m_httpd->stop();

  uv_stop(uv_default_loop());
}

void CCServer::onSignal(int signum)
{
  switch (signum)
  {
    case SIGHUP:
      LOG_WARN("SIGHUP received, exiting");
      break;

    case SIGTERM:
      LOG_WARN("SIGTERM received, exiting");
      break;

    case SIGINT:
      LOG_WARN("SIGINT received, exiting");
      break;

    default:
      break;
  }

  stop();
}

void CCServer::moveToBackground()
{
#ifdef WIN32
  HWND hcon = GetConsoleWindow();
  if (hcon) {
      ShowWindow(hcon, SW_HIDE);
  } else {
      HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
      CloseHandle(h);
      FreeConsole();
  }
#else
  int i = fork();
  if (i < 0)
  {
    exit(1);
  }

  if (i > 0)
  {
    exit(0);
  }

  i = setsid();

  if (i < 0)
  {
    LOG_ERR("setsid() failed (errno = %d)", errno);
  }

  i = chdir("/");
  if (i < 0)
  {
    LOG_ERR("chdir() failed (errno = %d)", errno);
  }
#endif
}
