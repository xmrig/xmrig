/* XMRigCC
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

#include <uv.h>

#include "CCServer.h"
#include "Service.h"
#include "Httpd.h"
#include "Console.h"
#include "log/ConsoleLog.h"
#include "log/FileLog.h"
#include "log/Log.h"
#include "Options.h"
#include "Summary.h"

#if _WIN32
    #include <winsock2.h>
    #include <windows.h>
#else
#   include "unistd.h"
#endif

#ifdef HAVE_SYSLOG_H
#   include "log/SysLog.h"
#endif

CCServer *CCServer::m_self = nullptr;


CCServer::CCServer(int argc, char** argv) :
    m_console(nullptr),
    m_httpd(nullptr),
    m_options(nullptr)
{
    m_self = this;

    Log::init();

    m_options = Options::parse(argc, argv);
    if (!m_options) {
        return;
    }
    
    if (!m_options->background()) {
        Log::add(new ConsoleLog(m_options->colors()));
        m_console = new Console(this);
    }

    if (m_options->logFile()) {
        Log::add(new FileLog(m_options->logFile()));
    }

#   ifdef HAVE_SYSLOG_H
    if (m_options->syslog()) {
        Log::add(new SysLog());
    }
#   endif

    uv_signal_init(uv_default_loop(), &m_signal);
}

CCServer::~CCServer()
{
    uv_tty_reset_mode();

    delete m_httpd;
}

int CCServer::start()
{
    if (!m_options) {
        return EINVAL;
    }

    uv_signal_start(&m_signal, CCServer::onSignal, SIGHUP);
    uv_signal_start(&m_signal, CCServer::onSignal, SIGTERM);
    uv_signal_start(&m_signal, CCServer::onSignal, SIGINT);

    if (m_options->background()) {
        moveToBackground();
    }

    Summary::print();

    Service::start();

    m_httpd = new Httpd(m_options);
    if (!m_httpd->start()) {
        return EINVAL;
    }

    const int r = uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    uv_loop_close(uv_default_loop());

    Options::release();

    return r;
}

void CCServer::onConsoleCommand(char command)
{
    switch (command) {
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
    uv_stop(uv_default_loop());
}

void CCServer::onSignal(uv_signal_t* handle, int signum)
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

    uv_signal_stop(handle);
    m_self->stop();
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
    if (i < 0) {
        exit(1);
    }

    if (i > 0) {
        exit(0);
    }

    i = setsid();

    if (i < 0) {
        LOG_ERR("setsid() failed (errno = %d)", errno);
    }

    i = chdir("/");
    if (i < 0) {
        LOG_ERR("chdir() failed (errno = %d)", errno);
    }
#endif
}
