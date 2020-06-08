/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <cstdlib>
#include <uv.h>


#include "App.h"
#include "backend/cpu/Cpu.h"
#include "base/io/Console.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/io/Signals.h"
#include "base/kernel/Platform.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "core/Miner.h"
#include "net/Network.h"
#include "Summary.h"
#include "version.h"


xmrig::App::App(Process *process)
{
    m_controller = new Controller(process);
}


xmrig::App::~App()
{
    Cpu::release();

    delete m_signals;
    delete m_console;
    delete m_controller;
}


int xmrig::App::exec()
{
    if (!m_controller->isReady()) {
        LOG_EMERG("no valid configuration found, try https://xmrig.com/wizard");

        return 2;
    }

    m_signals = new Signals(this);

    int rc = 0;
    if (background(rc)) {
        return rc;
    }

    rc = m_controller->init();
    if (rc != 0) {
        return rc;
    }

    if (!m_controller->isBackground()) {
        m_console = new Console(this);
    }

    Summary::print(m_controller);

    if (m_controller->config()->isDryRun()) {
        LOG_NOTICE("%s " WHITE_BOLD("OK"), Tags::config());

        return 0;
    }

#   ifdef XMRIG_FEATURE_BENCHMARK
    m_controller->pre_start();
    m_controller->config()->benchmark().set_controller(m_controller);

    if (m_controller->config()->benchmark().isNewBenchRun() || m_controller->config()->isRebenchAlgo()) {
        m_controller->config()->benchmark().start();
    } else {
        m_controller->start();
    }
#   else
    m_controller->start();
#   endif

    rc = uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    uv_loop_close(uv_default_loop());

    return rc;
}


void xmrig::App::onConsoleCommand(char command)
{
    if (command == 3) {
        LOG_WARN("%s " YELLOW("Ctrl+C received, exiting"), Tags::signal());
        close();
    }
    else {
        m_controller->miner()->execCommand(command);
    }
}


void xmrig::App::onSignal(int signum)
{
    switch (signum)
    {
    case SIGHUP:
        LOG_WARN("%s " YELLOW("SIGHUP received, exiting"), Tags::signal());
        break;

    case SIGTERM:
        LOG_WARN("%s " YELLOW("SIGTERM received, exiting"), Tags::signal());
        break;

    case SIGINT:
        LOG_WARN("%s " YELLOW("SIGINT received, exiting"), Tags::signal());
        break;

    default:
        return;
    }

    close();
}


void xmrig::App::close()
{
    m_signals->stop();

    if (m_console) {
        m_console->stop();
    }

    m_controller->stop();

    Log::destroy();
}
