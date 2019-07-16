/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <stdlib.h>
#include <uv.h>


#include "api/Api.h"
#include "App.h"
#include "backend/cpu/Cpu.h"
#include "base/io/Console.h"
#include "base/io/log/Log.h"
#include "base/kernel/Signals.h"
#include "core/config/Config.h"
#include "core/Controller.h"
#include "core/Miner.h"
#include "crypto/common/VirtualMemory.h"
#include "net/Network.h"
#include "Summary.h"
#include "version.h"


xmrig::App::App(Process *process) :
    m_console(nullptr),
    m_signals(nullptr)
{
    m_controller = new Controller(process);
    if (m_controller->init() != 0) {
        return;
    }

    if (!m_controller->config()->isBackground()) {
        m_console = new Console(this);
    }
}


xmrig::App::~App()
{
    delete m_signals;
    delete m_console;
    delete m_controller;
}


int xmrig::App::exec()
{
    if (!m_controller->isReady()) {
        return 2;
    }

    m_signals = new Signals(this);

    background();

    VirtualMemory::init(m_controller->config()->cpu().isHugePages());

    Summary::print(m_controller);

    if (m_controller->config()->isDryRun()) {
        LOG_NOTICE("OK");

        return 0;
    }

    m_controller->start();

    const int r = uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    uv_loop_close(uv_default_loop());

    return r;
}


void xmrig::App::onConsoleCommand(char command)
{
    switch (command) {
    case 'h':
    case 'H':
        m_controller->miner()->printHashrate(true);
        break;

    case 'p':
    case 'P':
        m_controller->miner()->setEnabled(false);
        break;

    case 'r':
    case 'R':
        m_controller->miner()->setEnabled(true);
        break;

    case 3:
        LOG_WARN("Ctrl+C received, exiting");
        close();
        break;

    default:
        break;
    }
}


void xmrig::App::onSignal(int signum)
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
        return;
    }

    close();
}


void xmrig::App::close()
{
    m_signals->stop();
    m_console->stop();
    m_controller->stop();

    Log::destroy();
}
