/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
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


#include "App.h"
#include "Console.h"
#include "Cpu.h"
#include "crypto/CryptoNight.h"
#include "Mem.h"
#include "net/Network.h"
#include "Options.h"
#include "Summary.h"
#include "version.h"
#include "workers/Workers.h"


App *App::m_self = nullptr;



App::App(int argc, char **argv) :
    m_network(nullptr),
    m_options(nullptr)
{
    m_self = this;

    Console::init();
    Cpu::init();

    m_options = Options::parse(argc, argv);
    m_network = new Network(m_options);

    uv_signal_init(uv_default_loop(), &m_signal);
}


App::~App()
{
}


int App::exec()
{
    if (!m_options->isReady()) {
        return 0;
    }

    if (!CryptoNight::init(m_options->algo(), m_options->algoVariant())) {
        LOG_ERR("\"%s\" hash self-test failed.", m_options->algoName());
        return 1;
    }

    uv_signal_start(&m_signal, App::onSignal, SIGHUP);
    uv_signal_start(&m_signal, App::onSignal, SIGTERM);
    uv_signal_start(&m_signal, App::onSignal, SIGINT);

    background();

    Mem::allocate(m_options->algo(), m_options->threads(), m_options->doubleHash());
    Summary::print();

    Workers::start(m_options->threads(), m_options->affinity(), m_options->nicehash());

    m_network->connect();

    const int r = uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    uv_loop_close(uv_default_loop());

    free(m_network);
    free(m_options);

    return r;
}


void App::close()
{
    uv_stop(uv_default_loop());
}


void App::onSignal(uv_signal_t *handle, int signum)
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

    m_self->close();
}
