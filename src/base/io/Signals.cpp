/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/kernel/interfaces/ISignalListener.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/io/Signals.h"
#include "base/tools/Handle.h"


#ifdef SIGUSR1
static const int signums[xmrig::Signals::kSignalsCount] = { SIGHUP, SIGINT, SIGTERM, SIGUSR1 };
#else
static const int signums[xmrig::Signals::kSignalsCount] = { SIGHUP, SIGINT, SIGTERM };
#endif


xmrig::Signals::Signals(ISignalListener *listener)
    : m_listener(listener)
{
#   ifndef XMRIG_OS_WIN
    signal(SIGPIPE, SIG_IGN);
#   endif

    for (size_t i = 0; i < kSignalsCount; ++i) {
        auto signal  = new uv_signal_t;
        signal->data = this;

        m_signals[i] = signal;

        uv_signal_init(uv_default_loop(), signal);
        uv_signal_start(signal, Signals::onSignal, signums[i]);
    }
}


xmrig::Signals::~Signals()
{
    for (auto signal : m_signals) {
        Handle::close(signal);
    }
}


void xmrig::Signals::onSignal(uv_signal_t *handle, int signum)
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

#   ifdef SIGUSR1
    case SIGUSR1:
        LOG_V5("%s " WHITE_BOLD("SIGUSR1 received"), Tags::signal());
        break;
#   endif

    default:
        break;
    }

    static_cast<Signals *>(handle->data)->m_listener->onSignal(signum);
}
