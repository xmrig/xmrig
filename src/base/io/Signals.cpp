/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
 *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
 */

#include "base/io/Signals.h"
#include "base/io/log/Log.h"
#include "base/kernel/interfaces/ISignalListener.h"
#include "base/kernel/Process.h"
#include "base/tools/Handle.h"


#ifdef XMRIG_FEATURE_EVENTS
#   include "base/kernel/Events.h"
#   include "base/kernel/events/SignalEvent.h"
#endif


#include <csignal>


namespace xmrig {


#ifdef SIGUSR1
constexpr static const size_t kSignalsCount = 4;
constexpr static const int signums[kSignalsCount] = { SIGHUP, SIGINT, SIGTERM, SIGUSR1 };
#else
constexpr static const size_t kSignalsCount = 3;
constexpr static const int signums[kSignalsCount] = { SIGHUP, SIGINT, SIGTERM };
#endif


#ifdef XMRIG_FEATURE_EVENTS
static const char *signame(int signum)
{
    switch (signum) {
    case SIGHUP:
        return "SIGHUP";

    case SIGINT:
        return "SIGINT";

    case SIGTERM:
        return "SIGTERM";

    default:
        break;
    }

    return nullptr;
}
#endif


class Signals::Private : public ISignalListener
{
public:
    XMRIG_DISABLE_COPY_MOVE(Private)

    Private();
    ~Private() override;

    ISignalListener *listener = nullptr;

protected:
    void onSignal(int signum) override;

private:
    static void onSignal(uv_signal_t *handle, int signum);



    uv_signal_t *signals[kSignalsCount]{};
};


} // namespace xmrig


xmrig::Signals::Signals() :
    d(std::make_shared<Private>())
{
    d->listener = d.get();
}


xmrig::Signals::Signals(ISignalListener *listener) :
    d(std::make_shared<Private>())
{
    d->listener = listener;
}


const char *xmrig::Signals::tag()
{
    static const char *tag = YELLOW_BG_BOLD(WHITE_BOLD_S " signal  ");

    return tag;
}


xmrig::Signals::Private::Private()
{
#   ifndef XMRIG_OS_WIN
    signal(SIGPIPE, SIG_IGN);
#   endif

    for (size_t i = 0; i < kSignalsCount; ++i) {
        auto signal  = new uv_signal_t;
        signal->data = this;

        signals[i] = signal;

        uv_signal_init(uv_default_loop(), signal);
        uv_signal_start(signal, onSignal, signums[i]);
    }
}


xmrig::Signals::Private::~Private()
{
    for (auto signal : signals) {
        Handle::close(signal);
    }
}


void xmrig::Signals::Private::onSignal(int signum)
{
#   ifdef XMRIG_FEATURE_EVENTS
    Process::events().send<SignalEvent>(signum);

    switch (signum)
    {
    case SIGHUP:
    case SIGTERM:
    case SIGINT:
        LOG_WARN("%s " YELLOW_BOLD("%s ") YELLOW("received, exiting"), tag(), signame(signum));
        Process::exit(128 + signum);
        break;

    default:
        break;
    }
#   endif
}


void xmrig::Signals::Private::onSignal(uv_signal_t *handle, int signum)
{
    static_cast<Private *>(handle->data)->listener->onSignal(signum);
}
