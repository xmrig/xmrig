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

#ifndef XMRIG_SIGNALS_H
#define XMRIG_SIGNALS_H


#include "base/tools/Object.h"


#include <csignal>
#include <cstddef>


using uv_signal_t = struct uv_signal_s;


namespace xmrig {


class ISignalListener;


class Signals
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Signals)

#   ifdef SIGUSR1
    constexpr static const size_t kSignalsCount = 4;
#   else
    constexpr static const size_t kSignalsCount = 3;
#   endif

    Signals(ISignalListener *listener);
    ~Signals();

private:
    void close(int signum);

    static void onSignal(uv_signal_t *handle, int signum);

    ISignalListener *m_listener;
    uv_signal_t *m_signals[kSignalsCount]{};
};


} /* namespace xmrig */


#endif /* XMRIG_SIGNALS_H */
