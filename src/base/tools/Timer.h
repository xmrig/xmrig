/* XMRig
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

#ifndef XMRIG_TIMER_H
#define XMRIG_TIMER_H


using uv_timer_t = struct uv_timer_s;


#include "base/tools/Object.h"


#include <cstdint>


namespace xmrig {


class ITimerListener;


class Timer
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Timer);

    Timer(ITimerListener *listener);
    Timer(ITimerListener *listener, uint64_t timeout, uint64_t repeat);
    ~Timer();

    inline int id() const { return m_id; }

    uint64_t repeat() const;
    void setRepeat(uint64_t repeat);
    void singleShot(uint64_t timeout, int id = 0);
    void start(uint64_t timeout, uint64_t repeat);
    void stop();

private:
    void init();

    static void onTimer(uv_timer_t *handle);

    int m_id                    = 0;
    ITimerListener *m_listener;
    uv_timer_t *m_timer         = nullptr;
};


} /* namespace xmrig */


#endif /* XMRIG_TIMER_H */
