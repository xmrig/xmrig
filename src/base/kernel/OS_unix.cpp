/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifdef __FreeBSD__
#   include <sys/types.h>
#   include <sys/param.h>
#   include <sys/cpuset.h>
#   include <pthread_np.h>
#endif


#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#include <uv.h>
#include <thread>
#include <fstream>
#include <limits>


#include "base/kernel/OS.h"


#ifdef __FreeBSD__
typedef cpuset_t cpu_set_t;
#endif


bool xmrig::OS::isOnBatteryPower()
{
    for (int i = 0; i <= 1; ++i) {
        char buf[64];
        snprintf(buf, 64, "/sys/class/power_supply/BAT%d/status", i);
        std::ifstream f(buf);
        if (f.is_open()) {
            std::string status;
            f >> status;
            return (status == "Discharging");
        }
    }
    return false;
}


#ifndef XMRIG_FEATURE_HWLOC
bool xmrig::OS::setThreadAffinity(uint64_t cpu_id)
{
    cpu_set_t mn;
    CPU_ZERO(&mn);
    CPU_SET(cpu_id, &mn);

#   ifndef __ANDROID__
    const bool result = (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mn) == 0);
#   else
    const bool result = (sched_setaffinity(gettid(), sizeof(cpu_set_t), &mn) == 0);
#   endif

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return result;
}
#endif


std::string xmrig::OS::name()
{
#   ifdef __FreeBSD__
    return "FreeBSD";
#   else
    return "Linux";
#   endif
}


uint64_t xmrig::OS::idleTime()
{
    return std::numeric_limits<uint64_t>::max();
}


void xmrig::OS::destroy()
{
}


void xmrig::OS::init()
{
}


void xmrig::OS::setProcessPriority(int)
{
}


void xmrig::OS::setThreadPriority(int priority)
{
    if (priority == -1) {
        return;
    }

    int prio = 19;
    switch (priority)
    {
    case 1:
        prio = 5;
        break;

    case 2:
        prio = 0;
        break;

    case 3:
        prio = -5;
        break;

    case 4:
        prio = -10;
        break;

    case 5:
        prio = -15;
        break;

    default:
        break;
    }

    setpriority(PRIO_PROCESS, 0, prio);

#   ifdef SCHED_IDLE
    if (priority == 0) {
        sched_param param;
        param.sched_priority = 0;

        if (sched_setscheduler(0, SCHED_IDLE, &param) != 0) {
            sched_setscheduler(0, SCHED_BATCH, &param);
        }
    }
#   endif
}
