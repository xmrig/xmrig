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


#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPowerSources.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <uv.h>
#include <thread>
#include <fstream>


#include "base/kernel/Platform.h"
#include "version.h"


char *xmrig::Platform::createUserAgent()
{
    constexpr const size_t max = 256;

    char *buf = new char[max]();
    int length = snprintf(buf, max,
                          "%s/%s (Macintosh; macOS"
#                         ifdef XMRIG_ARM
                          "; arm64"
#                         else
                          "; x86_64"
#                         endif
                          ") libuv/%s", APP_NAME, APP_VERSION, uv_version_string());

#   ifdef __clang__
    length += snprintf(buf + length, max - length, " clang/%d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#   elif defined(__GNUC__)
    length += snprintf(buf + length, max - length, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   endif

    return buf;
}


bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    return true;
}


void xmrig::Platform::setProcessPriority(int)
{
}


void xmrig::Platform::setThreadPriority(int priority)
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
}


bool xmrig::Platform::isOnBatteryPower()
{
    return IOPSGetTimeRemainingEstimate() != kIOPSTimeRemainingUnlimited;
}
