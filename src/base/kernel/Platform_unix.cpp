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

#ifdef XMRIG_OS_FREEBSD
#   include <sys/types.h>
#   include <sys/param.h>
#   ifndef __DragonFly__
#       include <sys/cpuset.h>
#   endif
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

#include "base/kernel/Platform.h"
#include "version.h"


char *xmrig::Platform::createUserAgent()
{
    constexpr const size_t max = 256;

    char *buf = new char[max]();
    int length = snprintf(buf, max, "%s/%s (Linux ", APP_NAME, APP_VERSION);

#   if defined(__x86_64__)
    length += snprintf(buf + length, max - length, "x86_64) libuv/%s", uv_version_string());
#   elif defined(__aarch64__)
    length += snprintf(buf + length, max - length, "aarch64) libuv/%s", uv_version_string());
#   elif defined(__arm__)
    length += snprintf(buf + length, max - length, "arm) libuv/%s", uv_version_string());
#   else
    length += snprintf(buf + length, max - length, "i686) libuv/%s", uv_version_string());
#   endif

#   ifdef __clang__
    length += snprintf(buf + length, max - length, " clang/%d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#   elif defined(__GNUC__)
    length += snprintf(buf + length, max - length, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   endif

    return buf;
}


#ifndef XMRIG_FEATURE_HWLOC
#ifdef __DragonFly__

bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    return true;
}

#else

#ifdef XMRIG_OS_FREEBSD
typedef cpuset_t cpu_set_t;
#endif

bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
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

#endif // __DragonFly__
#endif // XMRIG_FEATURE_HWLOC


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


bool xmrig::Platform::isOnBatteryPower()
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

#if defined(XMRIG_OS_LINUX) && defined(XMRIG_X11_FOUND)

#include <dlfcn.h>
#include <X11/extensions/scrnsaver.h>

template<typename T>
void bind_symbol(T& var, void *library, const char* name) {
    var = reinterpret_cast<T>(::dlsym(library, name));
}

uint64_t xmrig::Platform::idleTime()
{
    // libX11
    static Display* (*XOpenDisplay)(const char *name) = {};

    // libXss
    static XScreenSaverInfo* (*XScreenSaverAllocInfo)() = {};
    static Status (*XScreenSaverQueryInfo)(Display *dpy, Drawable drawable,
    XScreenSaverInfo *saver_info) = {};

    static bool initialized = false;
    static Display *dpy = {}; 
    static XScreenSaverInfo *info = {};

    if(!initialized) {
        static void *libx11 = dlopen("/usr/lib/libX11.so", RTLD_LAZY);
        static void *libxss = dlopen("/usr/lib/libXss.so", RTLD_LAZY);

        if(!libxss || !libx11) {
            initialized = true;
            return std::numeric_limits<uint64_t>::max();
        }

        bind_symbol(XOpenDisplay, libx11, "XOpenDisplay");
        bind_symbol(XScreenSaverAllocInfo, libxss, "XScreenSaverAllocInfo");
        bind_symbol(XScreenSaverQueryInfo, libxss, "XScreenSaverQueryInfo");

        dpy = XOpenDisplay(nullptr);
        info = XScreenSaverAllocInfo();

        initialized = true;
    }

    if (!dpy) {
        return std::numeric_limits<uint64_t>::max();
    }
 
    XScreenSaverQueryInfo(dpy, DefaultRootWindow(dpy), info);

    return info->idle;
}
#else

uint64_t xmrig::Platform::idleTime() { 
    return std::numeric_limits<uint64_t>::max();
}

#endif