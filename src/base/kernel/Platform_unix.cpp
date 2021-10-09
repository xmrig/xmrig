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
#ifdef XMRIG_FEATURE_PAUSE_PROCESS
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#endif


#include "base/kernel/Platform.h"
#include "version.h"


#ifdef __FreeBSD__
typedef cpuset_t cpu_set_t;
#endif


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
#endif


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


#ifdef XMRIG_FEATURE_PAUSE_PROCESS
struct ci_char_traits : public std::char_traits<char> {
    static bool eq(char c1, char c2) { return std::tolower(c1) == std::tolower(c2); }
    static bool ne(char c1, char c2) { return std::tolower(c1) != std::tolower(c2); }
    static bool lt(char c1, char c2) { return std::tolower(c1) <  std::tolower(c2); }
    static int compare(const char* s1, const char* s2, size_t n) {
        while( n-- != 0 ) {
            if( std::tolower(*s1) < std::tolower(*s2) ) return -1;
            if( std::tolower(*s1) > std::tolower(*s2) ) return 1;
            ++s1; ++s2;
        }
        return 0;
    }
    static const char* find(const char* s, int n, char a) {
        while( n-- > 0 && std::tolower(*s) != std::tolower(a) ) {
            ++s;
        }
        return s;
    }
};

bool IsNumeric(const std::string& s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

bool xmrig::Platform::checkProcesses(std::vector<std::string>& processList)
{
    const std::filesystem::path proc{"/proc/"};
    for(auto const& dirEnt: std::filesystem::directory_iterator{proc})
    {
        if (dirEnt.is_directory() && IsNumeric(dirEnt.path().stem().string()))
        {
            std::string procFile = dirEnt.path().string() + "/cmdline";
            std::ifstream myfile (procFile);
            if (myfile.is_open())
            {
                std::string cmdLine;
                std::getline(myfile, cmdLine);
                myfile.close();
                std::basic_string_view<std::string::value_type, ci_char_traits> cmdLineCI{ cmdLine.c_str() };
                if (!cmdLine.empty())
                {
                    for (auto const& processName : processList)
                    {
                        if (cmdLineCI.find(processName.c_str()) != std::string::npos)
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}
#endif


uint64_t xmrig::Platform::idleTime()
{
    return std::numeric_limits<uint64_t>::max();
}
