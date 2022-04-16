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

#include <uv.h>


#include "base/kernel/OS.h"
#include "3rdparty/fmt/core.h"
#include "base/kernel/Lib.h"
#include "base/tools/Cvt.h"


#ifndef UV_MAXHOSTNAMESIZE
#   define UV_MAXHOSTNAMESIZE 256
#endif


namespace xmrig {


typedef int (WSAAPI *GetHostNameW_t)(PWSTR, int); // NOLINT(modernize-use-using)


static Lib ws2_32;
static GetHostNameW_t pGetHostNameW = nullptr;


static inline OSVERSIONINFOEX winOsVersion()
{
    typedef NTSTATUS (NTAPI *RtlGetVersionFunction)(LPOSVERSIONINFO); // NOLINT(modernize-use-using)
    OSVERSIONINFOEX result = { sizeof(OSVERSIONINFOEX), 0, 0, 0, 0, {'\0'}, 0, 0, 0, 0, 0};

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll ) {
        auto pRtlGetVersion = reinterpret_cast<RtlGetVersionFunction>(GetProcAddress(ntdll, "RtlGetVersion"));

        if (pRtlGetVersion) {
            pRtlGetVersion(reinterpret_cast<LPOSVERSIONINFO>(&result));
        }
    }

    return result;
}


} // namespace xmrig


bool xmrig::OS::isOnBatteryPower()
{
    SYSTEM_POWER_STATUS st;
    if (GetSystemPowerStatus(&st)) {
        return (st.ACLineStatus == 0);
    }
    return false;
}


#ifndef XMRIG_FEATURE_HWLOC
bool xmrig::OS::setThreadAffinity(uint64_t cpu_id)
{
    const bool result = (SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu_id) != 0);
    Sleep(1);
    return result;
}
#endif


std::string xmrig::OS::name()
{
    const auto osver = winOsVersion();

    return fmt::format("Windows {}.{}", osver.dwMajorVersion, osver.dwMinorVersion);
}


xmrig::String xmrig::OS::hostname()
{
    if (pGetHostNameW) {
        WCHAR buf[UV_MAXHOSTNAMESIZE]{};

        if (pGetHostNameW(buf, UV_MAXHOSTNAMESIZE) == 0) {
            return Cvt::toUtf8(buf).c_str();
        }
    }

    char buf[UV_MAXHOSTNAMESIZE]{};

    if (gethostname(buf, sizeof(buf)) == 0) {
        return static_cast<const char *>(buf);
    }

    return {};
}


uint64_t xmrig::OS::idleTime()
{
    LASTINPUTINFO info{};
    info.cbSize = sizeof(LASTINPUTINFO);

    if (!GetLastInputInfo(&info)) {
        return std::numeric_limits<uint64_t>::max();
    }

    return static_cast<uint64_t>(GetTickCount() - info.dwTime);
}


void xmrig::OS::destroy()
{
#   ifdef XMRIG_FEATURE_COM
    CoUninitialize();
#   endif

    ws2_32.close();
}


void xmrig::OS::init()
{
#   ifdef XMRIG_FEATURE_COM
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
#   endif

    if (ws2_32.open("Ws2_32.dll")) {
        ws2_32.sym("GetHostNameW", &pGetHostNameW);
    }
}


void xmrig::OS::setProcessPriority(int priority)
{
    if (priority == -1) {
        return;
    }

    DWORD prio = IDLE_PRIORITY_CLASS;
    switch (priority)
    {
    case 1:
        prio = BELOW_NORMAL_PRIORITY_CLASS;
        break;

    case 2:
        prio = NORMAL_PRIORITY_CLASS;
        break;

    case 3:
        prio = ABOVE_NORMAL_PRIORITY_CLASS;
        break;

    case 4:
        prio = HIGH_PRIORITY_CLASS;
        break;

    case 5:
        prio = REALTIME_PRIORITY_CLASS;
        break;

    default:
        break;
    }

    SetPriorityClass(GetCurrentProcess(), prio);
}


void xmrig::OS::setThreadPriority(int priority)
{
    if (priority == -1) {
        return;
    }

    int prio = THREAD_PRIORITY_IDLE;
    switch (priority)
    {
    case 1:
        prio = THREAD_PRIORITY_BELOW_NORMAL;
        break;

    case 2:
        prio = THREAD_PRIORITY_NORMAL;
        break;

    case 3:
        prio = THREAD_PRIORITY_ABOVE_NORMAL;
        break;

    case 4:
        prio = THREAD_PRIORITY_HIGHEST;
        break;

    case 5:
        prio = THREAD_PRIORITY_TIME_CRITICAL;
        break;

    default:
        break;
    }

    SetThreadPriority(GetCurrentThread(), prio);
}
