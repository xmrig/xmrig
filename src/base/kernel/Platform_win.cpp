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


#include <algorithm>
#include <winsock2.h>
#include <windows.h>
#include <uv.h>
#include <limits>
#ifdef XMRIG_FEATURE_PAUSE_PROCESS
#include <tchar.h>
#include <psapi.h>
#include <shlwapi.h>
#include <locale>
#include <string>
#include <vector>
#endif


#include "base/kernel/Platform.h"
#include "version.h"


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


char *xmrig::Platform::createUserAgent()
{
    const auto osver = winOsVersion();
    constexpr const size_t max = 256;

    char *buf = new char[max]();
    int length = snprintf(buf, max, "%s/%s (Windows NT %lu.%lu", APP_NAME, APP_VERSION, osver.dwMajorVersion, osver.dwMinorVersion);

#   if defined(__x86_64__) || defined(_M_AMD64)
    length += snprintf(buf + length, max - length, "; Win64; x64) libuv/%s", uv_version_string());
#   else
    length += snprintf(buf + length, max - length, ") libuv/%s", uv_version_string());
#   endif

#   ifdef __GNUC__
    snprintf(buf + length, max - length, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif _MSC_VER
    snprintf(buf + length, max - length, " msvc/%d", MSVC_VERSION);
#   endif

    return buf;
}


#ifndef XMRIG_FEATURE_HWLOC
bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    const bool result = (SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu_id) != 0);
    Sleep(1);
    return result;
}
#endif


void xmrig::Platform::setProcessPriority(int priority)
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


void xmrig::Platform::setThreadPriority(int priority)
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


bool xmrig::Platform::isOnBatteryPower()
{
    SYSTEM_POWER_STATUS st;
    if (GetSystemPowerStatus(&st)) {
        return (st.ACLineStatus == 0);
    }
    return false;
}


uint64_t xmrig::Platform::idleTime()
{
    LASTINPUTINFO info{};
    info.cbSize = sizeof(LASTINPUTINFO);

    if (!GetLastInputInfo(&info)) {
        return std::numeric_limits<uint64_t>::max();
    }

    return static_cast<uint64_t>(GetTickCount() - info.dwTime);
}

#ifdef XMRIG_FEATURE_PAUSE_PROCESS
std::wstring s2ws(const std::string& s)
{
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    std::wstring buf;
    buf.resize(len);
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, const_cast<WCHAR*>(buf.c_str()), len);
    return buf;
}

namespace xmrig {

    uint8_t Platform::m_processListTicks = 0;
    bool Platform::m_processListState = false;

} // namespace xmrig

bool xmrig::Platform::checkProcesses(std::vector<std::string>& processList)
{
    if (m_processListTicks++ < 10)
    {
        return m_processListState;
    }
    m_processListTicks = 0;

    DWORD aProcesses[1024], cbNeeded;
    unsigned int i;
    DWORD dwProcessNameLen;

    if (EnumProcesses(aProcesses, sizeof(aProcesses), &cbNeeded))
    {
        for (i = 0; i < cbNeeded / sizeof(DWORD); i++)
        {
            if (aProcesses[i] != 0)
            {
                std::unique_ptr<WCHAR[]> wszProcessName(new WCHAR[MAX_PATH]);
                std::unique_ptr<WCHAR[]> wszSearchName(new WCHAR[MAX_PATH]);
                HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, aProcesses[i]);
                if (NULL != hProcess)
                {
                    HMODULE hMod;
                    DWORD cbNeeded;
                    if (EnumProcessModulesEx(hProcess, &hMod, sizeof(hMod), &cbNeeded, LIST_MODULES_ALL))
                    {
                        dwProcessNameLen = MAX_PATH;
                        if (QueryFullProcessImageName(hProcess, 0, wszProcessName.get(), &dwProcessNameLen))
                        {
                            for (auto const& searchName : processList)
                            {
                                std::wstring wrapper = s2ws(searchName);
                                wcsncpy_s(wszSearchName.get(), MAX_PATH / sizeof(WCHAR), wrapper.c_str(), wrapper.length());
                                if (NULL != StrStrI(wszProcessName.get(), wszSearchName.get()))
                                {
                                    CloseHandle(hProcess);
                                    m_processListState = true;
                                    return m_processListState;
                                }
                            }
                        }
                    }
                }
                CloseHandle(hProcess);
            }
        }
    }
    m_processListState = false;
    return m_processListState;
}
#endif
