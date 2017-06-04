/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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
#include <windows.h>


#include "net/Network.h"
#include "version.h"


static inline OSVERSIONINFOEX winOsVersion()
{
    typedef NTSTATUS (NTAPI *RtlGetVersionFunction)(LPOSVERSIONINFO);
    OSVERSIONINFOEX result = { sizeof(OSVERSIONINFOEX), 0, 0, 0, 0, {'\0'}, 0, 0, 0, 0, 0};

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll ) {
        RtlGetVersionFunction pRtlGetVersion = reinterpret_cast<RtlGetVersionFunction>(GetProcAddress(ntdll, "RtlGetVersion"));

        if (pRtlGetVersion) {
            pRtlGetVersion((LPOSVERSIONINFO) &result);
        }
    }

    return result;
}


char *Network::userAgent()
{
    const auto osver = winOsVersion();

    char *buf = static_cast<char*>(malloc(128));

#   ifdef __GNUC__
    snprintf(buf, 128, "%s/%s (Windows NT %lu.%lu; Win64; x64) libuv/%s gcc/%d.%d.%d", APP_NAME, APP_VERSION, osver.dwMajorVersion, osver.dwMinorVersion, uv_version_string(), __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   else
    snprintf(buf, 128, "%s/%s (Windows NT %lu.%lu; Win64; x64) libuv/%s", APP_NAME, APP_VERSION, osver.dwMajorVersion, osver.dwMinorVersion, uv_version_string());
#   endif

    return buf;
}
