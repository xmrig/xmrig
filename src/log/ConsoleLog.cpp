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


#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#ifdef WIN32
#   include <winsock2.h>
#   include <malloc.h>
#   include "3rdparty/winansi.h"
#endif

#include "log/ConsoleLog.h"
#include "log/Log.h"


ConsoleLog::ConsoleLog(bool colors) :
    m_colors(colors)
{
}


void ConsoleLog::message(int level, const char* fmt, va_list args)
{
    time_t now = time(nullptr);
    tm stime;

#   ifdef _WIN32
    localtime_s(&stime, &now);
#   else
    localtime_r(&now, &stime);
#   endif

    const char* color = nullptr;
    if (m_colors) {
        switch (level) {
        case Log::ERR:
            color = Log::kCL_RED;
            break;

        case Log::WARNING:
            color = Log::kCL_YELLOW;
            break;

        case Log::NOTICE:
            color = Log::kCL_WHITE;
            break;

        case Log::DEBUG:
            color = Log::kCL_GRAY;
            break;

        default:
            color = "";
            break;
        }
    }

    const size_t len = 64 + strlen(fmt) + 2;
    char *buf = static_cast<char *>(alloca(len));

    sprintf(buf, "[%d-%02d-%02d %02d:%02d:%02d]%s %s%s\n",
            stime.tm_year + 1900,
            stime.tm_mon + 1,
            stime.tm_mday,
            stime.tm_hour,
            stime.tm_min,
            stime.tm_sec,
            m_colors ? color : "",
            fmt,
            m_colors ? Log::kCL_N : ""
        );

    vfprintf(stdout, buf, args);
    fflush(stdout);
}


void ConsoleLog::text(const char* fmt, va_list args)
{
    const int len = 64 + strlen(fmt) + 2;
    char *buf = static_cast<char *>(alloca(len));

    sprintf(buf, "%s%s\n", fmt, m_colors ? Log::kCL_N : "");

    vfprintf(stdout, buf, args);
    fflush(stdout);
}
