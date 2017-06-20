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

#include "log/Log.h"


Log *Log::m_self = nullptr;


void Log::init(bool colors, bool background)
{
    if (!m_self) {
        m_self = new Log(colors, background);
    }
}


void Log::message(Log::Level level, const char* fmt, ...)
{
    time_t now = time(nullptr);
    tm stime;

#   ifdef _WIN32
    localtime_s(&stime, &now);
#   else
    localtime_r(&now, &stime);
#   endif

    va_list ap;
    va_start(ap, fmt);

    const char* color = nullptr;
    if (m_colors) {
        switch (level) {
        case ERR:
            color = kCL_RED;
            break;

        case WARNING:
            color = kCL_YELLOW;
            break;

        case NOTICE:
            color = kCL_WHITE;
            break;

        case DEBUG:
            color = kCL_GRAY;
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
            m_colors ? kCL_N : ""
        );

    uv_mutex_lock(&m_mutex);

    vfprintf(stdout, buf, ap);
    fflush(stdout);

    uv_mutex_unlock(&m_mutex);

    va_end(ap);
}


void Log::text(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    const int len = 64 + strlen(fmt) + 2;
    char *buf = static_cast<char *>(alloca(len));

    sprintf(buf, "%s%s\n", fmt, m_colors ? kCL_N : "");

    uv_mutex_lock(&m_mutex);

    vfprintf(stdout, buf, ap);
    fflush(stdout);

    uv_mutex_unlock(&m_mutex);

    va_end(ap);
}


Log::Log(bool colors, bool background) :
    m_background(background),
    m_colors(colors)
{
    uv_mutex_init(&m_mutex);
}
