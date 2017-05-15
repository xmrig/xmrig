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

#include "xmrig.h"
#include "applog.h"
#include "threads.h"
#include <sys/time.h>
#include <string.h>

#ifdef WIN32
# include "compat/winansi.h"
#endif

#include "options.h"


MUTEX applog_mutex;


void applog_init()
{
    MUTEX_INIT(applog_mutex);
}


void applog(int prio, const char *fmt, ...)
{
    if (opt_background) {
        return;
    }

    va_list ap;
    va_start(ap, fmt);

    struct tm tm;
    struct tm *tm_p;
    time_t now = time(NULL);

    MUTEX_LOCK(applog_mutex);
    tm_p = localtime(&now);
    memcpy(&tm, tm_p, sizeof(tm));
    MUTEX_UNLOCK(applog_mutex);

    const char* color = "";

    if (opt_colors) {
        switch (prio) {
            case LOG_ERR:     color = CL_RED; break;
            case LOG_WARNING: color = CL_YLW; break;
            case LOG_NOTICE:  color = CL_WHT; break;
            case LOG_INFO:    color = ""; break;
            case LOG_DEBUG:   color = CL_GRY; break;

            case LOG_BLUE:
                prio = LOG_NOTICE;
                color = CL_CYN;
                break;

            case LOG_GREEN:
                prio = LOG_NOTICE;
                color = CL_LGR;
                break;
        }
    }

    const int len = 64 + strlen(fmt) + 2;
    char *f       = alloca(len);

    sprintf(f, "[%d-%02d-%02d %02d:%02d:%02d]%s %s%s\n",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec,
            color,
            fmt,
            opt_colors ? CL_N : ""
        );

    MUTEX_LOCK(applog_mutex);
    vfprintf(stderr, f, ap);
    fflush(stderr);
    MUTEX_UNLOCK(applog_mutex);

    va_end(ap);
}


void applog_notime(int prio, const char *fmt, ...)
{
    if (opt_background) {
        return;
    }

    va_list ap;
    va_start(ap, fmt);

    const char* color = "";

    if (opt_colors) {
        switch (prio) {
            case LOG_ERR:     color = CL_RED; break;
            case LOG_WARNING: color = CL_LYL; break;
            case LOG_NOTICE:  color = CL_WHT; break;
            case LOG_INFO:    color = ""; break;
            case LOG_DEBUG:   color = CL_GRY; break;

            case LOG_BLUE:
                prio = LOG_NOTICE;
                color = CL_CYN;
                break;
        }
    }

    const int len = 64 + strlen(fmt) + 2;
    char *f       = alloca(len);

    sprintf(f, "%s%s%s\n",
            color,
            fmt,
            opt_colors ? CL_N : ""
        );

    MUTEX_LOCK(applog_mutex);
    vfprintf(stderr, f, ap);
    fflush(stderr);
    MUTEX_UNLOCK(applog_mutex);

    va_end(ap);
}
