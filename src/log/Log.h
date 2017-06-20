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

#ifndef __LOG_H__
#define __LOG_H__


#include <uv.h>


class Log
{
public:
    enum Level {
        ERR,
        WARNING,
        NOTICE,
        INFO,
        DEBUG
    };

    constexpr static const char* kCL_N      = "\x1B[0m";
    constexpr static const char* kCL_RED    = "\x1B[31m";
    constexpr static const char* kCL_YELLOW = "\x1B[33m";
    constexpr static const char* kCL_WHITE  = "\x1B[01;37m";

#   ifdef WIN32
    constexpr static const char* kCL_GRAY = "\x1B[01;30m";
#   else
    constexpr static const char* kCL_GRAY = "\x1B[90m";
#   endif

    static inline Log* i() { return m_self; }
    static void init(bool colors, bool background);

    void message(Level level, const char* fmt, ...);
    void text(const char* fmt, ...);

private:
    Log(bool colors, bool background);

    bool m_background;
    bool m_colors;
    static Log *m_self;
    uv_mutex_t m_mutex;
};


#define LOG_ERR(x, ...)    Log::i()->message(Log::ERR,     x, ##__VA_ARGS__)
#define LOG_WARN(x, ...)   Log::i()->message(Log::WARNING, x, ##__VA_ARGS__)
#define LOG_NOTICE(x, ...) Log::i()->message(Log::NOTICE,  x, ##__VA_ARGS__)
#define LOG_INFO(x, ...)   Log::i()->message(Log::INFO,    x, ##__VA_ARGS__)

#ifdef APP_DEBUG
#   define LOG_DEBUG(x, ...)      Log::i()->message(Log::DEBUG,   x, ##__VA_ARGS__)
#   define LOG_DEBUG_ERR(x, ...)  Log::i()->message(Log::ERR,     x, ##__VA_ARGS__)
#   define LOG_DEBUG_WARN(x, ...) Log::i()->message(Log::WARNING, x, ##__VA_ARGS__)
#else
#   define LOG_DEBUG(x, ...)
#   define LOG_DEBUG_ERR(x, ...)
#   define LOG_DEBUG_WARN(x, ...)
#endif

#endif /* __LOG_H__ */
