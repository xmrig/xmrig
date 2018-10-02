/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <assert.h>
#include <uv.h>
#include <vector>


#include "common/interfaces/ILogBackend.h"


class Log
{
public:
    static inline Log* i()                       { if (!m_self) { defaultInit(); } return m_self; }
    static inline void add(ILogBackend *backend) { i()->m_backends.push_back(backend); }
    static inline void init()                    { if (!m_self) { new Log(); } }
    static inline void release()                 { assert(m_self != nullptr); delete m_self; }

    void message(ILogBackend::Level level, const char* fmt, ...);
    void text(const char* fmt, ...);

    static const char *colorByLevel(ILogBackend::Level level, bool isColors = true);
    static const char *endl(bool isColors = true);
    static void defaultInit();

private:
    inline Log() {
        assert(m_self == nullptr);

        uv_mutex_init(&m_mutex);

        m_self = this;
    }

    ~Log();

    static Log *m_self;
    std::vector<ILogBackend*> m_backends;
    uv_mutex_t m_mutex;
};


#define RED_BOLD(x)     "\x1B[1;31m" x "\x1B[0m"
#define RED(x)          "\x1B[0;31m" x "\x1B[0m"
#define GREEN_BOLD(x)   "\x1B[1;32m" x "\x1B[0m"
#define GREEN(x)        "\x1B[0;32m" x "\x1B[0m"
#define YELLOW(x)       "\x1B[0;33m" x "\x1B[0m"
#define YELLOW_BOLD(x)  "\x1B[1;33m" x "\x1B[0m"
#define MAGENTA_BOLD(x) "\x1B[1;35m" x "\x1B[0m"
#define MAGENTA(x)      "\x1B[0;35m" x "\x1B[0m"
#define CYAN_BOLD(x)    "\x1B[1;36m" x "\x1B[0m"
#define CYAN(x)         "\x1B[0;36m" x "\x1B[0m"
#define WHITE_BOLD(x)   "\x1B[1;37m" x "\x1B[0m"
#define WHITE(x)        "\x1B[0;37m" x "\x1B[0m"


#define LOG_ERR(x, ...)    Log::i()->message(ILogBackend::ERR,     x, ##__VA_ARGS__)
#define LOG_WARN(x, ...)   Log::i()->message(ILogBackend::WARNING, x, ##__VA_ARGS__)
#define LOG_NOTICE(x, ...) Log::i()->message(ILogBackend::NOTICE,  x, ##__VA_ARGS__)
#define LOG_INFO(x, ...)   Log::i()->message(ILogBackend::INFO,    x, ##__VA_ARGS__)

#ifdef APP_DEBUG
#   define LOG_DEBUG(x, ...)      Log::i()->message(ILogBackend::DEBUG,   x, ##__VA_ARGS__)
#else
#   define LOG_DEBUG(x, ...)
#endif

#if defined(APP_DEBUG) || defined(APP_DEVEL)
#   define LOG_DEBUG_ERR(x, ...)  Log::i()->message(ILogBackend::ERR,     x, ##__VA_ARGS__)
#   define LOG_DEBUG_WARN(x, ...) Log::i()->message(ILogBackend::WARNING, x, ##__VA_ARGS__)
#else
#   define LOG_DEBUG_ERR(x, ...)
#   define LOG_DEBUG_WARN(x, ...)
#endif

#endif /* __LOG_H__ */
