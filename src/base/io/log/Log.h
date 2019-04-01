/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2019      Spudz76     <https://github.com/Spudz76>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_LOG_H
#define XMRIG_LOG_H


namespace xmrig {


class ILogBackend;
class LogPrivate;


class Log
{
public:
    enum Level : int {
        NONE = -1,
        EMERG,   // system is unusable
        ALERT,   // action must be taken immediately
        CRIT,    // critical conditions
        ERR,     // error conditions
        WARNING, // warning conditions
        NOTICE,  // normal but significant condition
        INFO,    // informational
        DEBUG,   // debug-level messages
    };

    static void add(ILogBackend *backend);
    static void destroy();
    static void print(const char *fmt, ...);
    static void print(Level level, const char *fmt, ...);

    static bool colors;

private:
    static LogPrivate *d;
};


#define CSI            "\x1B["     // Control Sequence Introducer (ANSI spec name)
#define CLEAR          CSI "0m"    // all attributes off
#define BRIGHT_BLACK_S CSI "0;90m" // somewhat MD.GRAY
#define BLACK_S        CSI "0;30m"
#define BLACK_BOLD_S   CSI "1;30m" // another name for GRAY
#define RED_S          CSI "0;31m"
#define RED_BOLD_S     CSI "1;31m"
#define GREEN_S        CSI "0;32m"
#define GREEN_BOLD_S   CSI "1;32m"
#define YELLOW_S       CSI "0;33m"
#define YELLOW_BOLD_S  CSI "1;33m"
#define BLUE_S         CSI "0;34m"
#define BLUE_BOLD_S    CSI "1;34m"
#define MAGENTA_S      CSI "0;35m"
#define MAGENTA_BOLD_S CSI "1;35m"
#define CYAN_S         CSI "0;36m"
#define CYAN_BOLD_S    CSI "1;36m"
#define WHITE_S        CSI "0;37m" // another name for LT.GRAY
#define WHITE_BOLD_S   CSI "1;37m" // actually white

//color wrappings
#define BLACK(x)        BLACK_S x CLEAR
#define BLACK_BOLD(x)   BLACK_BOLD_S x CLEAR
#define RED(x)          RED_S x CLEAR
#define RED_BOLD(x)     RED_BOLD_S x CLEAR
#define GREEN(x)        GREEN_S x CLEAR
#define GREEN_BOLD(x)   GREEN_BOLD_S x CLEAR
#define YELLOW(x)       YELLOW_S x CLEAR
#define YELLOW_BOLD(x)  YELLOW_BOLD_S x CLEAR
#define BLUE(x)         BLUE_S x CLEAR
#define BLUE_BOLD(x)    BLUE_BOLD_S x CLEAR
#define MAGENTA(x)      MAGENTA_S x CLEAR
#define MAGENTA_BOLD(x) MAGENTA_BOLD_S x CLEAR
#define CYAN(x)         CYAN_S x CLEAR
#define CYAN_BOLD(x)    CYAN_BOLD_S x CLEAR
#define WHITE(x)        WHITE_S x CLEAR
#define WHITE_BOLD(x)   WHITE_BOLD_S x CLEAR


#define LOG_EMERG(x, ...)  xmrig::Log::print(xmrig::Log::EMERG,   x, ##__VA_ARGS__)
#define LOG_ALERT(x, ...)  xmrig::Log::print(xmrig::Log::ALERT,   x, ##__VA_ARGS__)
#define LOG_CRIT(x, ...)   xmrig::Log::print(xmrig::Log::CRIT,    x, ##__VA_ARGS__)
#define LOG_ERR(x, ...)    xmrig::Log::print(xmrig::Log::ERR,     x, ##__VA_ARGS__)
#define LOG_WARN(x, ...)   xmrig::Log::print(xmrig::Log::WARNING, x, ##__VA_ARGS__)
#define LOG_NOTICE(x, ...) xmrig::Log::print(xmrig::Log::NOTICE,  x, ##__VA_ARGS__)
#define LOG_INFO(x, ...)   xmrig::Log::print(xmrig::Log::INFO,    x, ##__VA_ARGS__)

#ifdef APP_DEBUG
#   define LOG_DEBUG(x, ...) xmrig::Log::print(xmrig::Log::DEBUG, x, ##__VA_ARGS__)
#else
#   define LOG_DEBUG(x, ...)
#endif

#if defined(APP_DEBUG) || defined(APP_DEVEL)
#   define LOG_DEBUG_ERR(x, ...)  xmrig::Log::print(xmrig::Log::ERR,     x, ##__VA_ARGS__)
#   define LOG_DEBUG_WARN(x, ...) xmrig::Log::print(xmrig::Log::WARNING, x, ##__VA_ARGS__)
#else
#   define LOG_DEBUG_ERR(x, ...)
#   define LOG_DEBUG_WARN(x, ...)
#endif


} /* namespace xmrig */


#endif /* XMRIG_LOG_H */
