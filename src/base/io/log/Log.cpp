/* XMRig
 * Copyright (c) 2019      Spudz76     <https://github.com/Spudz76>
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
 *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
 */

#ifdef XMRIG_OS_WIN
#   include <winsock2.h>
#   include <windows.h>
#endif


#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <mutex>
#include <vector>


#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"


#ifdef XMRIG_FEATURE_EVENTS
#   include "base/kernel/Events.h"
#   include "base/kernel/events/LogEvent.h"
#   include "base/kernel/private/LogConfig.h"
#   include "base/kernel/Process.h"
#else
#   include "base/kernel/interfaces/ILogBackend.h"
#endif


namespace xmrig {


bool Log::m_background      = false;
bool Log::m_colors          = true;
LogPrivate *Log::d          = nullptr;
uint32_t Log::m_verbose     = 0;


static char buf[Log::kMaxBufferSize]{};
static std::mutex mutex;


static const char *colors_map[] = {
    RED_BOLD_S      "E ", // EMERG
    RED_BOLD_S      "A ", // ALERT
    RED_BOLD_S      "C ", // CRIT
    RED_S           "E ", // ERR
    YELLOW_S        "W ", // WARNING
    WHITE_BOLD_S    "N ", // NOTICE
                    "I ", // INFO
                    "1 ", // V1
                    "2 ", // V2
                    "3 ", // V3
                    "4 ", // V4
#   ifdef XMRIG_OS_WIN
    BLACK_BOLD_S    "5 ", // V5
    BLACK_BOLD_S    "D "  // DEBUG
#   else
    BRIGHT_BLACK_S  "5 ", // V5
    BRIGHT_BLACK_S  "D "  // DEBUG
#   endif
};


static void log_endl(size_t &size)
{
#   ifdef XMRIG_OS_WIN
    memcpy(buf + size, CLEAR "\r\n", 7);
    size += 6;
#   else
    memcpy(buf + size, CLEAR "\n", 6);
    size += 5;
#   endif
}


static void log_color(Log::Level level, size_t &size)
{
    if (level == Log::NONE) {
        return;
    }

    const char *color = colors_map[level];
    if (color == nullptr) {
        return;
    }

    const size_t s = strlen(color);
    memcpy(buf + size, color, s); // NOLINT(bugprone-not-null-terminated-result)

    size += s;
}


static uint64_t log_timestamp(Log::Level level, size_t &size, size_t &offset)
{
    const uint64_t ms = Chrono::currentMSecsSinceEpoch();

    if (level == Log::NONE) {
        return ms;
    }

    time_t now = ms / 1000;
    tm stime{};

#   ifdef XMRIG_OS_WIN
    localtime_s(&stime, &now);
#   else
    localtime_r(&now, &stime);
#   endif

    const int rc = snprintf(buf, sizeof(buf) - 1, "[%d-%02d-%02d %02d:%02d:%02d" BLACK_BOLD(".%03d") "] ",
                            stime.tm_year + 1900,
                            stime.tm_mon + 1,
                            stime.tm_mday,
                            stime.tm_hour,
                            stime.tm_min,
                            stime.tm_sec,
                            static_cast<int>(ms % 1000)
                            );

    if (rc > 0) {
        size = offset = static_cast<size_t>(rc);
    }

    return ms;
}


#ifdef XMRIG_FEATURE_EVENTS
static void log_print(Log::Level level, const char *fmt, va_list args)
{
    size_t size   = 0;
    size_t offset = 0;

    std::lock_guard<std::mutex> lock(mutex);

    const uint64_t ts = log_timestamp(level, size, offset);
    log_color(level, size);

    const int rc = vsnprintf(buf + size, sizeof(buf) - offset - 32, fmt, args);
    if (rc < 0) {
        return;
    }

    size += std::min(static_cast<size_t>(rc), sizeof(buf) - offset - 32);
    log_endl(size);

    Process::events().post<LogEvent>(ts, level, buf, offset, size);
}
#else
class LogPrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE(LogPrivate)


    LogPrivate() = default;


    inline ~LogPrivate()
    {
        for (auto backend : backends) {
            delete backend;
        }
    }


    void print(Log::Level level, const char *fmt, va_list args)
    {
        size_t size   = 0;
        size_t offset = 0;

        std::lock_guard<std::mutex> lock(mutex);

        if (Log::isBackground() && backends.empty()) {
            return;
        }

        const uint64_t ts = log_timestamp(level, size, offset);
        log_color(level, size);

        const int rc = vsnprintf(buf + size, sizeof (buf) - offset - 32, fmt, args);
        if (rc < 0) {
            return;
        }

        size += std::min(static_cast<size_t>(rc), sizeof (buf) - offset - 32);
        log_endl(size);

        std::string txt(buf);
        size_t i = 0;
        while ((i = txt.find(CSI)) != std::string::npos) {
            txt.erase(i, txt.find('m', i) - i + 1);
        }

        if (!backends.empty()) {
            for (auto backend : backends) {
                backend->print(ts, level, buf, offset, size, true);
                backend->print(ts, level, txt.c_str(), offset ? (offset - 11) : 0, txt.size(), false);
            }
        }
        else {
            fputs(txt.c_str(), stdout);
            fflush(stdout);
        }
    }


    std::vector<ILogBackend*> backends;
};
#endif


} // namespace xmrig


#ifndef XMRIG_FEATURE_EVENTS
void xmrig::Log::add(ILogBackend *backend)
{
    assert(d != nullptr);

    if (d) {
        d->backends.push_back(backend);
    }
}


void xmrig::Log::destroy()
{
    delete d;
    d = nullptr;
}


void xmrig::Log::init()
{
    d = new LogPrivate();
}
#endif


void xmrig::Log::print(const char *fmt, ...)
{
#   ifndef XMRIG_FEATURE_EVENTS
    if (!d) {
        return;
    }
#   endif

    va_list args{};
    va_start(args, fmt);

#   ifdef XMRIG_FEATURE_EVENTS
    log_print(NONE, fmt, args);
#   else
    d->print(NONE, fmt, args);
#   endif

    va_end(args);
}


void xmrig::Log::print(Level level, const char *fmt, ...)
{
#   ifndef XMRIG_FEATURE_EVENTS
    if (!d) {
        return;
    }
#   endif

    va_list args{};
    va_start(args, fmt);

#   ifdef XMRIG_FEATURE_EVENTS
    log_print(level, fmt, args);
#   else
    d->print(level, fmt, args);
#   endif

    va_end(args);
}


void xmrig::Log::setVerbose(uint32_t verbose)
{
    static constexpr uint32_t kMaxVerbose =
#   ifdef XMRIG_FEATURE_EVENTS
    LogConfig::kMaxVerbose;
#   else
    5U;
#   endif

    m_verbose = std::min(verbose, kMaxVerbose);
}
