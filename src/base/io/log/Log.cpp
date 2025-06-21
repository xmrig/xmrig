/* XMRig
 * Copyright (c) 2019      Spudz76     <https://github.com/Spudz76>
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

#ifdef WIN32
#   include <winsock2.h>
#   include <windows.h>
#endif


#include <algorithm>
#include <cassert>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <uv.h>
#include <vector>


#include "base/io/log/Log.h"
#include "base/kernel/interfaces/ILogBackend.h"
#include "base/tools/Chrono.h"
#include "base/tools/Object.h"


namespace xmrig {


static const char *colors_map[] = {
    RED_BOLD_S,    // EMERG
    RED_BOLD_S,    // ALERT
    RED_BOLD_S,    // CRIT
    RED_S,         // ERR
    YELLOW_S,      // WARNING
    WHITE_BOLD_S,  // NOTICE
    nullptr,       // INFO
#   ifdef WIN32
    BLACK_BOLD_S   // DEBUG
#   else
    BRIGHT_BLACK_S // DEBUG
#   endif
};



class LogPrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE(LogPrivate)


    LogPrivate() = default;
    ~LogPrivate() = default;


    inline void add(std::shared_ptr<ILogBackend> backend) { m_backends.emplace_back(backend); }


    void print(Log::Level level, const char *fmt, va_list args)
    {
        size_t size   = 0;
        size_t offset = 0;

        std::lock_guard<std::mutex> lock(m_mutex);

        if (Log::isBackground() && m_backends.empty()) {
            return;
        }

        const uint64_t ts = timestamp(level, size, offset);
        color(level, size);

        const int rc = vsnprintf(m_buf + size, sizeof (m_buf) - offset - 32, fmt, args);
        if (rc < 0) {
            return;
        }

        size += std::min(static_cast<size_t>(rc), sizeof (m_buf) - offset - 32);
        endl(size);

        std::string txt(m_buf);
        size_t i = 0;
        while ((i = txt.find(CSI)) != std::string::npos) {
            txt.erase(i, txt.find('m', i) - i + 1);
        }

        if (!m_backends.empty()) {
            for (auto& backend : m_backends) {
                backend->print(ts, level, m_buf, offset, size, true);
                backend->print(ts, level, txt.c_str(), offset ? (offset - 11) : 0, txt.size(), false);
            }
        }
        else {
            fputs(txt.c_str(), stdout);
            fflush(stdout);
        }
    }


private:
    inline uint64_t timestamp(Log::Level level, size_t &size, size_t &offset)
    {
        const uint64_t ms = Chrono::currentMSecsSinceEpoch();

        if (level == Log::NONE) {
            return ms;
        }

        time_t now = ms / 1000;
        tm stime{};

#       ifdef _WIN32
        localtime_s(&stime, &now);
#       else
        localtime_r(&now, &stime);
#       endif

        const int rc = snprintf(m_buf, sizeof(m_buf) - 1, "[%d-%02d-%02d %02d:%02d:%02d" BLACK_BOLD(".%03d") "] ",
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


    inline void color(Log::Level level, size_t &size)
    {
        if (level == Log::NONE) {
            return;
        }

        const char *color = colors_map[level];
        if (color == nullptr) {
            return;
        }

        const size_t s = strlen(color);
        memcpy(m_buf + size, color, s);

        size += s;
    }


    inline void endl(size_t &size)
    {
#       ifdef _WIN32
        memcpy(m_buf + size, CLEAR "\r\n", 7);
        size += 6;
#       else
        memcpy(m_buf + size, CLEAR "\n", 6);
        size += 5;
#       endif
    }


    char m_buf[Log::kMaxBufferSize]{};
    std::mutex m_mutex;
    std::vector<std::shared_ptr<ILogBackend>> m_backends;
};


bool Log::m_background      = false;
bool Log::m_colors          = true;
std::shared_ptr<LogPrivate> Log::d{};
uint32_t Log::m_verbose     = 0;


} /* namespace xmrig */



void xmrig::Log::add(std::shared_ptr<ILogBackend> backend)
{
    assert(d != nullptr);

    if (d) {
        d->add(backend);
    }
}


void xmrig::Log::destroy()
{
    d.reset();
}


void xmrig::Log::init()
{
    d = std::make_shared<LogPrivate>();
}


void xmrig::Log::print(const char *fmt, ...)
{
    if (!d) {
        return;
    }

    va_list args{};
    va_start(args, fmt);

    d->print(NONE, fmt, args);

    va_end(args);
}


void xmrig::Log::print(Level level, const char *fmt, ...)
{
    if (!d) {
        return;
    }

    va_list args{};
    va_start(args, fmt);

    d->print(level, fmt, args);

    va_end(args);
}
