/* XMRig
 * Copyright 2018-     BenDr0id <ben@graef.in>
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


#include <sstream>
#include <regex>
#include "log/RemoteLog.h"

RemoteLog* RemoteLog::m_self = nullptr;

RemoteLog::RemoteLog(size_t maxRows)
    : maxRows_(maxRows)
{
    m_self = this;
}

RemoteLog::~RemoteLog()
{
    m_self = nullptr;
}

void RemoteLog::message(int level, const char* fmt, va_list args)
{
    time_t now = time(nullptr);
    tm stime;

#   ifdef _WIN32
    localtime_s(&stime, &now);
#   else
    localtime_r(&now, &stime);
#   endif

    auto *buf = new char[512];
    int size = snprintf(buf, 23, "[%d-%02d-%02d %02d:%02d:%02d] ",
                        stime.tm_year + 1900,
                        stime.tm_mon + 1,
                        stime.tm_mday,
                        stime.tm_hour,
                        stime.tm_min,
                        stime.tm_sec);

    size = vsnprintf(buf + size, 512 - size - 1, fmt, args) + size;
    buf[size] = '\n';

    if (rows_.size() == maxRows_) {
        rows_.pop_front();
    }

    std::string row = std::regex_replace(std::string(buf, size+1), std::regex("\x1B\\[[0-9;]*[a-zA-Z]"), "");

    rows_.push_back(row);

    delete[](buf);
}


void RemoteLog::text(const char* fmt, va_list args)
{
    message(0, fmt, args);
}

void RemoteLog::flushRows()
{
    if (m_self) {
        m_self->rows_.clear();
    }
}


std::string RemoteLog::getRows()
{
    std::stringstream data;

    if (m_self) {
        for (std::list<std::string>::iterator it = m_self->rows_.begin(); it != m_self->rows_.end(); it++) {
            data << it->c_str();
        }
    }

    return data.str();
}
