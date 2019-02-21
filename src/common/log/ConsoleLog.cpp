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


#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef WIN32
#   include <winsock2.h>
#   include <windows.h>
#endif


#include "common/log/ConsoleLog.h"
#include "common/log/Log.h"
#include "core/Config.h"
#include "core/Controller.h"


ConsoleLog::ConsoleLog(xmrig::Controller *controller) :
    m_stream(nullptr),
    m_controller(controller)
{
    if (uv_tty_init(uv_default_loop(), &m_tty, 1, 0) < 0) {
        Log::colors = false;
        return;
    }

    uv_tty_set_mode(&m_tty, UV_TTY_MODE_NORMAL);
    m_uvBuf.base = m_buf;
    m_stream     = reinterpret_cast<uv_stream_t*>(&m_tty);

#   ifdef WIN32
    HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
    if (handle != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(handle, &mode)) {
           mode &= ~ENABLE_QUICK_EDIT_MODE;
           SetConsoleMode(handle, mode | ENABLE_EXTENDED_FLAGS);
        }
    }
#   endif
}


void ConsoleLog::message(Level level, const char* fmt, va_list args)
{
    time_t now = time(nullptr);
    tm stime;

#   ifdef _WIN32
    localtime_s(&stime, &now);
#   else
    localtime_r(&now, &stime);
#   endif

    const bool isColors = m_controller->config()->isColors();

    snprintf(m_fmt, sizeof(m_fmt) - 1, "[%d-%02d-%02d %02d:%02d:%02d]%s %s%s",
             stime.tm_year + 1900,
             stime.tm_mon + 1,
             stime.tm_mday,
             stime.tm_hour,
             stime.tm_min,
             stime.tm_sec,
             Log::colorByLevel(level, isColors),
             fmt,
             Log::endl(isColors)
        );

    print(args);
}


void ConsoleLog::text(const char* fmt, va_list args)
{
    snprintf(m_fmt, sizeof(m_fmt) - 1, "%s%s", fmt, Log::endl(m_controller->config()->isColors()));

    print(args);
}


bool ConsoleLog::isWritable() const
{
    if (!m_stream || uv_is_writable(m_stream) != 1) {
        return false;
    }

    const uv_handle_type type = uv_guess_handle(1);
    return type == UV_TTY || type == UV_NAMED_PIPE;
}


void ConsoleLog::print(va_list args)
{
    m_uvBuf.len = vsnprintf(m_buf, sizeof(m_buf) - 1, m_fmt, args);
    if (m_uvBuf.len <= 0) {
        return;
    }

    if (!isWritable()) {
        fputs(m_buf, stdout);
        fflush(stdout);
    }
    else {
        uv_try_write(m_stream, &m_uvBuf, 1);
    }
}
