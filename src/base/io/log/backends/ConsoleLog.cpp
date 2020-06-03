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


#include <cstdio>


#include "base/io/log/backends/ConsoleLog.h"
#include "base/tools/Handle.h"
#include "base/io/log/Log.h"
#include "version.h"


xmrig::ConsoleLog::ConsoleLog()
{
    if (!isSupported()) {
        Log::setColors(false);
        return;
    }

    m_tty = new uv_tty_t;

    if (uv_tty_init(uv_default_loop(), m_tty, 1, 0) < 0) {
        Log::setColors(false);
        return;
    }

    uv_tty_set_mode(m_tty, UV_TTY_MODE_NORMAL);

#   ifdef XMRIG_OS_WIN
    m_stream = reinterpret_cast<uv_stream_t*>(m_tty);

    HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
    if (handle != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(handle, &mode)) {
           mode &= ~ENABLE_QUICK_EDIT_MODE;
           SetConsoleMode(handle, mode | ENABLE_EXTENDED_FLAGS);
        }
    }

    SetConsoleTitleA(APP_NAME " " APP_VERSION);
#   endif
}


xmrig::ConsoleLog::~ConsoleLog()
{
    Handle::close(m_tty);
}


void xmrig::ConsoleLog::print(int, const char *line, size_t, size_t size, bool colors)
{
    if (!m_tty || Log::isColors() != colors) {
        return;
    }

#   ifdef XMRIG_OS_WIN
    uv_buf_t buf = uv_buf_init(const_cast<char *>(line), static_cast<unsigned int>(size));

    if (!isWritable()) {
        fputs(line, stdout);
        fflush(stdout);
    }
    else {
        uv_try_write(m_stream, &buf, 1);
    }
#   else
    fputs(line, stdout);
    fflush(stdout);
#   endif
}


bool xmrig::ConsoleLog::isSupported() const
{
    const uv_handle_type type = uv_guess_handle(1);
    return type == UV_TTY || type == UV_NAMED_PIPE;
}


#ifdef XMRIG_OS_WIN
bool xmrig::ConsoleLog::isWritable() const
{
    if (!m_stream || uv_is_writable(m_stream) != 1) {
        return false;
    }

    return isSupported();
}
#endif
