/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CONSOLE_H
#define XMRIG_CONSOLE_H


#include "base/tools/Object.h"


using uv_buf_t      = struct uv_buf_t;
using uv_handle_t   = struct uv_handle_s;
using uv_stream_t   = struct uv_stream_s;
using uv_tty_t      = struct uv_tty_s;

#ifdef XMRIG_OS_WIN
using ssize_t = intptr_t;
#else
#   include <sys/types.h>
#endif


namespace xmrig {


class IConsoleListener;


class Console
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Console)

    Console(IConsoleListener *listener);
    ~Console();

private:
    bool isSupported() const;

    static void onAllocBuffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf);
    static void onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);

    char m_buf[1] = { 0 };
    IConsoleListener *m_listener;
    uv_tty_t *m_tty = nullptr;
};


} /* namespace xmrig */


#endif /* XMRIG_CONSOLE_H */
