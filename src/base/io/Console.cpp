/* XMRig
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

#include "base/io/Console.h"
#include "base/kernel/interfaces/IConsoleListener.h"
#include "base/tools/Handle.h"


xmrig::Console::Console(IConsoleListener *listener)
    : m_listener(listener)
{
    if (!isSupported()) {
        return;
    }

    m_input = reinterpret_cast<uv_stream_t*>(new uv_tty_t);
    m_input->data = this;
    uv_tty_init(uv_default_loop(), reinterpret_cast<uv_tty_t*>(m_input), 0, 1);

    if (uv_is_readable(reinterpret_cast<uv_stream_t*>(m_input))) {
        uv_tty_set_mode(reinterpret_cast<uv_tty_t*>(m_input), UV_TTY_MODE_RAW);
        uv_read_start(m_input, Console::onAllocBuffer, Console::onRead);
    } else {
        /* Direct TTY connection doesn't work, so use stdin as pipe as fallback. 
         * N.B. Requires pipe to be flushed on the producer side before it is received in our stdin. 
         * For example if you run xmrig from another process and send commands to the stdin of the xmrig process, 
         * flush that stream after writing a character to it.
         */
        m_input = reinterpret_cast<uv_stream_t*>(new uv_pipe_t);
        m_input->data = this;
        uv_pipe_init(uv_default_loop(), reinterpret_cast<uv_pipe_t*>(m_input), 0);
        uv_pipe_open(reinterpret_cast<uv_pipe_t*>(m_input), 0);
        uv_read_start(m_input, Console::onAllocBuffer, Console::onRead);
    }
}


xmrig::Console::~Console()
{
    uv_tty_reset_mode();

    Handle::close(m_input);
}


bool xmrig::Console::isSupported()
{
    const uv_handle_type type = uv_guess_handle(0);
    return type == UV_TTY || type == UV_NAMED_PIPE;
}


void xmrig::Console::onAllocBuffer(uv_handle_t *handle, size_t, uv_buf_t *buf)
{
    auto console = static_cast<Console*>(handle->data);
    buf->len  = 1;
    buf->base = console->m_buf;
}


void xmrig::Console::onRead(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf)
{
    if (nread < 0) {
        return uv_close(reinterpret_cast<uv_handle_t*>(stream), nullptr);
    }

    if (nread == 1) {
        static_cast<Console*>(stream->data)->m_listener->onConsoleCommand(buf->base[0]);
    }
}
