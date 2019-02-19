/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include <uv.h>


#include "base/tools/Handle.h"


void xmrig::Handle::close(uv_fs_event_t *handle)
{
    if (handle) {
        uv_fs_event_stop(handle);
        close(reinterpret_cast<uv_handle_t *>(handle));
    }
}


void xmrig::Handle::close(uv_getaddrinfo_t *handle)
{
    if (handle) {
        uv_cancel(reinterpret_cast<uv_req_t *>(handle));
        close(reinterpret_cast<uv_handle_t *>(handle));
    }
}


void xmrig::Handle::close(uv_handle_t *handle)
{
    uv_close(handle, [](uv_handle_t *handle) { delete handle; });
}


void xmrig::Handle::close(uv_signal_t *handle)
{
    if (handle) {
        uv_signal_stop(handle);
        close(reinterpret_cast<uv_handle_t *>(handle));
    }
}


void xmrig::Handle::close(uv_tcp_t *handle)
{
    if (handle) {
        close(reinterpret_cast<uv_handle_t *>(handle));
    }
}


void xmrig::Handle::close(uv_timer_s *handle)
{
    if (handle) {
        uv_timer_stop(handle);
        close(reinterpret_cast<uv_handle_t *>(handle));
    }
}
