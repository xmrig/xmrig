/* XMRig
 * Copyright (c) 2015-2020 libuv project contributors.
 * Copyright (c) 2020      cohcho      <https://github.com/cohcho>
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

#ifndef XMRIG_ASYNC_H
#define XMRIG_ASYNC_H


#include <uv.h>


// since 2019.05.16, Version 1.29.0 (Stable)
#if (UV_VERSION_MAJOR >= 1) && (UV_VERSION_MINOR >= 29) && defined(__linux__)
#define XMRIG_UV_PERFORMANCE_BUG
namespace xmrig {


struct uv_async_t: uv_poll_t
{
    typedef void (*uv_async_cb)(uv_async_t* handle);
    ~uv_async_t();
    int m_fd = -1;
    uv_async_cb m_cb = nullptr;
};


using uv_async_cb = uv_async_t::uv_async_cb;
extern int uv_async_init(uv_loop_t *loop, uv_async_t *async, uv_async_cb cb);
extern int uv_async_send(uv_async_t *async);


} // namespace xmrig
#endif


#endif /* XMRIG_ASYNC_H */
