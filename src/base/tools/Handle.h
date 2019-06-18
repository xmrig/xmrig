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

#ifndef XMRIG_HANDLE_H
#define XMRIG_HANDLE_H


#include <uv.h>


namespace xmrig {


class Handle
{
public:
    template<typename T>
    static inline void close(T handle)
    {
        if (handle) {
            deleteLater(handle);
        }
    }


    template<typename T>
    static inline void deleteLater(T handle)
    {
        if (uv_is_closing(reinterpret_cast<uv_handle_t *>(handle))) {
            return;
        }

        uv_close(reinterpret_cast<uv_handle_t *>(handle), [](uv_handle_t *handle) { delete handle; });
    }
};


template<>
inline void Handle::close(uv_timer_t *handle)
{
    if (handle) {
        uv_timer_stop(handle);
        deleteLater(handle);
    }
}


template<>
inline void Handle::close(uv_signal_t *handle)
{
    if (handle) {
        uv_signal_stop(handle);
        deleteLater(handle);
    }
}


template<>
inline void Handle::close(uv_fs_event_t *handle)
{
    if (handle) {
        uv_fs_event_stop(handle);
        deleteLater(handle);
    }
}


} /* namespace xmrig */


#endif /* XMRIG_HANDLE_H */
