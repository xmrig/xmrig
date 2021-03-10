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

        uv_close(reinterpret_cast<uv_handle_t *>(handle), [](uv_handle_t *handle) { delete reinterpret_cast<T>(handle); });
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
