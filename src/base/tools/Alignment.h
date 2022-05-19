/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_ALIGNMENT_H
#define XMRIG_ALIGNMENT_H


#include <type_traits>
#include <cstring>


namespace xmrig {


template<typename T>
inline T readUnaligned(const T* ptr)
{
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

    T result;
    memcpy(&result, ptr, sizeof(T));
    return result;
}


template<typename T>
inline void writeUnaligned(T* ptr, T data)
{
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

    memcpy(ptr, &data, sizeof(T));
}


} /* namespace xmrig */


#endif /* XMRIG_ALIGNMENT_H */
