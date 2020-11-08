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

#ifndef XMRIG_LINUXMEMORY_H
#define XMRIG_LINUXMEMORY_H


#include <cstdint>
#include <cstddef>


namespace xmrig {


class LinuxMemory
{
public:
    static bool reserve(size_t size, uint32_t node, bool oneGbPages = false);

    static bool write(const char *path, uint64_t value);
    static int64_t read(const char *path);
};


} /* namespace xmrig */


#endif /* XMRIG_LINUXMEMORY_H */
