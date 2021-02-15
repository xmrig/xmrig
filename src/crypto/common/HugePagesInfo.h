/* xmlcore
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_HUGEPAGESINFO_H
#define xmlcore_HUGEPAGESINFO_H


#include <cstdint>
#include <cstddef>


namespace xmlcore {


class VirtualMemory;


class HugePagesInfo
{
public:
    HugePagesInfo() = default;
    HugePagesInfo(const VirtualMemory *memory);

    size_t allocated    = 0;
    size_t total        = 0;
    size_t size         = 0;

    inline bool isFullyAllocated() const { return allocated == total; }
    inline double percent() const        { return allocated == 0 ? 0.0 : static_cast<double>(allocated) / total * 100.0; }
    inline void reset()                  { allocated = 0; total = 0; size = 0; }

    inline HugePagesInfo &operator+=(const HugePagesInfo &other)
    {
        allocated += other.allocated;
        total     += other.total;
        size      += other.size;

        return *this;
    }
};


} /* namespace xmlcore */


#endif /* xmlcore_HUGEPAGESINFO_H */
