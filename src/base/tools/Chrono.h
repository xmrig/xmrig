/* xmlcore
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_CHRONO_H
#define xmlcore_CHRONO_H


#include <chrono>


namespace xmlcore {


class Chrono
{
public:
    static inline uint64_t highResolutionMSecs()
    {
        using namespace std::chrono;

        return static_cast<uint64_t>(time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count());
    }


    static inline uint64_t steadyMSecs()
    {
        using namespace std::chrono;
        if (high_resolution_clock::is_steady) {
            return static_cast<uint64_t>(time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count());
        }

        return static_cast<uint64_t>(time_point_cast<milliseconds>(steady_clock::now()).time_since_epoch().count());
    }


    static inline uint64_t currentMSecsSinceEpoch()
    {
        using namespace std::chrono;

        return static_cast<uint64_t>(time_point_cast<milliseconds>(system_clock::now()).time_since_epoch().count());
    }
};


} /* namespace xmlcore */

#endif /* xmlcore_CHRONO_H */
