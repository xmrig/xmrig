/* XMRig
 * Copyright (c) 2021      Spudz76     <https://github.com/Spudz76>
 * Copyright (c) 2018-2024 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "backend/opencl/wrappers/OclDevice.h"


xmrig::OclDevice::Type xmrig::OclDevice::getType(const String &name)
{
    // Apple Platform: uses product names, not gfx# or codenames
    if (name.contains("AMD Radeon")) {
        if (name.contains(" 450 ") ||
            name.contains(" 455 ") ||
            name.contains(" 460 ")) {
            return Baffin;
        }

        if (name.contains(" 555 ") || name.contains(" 555X ") ||
            name.contains(" 560 ") || name.contains(" 560X ") ||
            name.contains(" 570 ") || name.contains(" 570X ") ||
            name.contains(" 575 ") || name.contains(" 575X ")) {
            return Polaris;
        }

        if (name.contains(" 580 ") || name.contains(" 580X ")) {
            return Ellesmere;
        }

        if (name.contains(" Vega ")) {
            if (name.contains(" 48 ") ||
                name.contains(" 56 ") ||
                name.contains(" 64 ") ||
                name.contains(" 64X ")) {
                return Vega_10;
            }
            if (name.contains(" 16 ") ||
                name.contains(" 20 ") ||
                name.contains(" II ")) {
                return Vega_20;
            }
        }

        if (name.contains(" 5700 ") || name.contains(" W5700X ")) {
            return Navi_10;
        }

        if (name.contains(" 5600 ") || name.contains(" 5600M ")) {
            return Navi_12;
        }

        if (name.contains(" 5300 ") || name.contains(" 5300M ") ||
            name.contains(" 5500 ") || name.contains(" 5500M ")) {
            return Navi_14;
        }

        if (name.contains(" W6800 ") || name.contains(" W6900X ")) {
            return Navi_21;
        }
    }

    return OclDevice::Unknown;
}
