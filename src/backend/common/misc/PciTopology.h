/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef XMRIG_PCITOPOLOGY_H
#define XMRIG_PCITOPOLOGY_H


#include <stdio.h>


#include "base/tools/String.h"


namespace xmrig {


class PciTopology
{
public:
    PciTopology() = default;
    PciTopology(uint32_t bus, uint32_t device, uint32_t function) : bus(bus), device(device), function(function) {}

    uint32_t bus        = 0;
    uint32_t device     = 0;
    uint32_t function   = 0;

    String toString() const
    {
        char *buf = new char[8]();
        snprintf(buf, 8, "%02x:%02x.%01x", bus, device, function);

        return buf;
    }
};


} // namespace xmrig


#endif /* XMRIG_PCITOPOLOGY_H */
