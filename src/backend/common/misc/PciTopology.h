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


#include <cstdio>


#include "base/tools/String.h"


namespace xmrig {


class PciTopology
{
public:
    PciTopology() = default;
    PciTopology(uint32_t bus, uint32_t device, uint32_t function) : m_valid(true), m_bus(bus), m_device(device), m_function(function) {}

    inline bool isValid() const        { return m_valid; }
    inline uint8_t bus() const         { return m_bus; }
    inline uint8_t device() const      { return m_device; }
    inline uint8_t function() const    { return m_function; }

    String toString() const
    {
        if (!isValid()) {
            return "n/a";
        }

        char *buf = new char[8]();
        snprintf(buf, 8, "%02hhx:%02hhx.%01hhx", bus(), device(), function());

        return buf;
    }

private:
    bool m_valid         = false;
    uint8_t m_bus        = 0;
    uint8_t m_device     = 0;
    uint8_t m_function   = 0;
};


} // namespace xmrig


#endif /* XMRIG_PCITOPOLOGY_H */
