/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CPUTHREAD_H
#define XMRIG_CPUTHREAD_H


#include "3rdparty/rapidjson/fwd.h"


namespace xmrig {


class CpuThread
{
public:
    inline constexpr CpuThread() = default;
    inline constexpr CpuThread(int64_t affinity, uint32_t intensity) : m_affinity(affinity), m_intensity(intensity) {}

    CpuThread(const rapidjson::Value &value);

    inline bool isEqual(const CpuThread &other) const       { return other.m_affinity == m_affinity && other.m_intensity == m_intensity; }
    inline bool isValid() const                             { return m_intensity <= 5; }
    inline int64_t affinity() const                         { return m_affinity; }
    inline uint32_t intensity() const                       { return m_intensity == 0 ? 1 : m_intensity; }

    inline bool operator!=(const CpuThread &other) const    { return !isEqual(other); }
    inline bool operator==(const CpuThread &other) const    { return isEqual(other); }

    rapidjson::Value toJSON(rapidjson::Document &doc) const;

private:
    int64_t m_affinity   = -1;
    uint32_t m_intensity = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_CPUTHREAD_H */
