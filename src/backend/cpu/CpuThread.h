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

#ifndef XMRIG_CPUTHREADCONFIG_H
#define XMRIG_CPUTHREADCONFIG_H


#include <vector>


#include "rapidjson/fwd.h"


namespace xmrig {


class CpuThread
{
public:
    inline constexpr CpuThread(int intensity = 1, int64_t affinity = -1) : m_intensity(intensity), m_affinity(affinity) {}

    CpuThread(const rapidjson::Value &value);

    inline bool isEqual(const CpuThread &other) const       { return other.m_affinity == m_affinity && other.m_intensity == m_intensity; }
    inline bool isValid() const                             { return m_intensity >= 1 && m_intensity <= 5; }
    inline int intensity() const                            { return m_intensity; }
    inline int64_t affinity() const                         { return m_affinity; }

    inline bool operator!=(const CpuThread &other) const    { return !isEqual(other); }
    inline bool operator==(const CpuThread &other) const    { return isEqual(other); }

    rapidjson::Value toJSON(rapidjson::Document &doc) const;

private:
    int m_intensity     = -1;
    int64_t m_affinity  = -1;
};


typedef std::vector<CpuThread> CpuThreads;


} /* namespace xmrig */


#endif /* XMRIG_CPUTHREADCONFIG_H */
