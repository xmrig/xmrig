/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <support@xmrig.com>
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

#ifndef XMRIG_HWLOCCPUINFO_H
#define XMRIG_HWLOCCPUINFO_H


#include "backend/cpu/platform/BasicCpuInfo.h"


typedef struct hwloc_obj *hwloc_obj_t;
typedef struct hwloc_topology *hwloc_topology_t;


namespace xmrig {


class HwlocCpuInfo : public BasicCpuInfo
{
public:
    enum Feature : uint32_t {
        SET_THISTHREAD_MEMBIND = 1
    };


    HwlocCpuInfo();
    ~HwlocCpuInfo() override;

    static inline bool has(Feature feature)                     { return m_features & feature; }
    static inline const std::vector<uint32_t> &nodeIndexes()    { return m_nodeIndexes; }

protected:
    CpuThreads threads(const Algorithm &algorithm) const override;

    inline const char *backend() const override     { return m_backend; }
    inline size_t cores() const override            { return m_cores; }
    inline size_t L2() const override               { return m_cache[2]; }
    inline size_t L3() const override               { return m_cache[3]; }
    inline size_t nodes() const override            { return m_nodes; }
    inline size_t packages() const override         { return m_packages; }

private:
    void processTopLevelCache(hwloc_obj_t obj, const Algorithm &algorithm, CpuThreads &threads) const;

    static std::vector<uint32_t> m_nodeIndexes;
    static uint32_t m_features;

    char m_backend[20];
    hwloc_topology_t m_topology;
    size_t m_cache[5];
    size_t m_cores      = 0;
    size_t m_nodes      = 0;
    size_t m_packages   = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_HWLOCCPUINFO_H */
