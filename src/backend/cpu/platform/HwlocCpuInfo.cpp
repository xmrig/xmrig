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


#include <hwloc.h>


#include "backend/cpu/platform/HwlocCpuInfo.h"


namespace xmrig {


inline bool isCacheObject(hwloc_obj_t obj)
{
#   if HWLOC_API_VERSION >= 0x20000
    return hwloc_obj_type_is_cache(obj->type);
#   else
    return obj->type == HWLOC_OBJ_CACHE;
#   endif
}


template <typename func>
inline void findCache(hwloc_obj_t obj, func lambda)
{
    for (size_t i = 0; i < obj->arity; i++) {
        if (isCacheObject(obj->children[i])) {
            if (obj->children[i]->attr->cache.depth < 2) {
                continue;
            }

            lambda(obj->children[i]);
        }

        findCache(obj->children[i], lambda);
    }
}


inline size_t countByType(hwloc_topology_t topology, hwloc_obj_type_t type)
{
    const int count = hwloc_get_nbobjs_by_type(topology, type);

    return count > 0 ? static_cast<size_t>(count) : 0;
}


} // namespace xmrig


xmrig::HwlocCpuInfo::HwlocCpuInfo() : BasicCpuInfo(),
    m_cache()
{
    m_threads = 0;

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    findCache(hwloc_get_root_obj(topology), [this](hwloc_obj_t found) { this->m_cache[found->attr->cache.depth] += found->attr->cache.size; });

    m_threads   = countByType(topology, HWLOC_OBJ_PU);
    m_cores     = countByType(topology, HWLOC_OBJ_CORE);
    m_nodes     = countByType(topology, HWLOC_OBJ_NUMANODE);
    m_packages  = countByType(topology, HWLOC_OBJ_PACKAGE);

    hwloc_topology_destroy(topology);
}
