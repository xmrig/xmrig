/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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


#ifdef XMRIG_FEATURE_HWLOC
#   include <hwloc.h>
#   include "backend/cpu/platform/HwlocCpuInfo.h"
#endif


#include "base/io/log/Log.h"
#include "crypto/common/VirtualMemory.h"


void xmrig::VirtualMemory::bindToNUMANode(int64_t affinity)
{
#   ifdef XMRIG_FEATURE_HWLOC
    if (affinity < 0 || !HwlocCpuInfo::has(HwlocCpuInfo::SET_THISTHREAD_MEMBIND)) {
        return;
    }

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    const int depth     = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    const unsigned puId = static_cast<unsigned>(affinity);

    for (unsigned i = 0; i < hwloc_get_nbobjs_by_depth(topology, depth); i++) {
        hwloc_obj_t pu = hwloc_get_obj_by_depth(topology, depth, i);

        if (pu->os_index == puId) {
            if (hwloc_set_membind_nodeset(topology, pu->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD) < 0) {
                LOG_WARN("CPU #%02u warning: \"can't bind memory\"", puId);
            }

            break;
        }
    }

    hwloc_topology_destroy(topology);
#   endif
}
