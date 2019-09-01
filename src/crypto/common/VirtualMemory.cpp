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
#
#   if HWLOC_API_VERSION < 0x00010b00
#       define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#   endif
#endif


#include "base/io/log/Log.h"
#include "crypto/common/VirtualMemory.h"


uint32_t xmrig::VirtualMemory::bindToNUMANode(int64_t affinity)
{
#   ifdef XMRIG_FEATURE_HWLOC
    if (affinity < 0 || !HwlocCpuInfo::has(HwlocCpuInfo::SET_THISTHREAD_MEMBIND)) {
        return 0;
    }

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    const unsigned puId = static_cast<unsigned>(affinity);

    hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topology, puId);

#   if HWLOC_API_VERSION >= 0x20000
    if (pu == nullptr || hwloc_set_membind(topology, pu->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_BYNODESET) < 0) {
#   else
    if (pu == nullptr || hwloc_set_membind_nodeset(topology, pu->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD) < 0) {
#   endif
        LOG_WARN("CPU #%02u warning: \"can't bind memory\"", puId);
    }

    uint32_t nodeId = 0;

    if (pu) {
        hwloc_obj_t node = nullptr;

        while ((node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, node)) != nullptr) {
          if (hwloc_bitmap_intersects(node->cpuset, pu->cpuset)) {
              nodeId = node->os_index;

              break;
          }
        }
    }

    hwloc_topology_destroy(topology);

    return nodeId;
#   else
    return 0;
#   endif
}
