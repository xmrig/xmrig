/* XMRig
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2023 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/common/VirtualMemory.h"
#include "backend/cpu/Cpu.h"
#include "base/io/log/Log.h"


#include <hwloc.h>


uint32_t xmrig::VirtualMemory::bindToNUMANode(int64_t affinity)
{
    if (affinity < 0 || Cpu::info()->nodes() < 2) {
        return 0;
    }

    auto pu = hwloc_get_pu_obj_by_os_index(Cpu::info()->topology(), static_cast<unsigned>(affinity));

    if (pu == nullptr) {
        LOG_WARN("CPU #%02" PRId64 " warning: \"can't bind memory: hwloc_get_pu_obj_by_os_index() failed\"", affinity);

        return 0;
    }
    if (Cpu::info()->membind(pu->nodeset) < 0) {
        LOG_WARN("CPU #%02" PRId64 " warning: \"can't bind memory: Cpu::info()->membind() failed\"", affinity);

        return 0;
    }

    return hwloc_bitmap_first(pu->nodeset);
}
