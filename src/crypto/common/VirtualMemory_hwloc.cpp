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


#include "crypto/common/VirtualMemory.h"
#include "backend/cpu/Cpu.h"
#include "backend/cpu/platform/HwlocCpuInfo.h"
#include "base/io/log/Log.h"


#include <hwloc.h>


uint32_t xmrig::VirtualMemory::bindToNUMANode(int64_t affinity)
{
    if (affinity < 0 || Cpu::info()->nodes() < 2) {
        return 0;
    }

    auto cpu       = static_cast<HwlocCpuInfo *>(Cpu::info());
    hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(cpu->topology(), static_cast<unsigned>(affinity));

    if (pu == nullptr || !cpu->membind(pu->nodeset)) {
        LOG_WARN("CPU #%02" PRId64 " warning: \"can't bind memory\"", affinity);

        return 0;
    }

    return hwloc_bitmap_first(pu->nodeset);
}
