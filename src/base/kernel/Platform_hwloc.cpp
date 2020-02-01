/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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


#include "base/kernel/Platform.h"
#include "backend/cpu/platform/HwlocCpuInfo.h"
#include "backend/cpu/Cpu.h"


#include <hwloc.h>
#include <thread>


bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    auto cpu       = static_cast<HwlocCpuInfo *>(Cpu::info());
    hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(cpu->topology(), static_cast<unsigned>(cpu_id));

    if (pu == nullptr) {
        return false;
    }

    if (hwloc_set_cpubind(cpu->topology(), pu->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) >= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    }

    const bool result = (hwloc_set_cpubind(cpu->topology(), pu->cpuset, HWLOC_CPUBIND_THREAD) >= 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return result;
}
