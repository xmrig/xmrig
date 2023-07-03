/* XMRig
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

#include "base/kernel/Platform.h"
#include "backend/cpu/Cpu.h"


#include <hwloc.h>
#include <thread>


#ifndef XMRIG_OS_APPLE
bool xmrig::Platform::setThreadAffinity(uint64_t cpu_id)
{
    auto topology = Cpu::info()->topology();
    auto pu       = hwloc_get_pu_obj_by_os_index(topology, static_cast<unsigned>(cpu_id));

    if (pu == nullptr) {
        return false;
    }

    if (hwloc_set_cpubind(topology, pu->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) >= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    }

    const bool result = (hwloc_set_cpubind(topology, pu->cpuset, HWLOC_CPUBIND_THREAD) >= 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    return result;
}
#endif
