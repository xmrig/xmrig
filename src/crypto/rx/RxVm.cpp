/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "crypto/randomx/randomx.h"
#include "backend/cpu/Cpu.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxVm.h"


randomx_vm *xmrig::RxVm::create(RxDataset *dataset, uint8_t *scratchpad, bool softAes, const Assembly &assembly, uint32_t node)
{
    int flags = 0;

    if (!softAes) {
       flags |= RANDOMX_FLAG_HARD_AES;
    }

    if (dataset->get()) {
        flags |= RANDOMX_FLAG_FULL_MEM;
    }

    if (!dataset->cache() || dataset->cache()->isJIT()) {
        flags |= RANDOMX_FLAG_JIT;
    }

    const auto asmId = assembly == Assembly::AUTO ? Cpu::info()->assembly() : assembly.id();
    if ((asmId == Assembly::RYZEN) || (asmId == Assembly::BULLDOZER)) {
        flags |= RANDOMX_FLAG_AMD;
    }

    return randomx_create_vm(static_cast<randomx_flags>(flags), !dataset->get() ? dataset->cache()->get() : nullptr, dataset->get(), scratchpad, node);
}


void xmrig::RxVm::destroy(randomx_vm* vm)
{
    if (vm) {
        randomx_destroy_vm(vm);
    }
}
