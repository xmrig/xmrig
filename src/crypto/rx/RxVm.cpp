/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/randomx/randomx.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxVm.h"


#if defined(_M_X64) || defined(__x86_64__)
extern "C" uint32_t rx_blake2b_use_sse41;
#endif


randomx_vm* xmrig::RxVm::create(RxDataset *dataset, uint8_t *scratchpad, bool softAes, xmrig::Assembly assembly, uint32_t node)
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

    if (assembly == Assembly::AUTO) {
        assembly = Cpu::info()->assembly();
    }

    if ((assembly == Assembly::RYZEN) || (assembly == Assembly::BULLDOZER)) {
        flags |= RANDOMX_FLAG_AMD;
    }

#   if defined(_M_X64) || defined(__x86_64__)
    rx_blake2b_use_sse41 = Cpu::info()->has(ICpuInfo::FLAG_SSE41) ? 1 : 0;
#   endif

    return randomx_create_vm(static_cast<randomx_flags>(flags), dataset->cache() ? dataset->cache()->get() : nullptr, dataset->get(), scratchpad, node);
}


void xmrig::RxVm::destroy(randomx_vm* vm)
{
    if (vm) {
        randomx_destroy_vm(vm);
    }
}
