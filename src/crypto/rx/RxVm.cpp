/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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


#include "crypto/randomx/randomx.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxVm.h"


xmrig::RxVm::RxVm(RxDataset *dataset, uint8_t *scratchpad, bool softAes, xmrig::Assembly assembly)
{
    if (!softAes) {
       m_flags |= RANDOMX_FLAG_HARD_AES;
    }

    if (dataset->get()) {
        m_flags |= RANDOMX_FLAG_FULL_MEM;
    }

    if (!dataset->cache() || dataset->cache()->isJIT()) {
        m_flags |= RANDOMX_FLAG_JIT;
    }

    if ((assembly == Assembly::RYZEN) || ((assembly == Assembly::AUTO) && (Cpu::info()->assembly() == Assembly::RYZEN))) {
        m_flags |= RANDOMX_FLAG_RYZEN;
    }

    m_vm = randomx_create_vm(static_cast<randomx_flags>(m_flags), dataset->cache() ? dataset->cache()->get() : nullptr, dataset->get(), scratchpad);
}


xmrig::RxVm::~RxVm()
{
    if (m_vm) {
        randomx_destroy_vm(m_vm);
    }
}
