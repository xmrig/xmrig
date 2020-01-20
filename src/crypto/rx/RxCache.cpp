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


#include "crypto/rx/RxCache.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/randomx/randomx.h"


static_assert(RANDOMX_FLAG_JIT == 8, "RANDOMX_FLAG_JIT flag mismatch");


xmrig::RxCache::RxCache(bool hugePages, uint32_t nodeId)
{
    m_memory = new VirtualMemory(maxSize(), hugePages, false, false, nodeId);

    create(m_memory->raw());
}


xmrig::RxCache::RxCache(uint8_t *memory)
{
    create(memory);
}


xmrig::RxCache::~RxCache()
{
    randomx_release_cache(m_cache);

    delete m_memory;
}


bool xmrig::RxCache::init(const Buffer &seed)
{
    if (m_seed == seed) {
        return false;
    }

    m_seed = seed;

    if (m_cache) {
        randomx_init_cache(m_cache, m_seed.data(), m_seed.size());

        return true;
    }

    return false;
}


xmrig::HugePagesInfo xmrig::RxCache::hugePages() const
{
    return m_memory ? m_memory->hugePages() : HugePagesInfo();
}


void xmrig::RxCache::create(uint8_t *memory)
{
    if (!memory) {
        return;
    }

    m_cache = randomx_create_cache(RANDOMX_FLAG_JIT, memory);

    if (!m_cache) {
        m_jit   = false;
        m_cache = randomx_create_cache(RANDOMX_FLAG_DEFAULT, memory);
    }
}
