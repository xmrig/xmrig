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


static_assert(RANDOMX_FLAG_JIT == 8,         "RANDOMX_FLAG_JIT flag mismatch");
static_assert(RANDOMX_FLAG_LARGE_PAGES == 1, "RANDOMX_FLAG_LARGE_PAGES flag mismatch");



xmrig::RxCache::RxCache(bool hugePages) :
    m_seed()
{
    if (hugePages) {
        m_flags = RANDOMX_FLAG_JIT | RANDOMX_FLAG_LARGE_PAGES;
        m_cache = randomx_alloc_cache(static_cast<randomx_flags>(m_flags));
    }

    if (!m_cache) {
        m_flags = RANDOMX_FLAG_JIT;
        m_cache = randomx_alloc_cache(static_cast<randomx_flags>(m_flags));
    }

    if (!m_cache) {
        m_flags = RANDOMX_FLAG_DEFAULT;
        m_cache = randomx_alloc_cache(static_cast<randomx_flags>(m_flags));
    }
}


xmrig::RxCache::~RxCache()
{
    if (m_cache) {
        randomx_release_cache(m_cache);
    }
}


bool xmrig::RxCache::init(const void *seed)
{
    if (isReady(seed)) {
        return false;
    }

    memcpy(m_seed, seed, sizeof(m_seed));
    randomx_init_cache(m_cache, m_seed, sizeof(m_seed));

    return true;
}


bool xmrig::RxCache::isReady(const void *seed) const
{
    return memcmp(m_seed, seed, sizeof(m_seed)) == 0;
}
