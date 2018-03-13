/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <stdlib.h>
#include <sys/mman.h>


#if defined(XMRIG_ARM) && !defined(__clang__)
#   include "aligned_malloc.h"
#else
#   include <mm_malloc.h>
#endif


#include "crypto/CryptoNight.h"
#include "log/Log.h"
#include "Mem.h"
#include "Options.h"
#include "xmrig.h"


bool Mem::allocate(int algo, int threads, bool doubleHash, bool enabled)
{
    m_algo       = algo;
    m_threads    = threads;
    m_doubleHash = doubleHash;

    const int ratio = (doubleHash && algo != xmrig::ALGO_CRYPTONIGHT_LITE) ? 2 : 1;
    m_size          = MONERO_MEMORY * (threads * ratio + 1);

    if (!enabled) {
        m_memory = static_cast<uint8_t*>(_mm_malloc(size, 16));
        return true;
    }

    m_flags |= HugepagesAvailable;

#   if defined(__APPLE__)
    m_memory = static_cast<uint8_t*>(mmap(0, m_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0));
#   elif defined(__FreeBSD__)
    m_memory = static_cast<uint8_t*>(mmap(0, m_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER | MAP_PREFAULT_READ, -1, 0));
#   else
    m_memory = static_cast<uint8_t*>(mmap(0, m_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, 0, 0));
#   endif
    if (m_memory == MAP_FAILED) {
        m_memory = static_cast<uint8_t*>(_mm_malloc(m_size, 16));
        return true;
    }

    m_flags |= HugepagesEnabled;

    if (madvise(m_memory, m_size, MADV_RANDOM | MADV_WILLNEED) != 0) {
        LOG_ERR("madvise failed");
    }

    if (mlock(m_memory, m_size) == 0) {
        m_flags |= Lock;
    }

    return true;
}


void Mem::release()
{
    if (m_flags & HugepagesEnabled) {
        if (m_flags & Lock) {
            munlock(m_memory, m_size);
        }

        munmap(m_memory, m_size);
    }
    else {
        _mm_free(m_memory);
    }
}
