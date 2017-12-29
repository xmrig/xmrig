/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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


bool Mem::allocate(const Options* options)
{
    m_algo       = options->algo();
    m_threads    = options->threads();
    m_doubleHash = options->doubleHash();
    m_doubleHashThreadMask = options->doubleHashThreadMask();
    m_memorySize = 0;

    size_t scratchPadSize = m_algo == Options::ALGO_CRYPTONIGHT ? MEMORY : MEMORY_LITE;
    for (int i=0; i < m_threads; i++) {
        m_memorySize += sizeof(cryptonight_ctx);

        if (isDoubleHash(i)) {
            m_memorySize += scratchPadSize*2;
        } else {
            m_memorySize += scratchPadSize;
        }
    }

    if (!options->hugePages()) {
        m_memory = static_cast<uint8_t*>(_mm_malloc(m_memorySize, 16));
        return true;
    }

    m_flags |= HugepagesAvailable;

#   if defined(__APPLE__)
    m_memory = static_cast<uint8_t*>(mmap(0, m_memorySize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0));
#   elif defined(__FreeBSD__)
    m_memory = static_cast<uint8_t*>(mmap(0, m_memorySize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER | MAP_PREFAULT_READ, -1, 0));
#   else
    m_memory = static_cast<uint8_t*>(mmap(0, m_memorySize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, 0, 0));
#   endif
    if (m_memory == MAP_FAILED) {
        m_memory = static_cast<uint8_t*>(_mm_malloc(m_memorySize, 16));
        return true;
    }

    m_flags |= HugepagesEnabled;

    if (madvise(m_memory, m_memorySize, MADV_RANDOM | MADV_WILLNEED) != 0) {
        LOG_ERR("madvise failed");
    }

    if (mlock(m_memory, m_memorySize) == 0) {
        m_flags |= Lock;
    }

    return true;
}


void Mem::release()
{
    if (m_flags & HugepagesEnabled) {
        if (m_flags & Lock) {
            munlock(m_memory, m_memorySize);
        }

        munmap(m_memory, m_memorySize);
    }
    else {
        _mm_free(m_memory);
    }
}
