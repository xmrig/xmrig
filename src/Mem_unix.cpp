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


#include <cstdlib>
#include <sys/mman.h>

#include "crypto/CryptoNight.h"
#include "log/Log.h"
#include "Mem.h"


void Mem::init(const Options* options)
{
    m_hashFactor = options->hashFactor();
    m_useHugePages = options->hugePages();
    m_algo = options->algo();
    m_multiHashThreadMask = Mem::ThreadBitSet(static_cast<unsigned long long int>(options->multiHashThreadMask()));
}

void Mem::allocate(ScratchPadMem& scratchPadMem, bool useHugePages)
{
    scratchPadMem.hugePages = 0;

    if (!useHugePages) {
        scratchPadMem.memory = static_cast<uint8_t*>(_mm_malloc(scratchPadMem.size, 4096));
        return;
    }

#   if defined(__APPLE__)
    scratchPadMem.memory = static_cast<uint8_t*>(mmap(0, scratchPadMem.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0));
#   elif defined(__FreeBSD__)
    scratchPadMem.memory = static_cast<uint8_t*>(mmap(0, scratchPadMem.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER | MAP_PREFAULT_READ, -1, 0));
#   else
    scratchPadMem.memory = static_cast<uint8_t*>(mmap(0, scratchPadMem.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, 0, 0));
#   endif

    if (scratchPadMem.memory == MAP_FAILED) {
        return allocate(scratchPadMem, false);
    }

    scratchPadMem.hugePages = scratchPadMem.pages;

    m_flags |= HugepagesAvailable;
    m_flags |= HugepagesEnabled;

    if (madvise(scratchPadMem.memory, scratchPadMem.size, MADV_RANDOM | MADV_WILLNEED) != 0) {
        LOG_ERR("madvise failed");
    }

    if (mlock(scratchPadMem.memory, scratchPadMem.size) == 0) {
        m_flags |= Lock;
    }
}

void Mem::release(ScratchPadMem &scratchPadMem)
{
    if (scratchPadMem.hugePages) {
        if (m_flags & Lock) {
            munlock(scratchPadMem.memory, scratchPadMem.size);
        }

        munmap(scratchPadMem.memory, scratchPadMem.size);
    }
    else {
        _mm_free(scratchPadMem.memory);
    }
}
