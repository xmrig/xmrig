/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2018      Sebastian Stolzenberg <https://github.com/sebastianstolzenberg>
 * Copyright 2018      BenDroid    <ben@graef.in>
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

#ifndef __MEM_H__
#define __MEM_H__


#include <stddef.h>
#include <stdint.h>
#include <bitset>

#include "Options.h"

#ifdef _WIN32
#   ifdef __GNUC__
#       include <mm_malloc.h>
#   else
#       include <malloc.h>
#   endif
#else
#   if defined(XMRIG_ARM) && !defined(__clang__)
#       include "aligned_malloc.h"
#   else
#       include <mm_malloc.h>
#   endif
#endif

struct ScratchPad;

struct ScratchPadMem
{
    alignas(16) uint8_t *memory;

    size_t hugePages;
    size_t pages;
    size_t size;
    size_t realSize;
};


class Mem
{
public:
    typedef std::bitset<128> ThreadBitSet;

    enum Flags {
        HugepagesAvailable = 1,
        HugepagesEnabled   = 2,
        Lock               = 4
    };

    static void init(const Options* option);
    static ScratchPadMem create(ScratchPad** scratchPads, int threadId);
    static void release(ScratchPad** scratchPads, ScratchPadMem& scratchPadMem, int threadId);

    static void *allocateExecutableMemory(size_t size);
    static void flushInstructionCache(void *p, size_t size);

    static inline size_t hashFactor()         { return m_hashFactor; }
    static inline size_t getThreadHashFactor(int threadId)
    {
        size_t threadHashFactor = 1;
        if (Options::isCNAlgo(m_algo)) {
          threadHashFactor = (m_multiHashThreadMask.all() || m_multiHashThreadMask.test(threadId)) ? m_hashFactor : 1;
        }

        return threadHashFactor;
    }

    static inline bool isHugepagesAvailable() { return (m_flags & HugepagesAvailable) != 0; }
    static inline bool isHugepagesEnabled() { return (m_flags & HugepagesEnabled) != 0; }

    static inline int getTotalPages()            { return m_totalPages; }
    static inline int getTotalHugepages()        { return m_totalHugepages; }

private:
    static void allocate(ScratchPadMem& scratchPadMem, bool useHugePages);
    static void release(ScratchPadMem& scratchPadMem);

private:
    static bool m_useHugePages;
    static size_t m_hashFactor;
    static int m_flags;
    static int m_totalPages;
    static int m_totalHugepages;
    static Options::Algo m_algo;
    static ThreadBitSet m_multiHashThreadMask;
};


#endif /* __MEM_H__ */
