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


#include <cstddef>
#include <cstdint>
#include <bitset>

#include "align.h"
#include "Options.h"

struct cryptonight_ctx;


class Mem
{
public:
    typedef std::bitset<64> ThreadBitSet;
    enum Flags {
        HugepagesAvailable = 1,
        HugepagesEnabled   = 2,
        Lock               = 4
    };

    static bool allocate(const Options* options);
    static cryptonight_ctx *create(int threadId);
    static void release();

    static inline size_t hashFactor()         { return m_hashFactor; }
    static inline size_t getThreadHashFactor(int threadId)
    {
        return (m_multiHashThreadMask.all() ||
                m_multiHashThreadMask.test(threadId)) ? m_hashFactor : 1;
    }
    static inline bool isHugepagesAvailable() { return (m_flags & HugepagesAvailable) != 0; }
    static inline bool isHugepagesEnabled()   { return (m_flags & HugepagesEnabled) != 0; }
    static inline int flags()                 { return m_flags; }
    static inline size_t threads()            { return m_threads; }

private:
    static size_t m_hashFactor;
    static size_t m_threads;
    static int m_algo;
    static int m_flags;
    static ThreadBitSet m_multiHashThreadMask;
    static size_t m_memorySize;
    VAR_ALIGN(16, static uint8_t *m_memory);
};


#endif /* __MEM_H__ */
