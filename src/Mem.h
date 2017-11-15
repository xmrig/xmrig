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

#ifndef __MEM_H__
#define __MEM_H__


#include <stddef.h>
#include <stdint.h>


#include "align.h"


struct cryptonight_ctx;


class Mem
{
public:
    enum Flags {
        HugepagesAvailable = 1,
        HugepagesEnabled   = 2,
        Lock               = 4
    };

    static bool allocate(int algo, int threads, bool doubleHash, bool enabled);
    static cryptonight_ctx *create(int threadId);
    static void *calloc(size_t num, size_t size);
    static void release();

    static inline bool isDoubleHash()         { return m_doubleHash; }
    static inline bool isHugepagesAvailable() { return (m_flags & HugepagesAvailable) != 0; }
    static inline bool isHugepagesEnabled()   { return (m_flags & HugepagesEnabled) != 0; }
    static inline int flags()                 { return m_flags; }
    static inline int threads()               { return m_threads; }

private:
    static bool m_doubleHash;
    static int m_algo;
    static int m_flags;
    static int m_threads;
    static size_t m_offset;
    VAR_ALIGN(16, static uint8_t *m_memory);

#   ifndef XMRIG_NO_AEON
    static cryptonight_ctx *createLite(int threadId);
#   endif
};


#endif /* __MEM_H__ */
