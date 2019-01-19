/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef XMRIG_MEM_H
#define XMRIG_MEM_H


#include <stddef.h>
#include <stdint.h>


#include "common/xmrig.h"


struct cryptonight_ctx;


struct MemInfo
{
    alignas(16) uint8_t *memory;

    size_t hugePages;
    size_t pages;
    size_t size;
};


class Mem
{
public:
    enum Flags {
        HugepagesAvailable = 1,
        HugepagesEnabled   = 2,
        Lock               = 4
    };

    static MemInfo create(cryptonight_ctx **ctx, xmrig::Algo algorithm, size_t count);
    static void init(bool enabled);
    static void release(cryptonight_ctx **ctx, size_t count, MemInfo &info);

    static void *allocateExecutableMemory(size_t size);
    static void protectExecutableMemory(void *p, size_t size);
    static void flushInstructionCache(void *p, size_t size);

    static inline bool isHugepagesAvailable() { return (m_flags & HugepagesAvailable) != 0; }

private:
    static void allocate(MemInfo &info, bool enabled);
    static void release(MemInfo &info);

    static int m_flags;
    static bool m_enabled;
};


#endif /* XMRIG_MEM_H */
