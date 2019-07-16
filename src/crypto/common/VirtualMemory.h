/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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

#ifndef XMRIG_VIRTUALMEMORY_H
#define XMRIG_VIRTUALMEMORY_H


#include <stddef.h>
#include <stdint.h>


namespace xmrig {


class VirtualMemory
{
public:
    inline VirtualMemory() {}
    VirtualMemory(size_t size, bool hugePages = true, size_t align = 64);
    ~VirtualMemory();

    inline bool isHugePages() const     { return m_flags & HUGEPAGES; }
    inline size_t hugePages() const     { return isHugePages() ? (align(size()) / 2097152) : 0; }
    inline size_t size() const          { return m_size; }
    inline uint8_t *scratchpad() const  { return m_scratchpad; }

    static void *allocateExecutableMemory(size_t size);
    static void *allocateLargePagesMemory(size_t size);
    static void flushInstructionCache(void *p, size_t size);
    static void freeLargePagesMemory(void *p, size_t size);
    static void protectExecutableMemory(void *p, size_t size);
    static void unprotectExecutableMemory(void *p, size_t size);

    static inline constexpr size_t align(size_t pos, size_t align = 2097152) { return ((pos - 1) / align + 1) * align; }

private:
    enum Flags {
        HUGEPAGES_AVAILABLE = 1,
        HUGEPAGES           = 2,
        LOCK                = 4
    };

    int m_flags             = 0;
    size_t m_size           = 0;
    uint8_t *m_scratchpad   = nullptr;
};


} /* namespace xmrig */



#endif /* XMRIG_VIRTUALMEMORY_H */
