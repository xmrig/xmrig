/* XMRig
 * Copyright (c) 2018-2020 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/tools/Object.h"
#include "crypto/common/HugePagesInfo.h"


#include <bitset>
#include <cstddef>
#include <cstdint>
#include <utility>


namespace xmrig {


class VirtualMemory
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(VirtualMemory)

    constexpr static size_t kDefaultHugePageSize    = 2U * 1024U * 1024U;
    constexpr static size_t kOneGiB                 = 1024U * 1024U * 1024U;

    VirtualMemory(size_t size, bool hugePages, bool oneGbPages, bool usePool, uint32_t node = 0, size_t alignSize = 64);
    ~VirtualMemory();

    inline bool isHugePages() const                                 { return m_flags.test(FLAG_HUGEPAGES); }
    inline bool isOneGbPages() const                                { return m_flags.test(FLAG_1GB_PAGES); }
    inline size_t size() const                                      { return m_size; }
    inline size_t capacity() const                                  { return m_capacity; }
    inline uint8_t *raw() const                                     { return m_scratchpad; }
    inline uint8_t *scratchpad() const                              { return m_scratchpad; }

    inline static void flushInstructionCache(void *p1, void *p2)    { flushInstructionCache(p1, static_cast<uint8_t*>(p2) - static_cast<uint8_t*>(p1)); }

    HugePagesInfo hugePages() const;

    static bool isHugepagesAvailable();
    static bool isOneGbPagesAvailable();
    static bool protectRW(void *p, size_t size);
    static bool protectRWX(void *p, size_t size);
    static bool protectRX(void *p, size_t size);
    static uint32_t bindToNUMANode(int64_t affinity);
    static void *allocateExecutableMemory(size_t size, bool hugePages);
    static void *allocateLargePagesMemory(size_t size);
    static void *allocateOneGbPagesMemory(size_t size);
    static void destroy();
    static void flushInstructionCache(void *p, size_t size);
    static void freeLargePagesMemory(void *p, size_t size);
    static void init(size_t poolSize, size_t hugePageSize);

    static inline constexpr size_t align(size_t pos, size_t align = kDefaultHugePageSize)   { return ((pos - 1) / align + 1) * align; }
    static inline size_t alignToHugePageSize(size_t pos)                                    { return align(pos, hugePageSize()); }
    static inline size_t hugePageSize()                                                     { return m_hugePageSize; }

private:
    enum Flags {
        FLAG_HUGEPAGES,
        FLAG_1GB_PAGES,
        FLAG_LOCK,
        FLAG_EXTERNAL,
        FLAG_MAX
    };

    static void osInit(size_t hugePageSize);

    bool allocateLargePagesMemory();
    bool allocateOneGbPagesMemory();
    void freeLargePagesMemory();

    static size_t m_hugePageSize;

    const size_t m_size;
    const uint32_t m_node;
    size_t m_capacity;
    std::bitset<FLAG_MAX> m_flags;
    uint8_t *m_scratchpad = nullptr;
};


} /* namespace xmrig */



#endif /* XMRIG_VIRTUALMEMORY_H */
