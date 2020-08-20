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


#include <cstdlib>
#include <sys/mman.h>


#include "backend/cpu/Cpu.h"
#include "crypto/common/portable/mm_malloc.h"
#include "crypto/common/VirtualMemory.h"


#if defined(__APPLE__)
#   include <mach/vm_statistics.h>
#endif


#if defined(XMRIG_OS_LINUX)
#   if (defined(MAP_HUGE_1GB) || defined(MAP_HUGE_SHIFT))
#       define XMRIG_HAS_1GB_PAGES
#   endif
#   include "crypto/common/LinuxMemory.h"
#endif


bool xmrig::VirtualMemory::isHugepagesAvailable()
{
    return true;
}


bool xmrig::VirtualMemory::isOneGbPagesAvailable()
{
#   ifdef XMRIG_HAS_1GB_PAGES
    return Cpu::info()->hasOneGbPages();
#   else
    return false;
#   endif
}


void *xmrig::VirtualMemory::allocateExecutableMemory(size_t size)
{
#   if defined(__APPLE__)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
#   else
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#   endif

    return mem == MAP_FAILED ? nullptr : mem;
}


void *xmrig::VirtualMemory::allocateLargePagesMemory(size_t size)
{
#   if defined(__APPLE__)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
#   elif defined(__FreeBSD__)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER | MAP_PREFAULT_READ, -1, 0);
#   else

#   if defined(MAP_HUGE_2MB)
    constexpr int flag_2mb = MAP_HUGE_2MB;
#   elif defined(MAP_HUGE_SHIFT)
    constexpr int flag_2mb = (21 << MAP_HUGE_SHIFT);
#   else
    constexpr int flag_2mb = 0;
#   endif

    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | flag_2mb, 0, 0);

#   endif

    return mem == MAP_FAILED ? nullptr : mem;
}


void *xmrig::VirtualMemory::allocateOneGbPagesMemory(size_t size)
{
#   ifdef XMRIG_HAS_1GB_PAGES
    if (isOneGbPagesAvailable()) {
#       if defined(MAP_HUGE_1GB)
        constexpr int flag_1gb = MAP_HUGE_1GB;
#       elif defined(MAP_HUGE_SHIFT)
        constexpr int flag_1gb = (30 << MAP_HUGE_SHIFT);
#       else
        constexpr int flag_1gb = 0;
#       endif

        void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | flag_1gb, 0, 0);

        return mem == MAP_FAILED ? nullptr : mem;
    }
#   endif

    return nullptr;
}


void xmrig::VirtualMemory::flushInstructionCache(void *p, size_t size)
{
#   ifdef HAVE_BUILTIN_CLEAR_CACHE
    __builtin___clear_cache(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + size);
#   endif
}


void xmrig::VirtualMemory::freeLargePagesMemory(void *p, size_t size)
{
    munmap(p, size);
}


void xmrig::VirtualMemory::protectExecutableMemory(void *p, size_t size)
{
    mprotect(p, size, PROT_READ | PROT_EXEC);
}


void xmrig::VirtualMemory::unprotectExecutableMemory(void *p, size_t size)
{
    mprotect(p, size, PROT_WRITE | PROT_EXEC);
}


void xmrig::VirtualMemory::osInit(bool)
{
}


bool xmrig::VirtualMemory::allocateLargePagesMemory()
{
#   if defined(XMRIG_OS_LINUX)
    LinuxMemory::reserve(m_size, m_node);
#   endif

    m_scratchpad = static_cast<uint8_t*>(allocateLargePagesMemory(m_size));
    if (m_scratchpad) {
        m_flags.set(FLAG_HUGEPAGES, true);

        madvise(m_scratchpad, m_size, MADV_RANDOM | MADV_WILLNEED);

        if (mlock(m_scratchpad, m_size) == 0) {
            m_flags.set(FLAG_LOCK, true);
        }

        return true;
    }

    return false;
}


bool xmrig::VirtualMemory::allocateOneGbPagesMemory()
{
#   if defined(XMRIG_HAS_1GB_PAGES)
    LinuxMemory::reserve(m_size, m_node, true);
#   endif

    m_scratchpad = static_cast<uint8_t*>(allocateOneGbPagesMemory(m_size));
    if (m_scratchpad) {
        m_flags.set(FLAG_1GB_PAGES, true);

        madvise(m_scratchpad, m_size, MADV_RANDOM | MADV_WILLNEED);

        if (mlock(m_scratchpad, m_size) == 0) {
            m_flags.set(FLAG_LOCK, true);
        }

        return true;
    }

    return false;
}


void xmrig::VirtualMemory::freeLargePagesMemory()
{
    if (m_flags.test(FLAG_LOCK)) {
        munlock(m_scratchpad, m_size);
    }

    freeLargePagesMemory(m_scratchpad, m_size);
}
