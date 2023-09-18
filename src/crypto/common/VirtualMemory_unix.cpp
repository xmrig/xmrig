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


#include "crypto/common/VirtualMemory.h"
#include "backend/cpu/Cpu.h"
#include "crypto/common/portable/mm_malloc.h"


#include <cmath>
#include <cstdlib>
#include <sys/mman.h>


#ifdef XMRIG_OS_APPLE
#   include <libkern/OSCacheControl.h>
#   include <mach/vm_statistics.h>
#   include <pthread.h>
#   include <TargetConditionals.h>
#   ifdef XMRIG_ARM
#       define MEXTRA MAP_JIT
#   else
#       define MEXTRA 0
#   endif
#else
#   define MEXTRA 0
#endif


#ifdef XMRIG_OS_LINUX
#   include "crypto/common/LinuxMemory.h"
#endif


#ifndef MAP_HUGE_SHIFT
#   define MAP_HUGE_SHIFT 26
#endif


#ifndef MAP_HUGE_MASK
#   define MAP_HUGE_MASK 0x3f
#endif

#ifdef XMRIG_OS_FREEBSD
#   ifndef MAP_ALIGNED_SUPER
#       define MAP_ALIGNED_SUPER 0
#   endif
#   ifndef MAP_PREFAULT_READ
#       define MAP_PREFAULT_READ 0
#   endif
#endif


#ifdef XMRIG_SECURE_JIT
#   define SECURE_PROT_EXEC 0
#else
#   define SECURE_PROT_EXEC PROT_EXEC
#endif


#if defined(XMRIG_OS_LINUX) || (!defined(XMRIG_OS_APPLE) && !defined(XMRIG_OS_FREEBSD))
static inline int hugePagesFlag(size_t size)
{
    return (static_cast<int>(log2(size)) & MAP_HUGE_MASK) << MAP_HUGE_SHIFT;
}
#endif


bool xmrig::VirtualMemory::isHugepagesAvailable()
{
#   if defined(XMRIG_OS_MACOS) && defined(XMRIG_ARM)
    return false;
#   else
    return true;
#   endif
}


bool xmrig::VirtualMemory::isOneGbPagesAvailable()
{
#   ifdef XMRIG_OS_LINUX
    return Cpu::info()->hasOneGbPages();
#   else
    return false;
#   endif
}


bool xmrig::VirtualMemory::protectRW(void *p, size_t size)
{
#   if defined(XMRIG_OS_APPLE) && defined(XMRIG_ARM)
    pthread_jit_write_protect_np(false);
    return true;
#   else
    return mprotect(p, size, PROT_READ | PROT_WRITE) == 0;
#   endif
}


bool xmrig::VirtualMemory::protectRWX(void *p, size_t size)
{
    return mprotect(p, size, PROT_READ | PROT_WRITE | PROT_EXEC) == 0;
}


bool xmrig::VirtualMemory::protectRX(void *p, size_t size)
{
    bool result = true;

#   if defined(XMRIG_OS_APPLE) && defined(XMRIG_ARM)
    pthread_jit_write_protect_np(true);
#   else
    result = (mprotect(p, size, PROT_READ | PROT_EXEC) == 0);
#   endif

#   if defined(XMRIG_ARM)
    flushInstructionCache(p, size);
#   endif

    return result;
}


void *xmrig::VirtualMemory::allocateExecutableMemory(size_t size, bool hugePages)
{
#   if defined(XMRIG_OS_APPLE)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MEXTRA, -1, 0);
#   ifdef XMRIG_ARM
    pthread_jit_write_protect_np(false);
#   endif
#   elif defined(XMRIG_OS_FREEBSD)
    void *mem = nullptr;

    if (hugePages) {
        mem = mmap(0, size, PROT_READ | PROT_WRITE | SECURE_PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER | MAP_PREFAULT_READ, -1, 0);
    }

    if (!mem) {
        mem = mmap(0, size, PROT_READ | PROT_WRITE | SECURE_PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }

#   else

    void *mem = nullptr;

    if (hugePages) {
        mem = mmap(0, align(size), PROT_READ | PROT_WRITE | SECURE_PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | hugePagesFlag(hugePageSize()), -1, 0);
    }

    if (!mem) {
        mem = mmap(0, size, PROT_READ | PROT_WRITE | SECURE_PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }

#   endif

    return mem == MAP_FAILED ? nullptr : mem;
}


void *xmrig::VirtualMemory::allocateLargePagesMemory(size_t size)
{
#   if defined(XMRIG_OS_APPLE)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
#   elif defined(XMRIG_OS_FREEBSD)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER | MAP_PREFAULT_READ, -1, 0);
#   else
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | hugePagesFlag(hugePageSize()), 0, 0);
#   endif

    return mem == MAP_FAILED ? nullptr : mem;
}


void *xmrig::VirtualMemory::allocateOneGbPagesMemory(size_t size)
{
#   ifdef XMRIG_OS_LINUX
    if (isOneGbPagesAvailable()) {
        void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | hugePagesFlag(kOneGiB), 0, 0);

        return mem == MAP_FAILED ? nullptr : mem;
    }
#   endif

    return nullptr;
}


void xmrig::VirtualMemory::flushInstructionCache(void *p, size_t size)
{
#   if defined(XMRIG_OS_APPLE)
    sys_icache_invalidate(p, size);
#   elif defined (HAVE_BUILTIN_CLEAR_CACHE) || defined (__GNUC__)
    __builtin___clear_cache(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + size);
#   endif
}


void xmrig::VirtualMemory::freeLargePagesMemory(void *p, size_t size)
{
    munmap(p, size);
}


void xmrig::VirtualMemory::osInit(size_t hugePageSize)
{
    if (hugePageSize) {
        m_hugePageSize = hugePageSize;
    }
}


bool xmrig::VirtualMemory::allocateLargePagesMemory()
{
#   ifdef XMRIG_OS_LINUX
    LinuxMemory::reserve(m_size, m_node, hugePageSize());
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
#   ifdef XMRIG_OS_LINUX
    LinuxMemory::reserve(m_size, m_node, kOneGiB);
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
