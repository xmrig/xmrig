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


#include <winsock2.h>
#include <windows.h>


#include "crypto/common/VirtualMemory.h"


namespace xmrig {

constexpr size_t align(size_t pos, size_t align) {
    return ((pos - 1) / align + 1) * align;
}

}


void *xmrig::VirtualMemory::allocateExecutableMemory(size_t size)
{
    return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
}


void *xmrig::VirtualMemory::allocateLargePagesMemory(size_t size)
{
    const size_t min = GetLargePageMinimum();
    void *mem        = nullptr;

    if (min > 0) {
        mem = VirtualAlloc(nullptr, align(size, min), MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
    }

    return mem;
}


void xmrig::VirtualMemory::flushInstructionCache(void *p, size_t size)
{
    ::FlushInstructionCache(GetCurrentProcess(), p, size);
}


void xmrig::VirtualMemory::freeLargePagesMemory(void *p, size_t)
{
    VirtualFree(p, 0, MEM_RELEASE);
}


void xmrig::VirtualMemory::protectExecutableMemory(void *p, size_t size)
{
    DWORD oldProtect;
    VirtualProtect(p, size, PAGE_EXECUTE_READ, &oldProtect);
}


void xmrig::VirtualMemory::unprotectExecutableMemory(void *p, size_t size)
{
    DWORD oldProtect;
    VirtualProtect(p, size, PAGE_EXECUTE_READWRITE, &oldProtect);
}
