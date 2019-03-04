/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#include <stdlib.h>
#include <mm_malloc.h>
#include <sys/mman.h>
#include <mach/vm_statistics.h>

#include "persistent_memory.h"
#include "options.h"
#include "utils/applog.h"

char *persistent_memory;
int persistent_memory_flags = 0;


const char * persistent_memory_allocate() {
    const int ratio = (opt_double_hash && opt_algo != ALGO_CRYPTONIGHT_LITE) ? 2 : 1;
    const int size = MEMORY * (opt_n_threads * ratio + 1);
    persistent_memory_flags |= MEMORY_HUGEPAGES_AVAILABLE;
    
    persistent_memory = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
    
    if (persistent_memory == MAP_FAILED) {
        persistent_memory = _mm_malloc(size, 16);
        return persistent_memory;
    }
    
    persistent_memory_flags |= MEMORY_HUGEPAGES_ENABLED;
    
    if (madvise(persistent_memory, size, MADV_RANDOM | MADV_WILLNEED) != 0) {
        applog(LOG_ERR, "madvise failed");
    }
    
    if (mlock(persistent_memory, size) == 0) {
        persistent_memory_flags |= MEMORY_LOCK;
    }
    
    return persistent_memory;
}


void persistent_memory_free() {
    const int size = MEMORY * (opt_n_threads + 1);
    
    if (persistent_memory_flags & MEMORY_HUGEPAGES_ENABLED) {
        if (persistent_memory_flags & MEMORY_LOCK) {
            munlock(persistent_memory, size);
        }
        
        munmap(persistent_memory, size);
    }
    else {
        _mm_free(persistent_memory);
    }
}


void *allocate_executable_memory(size_t size)
{
    return mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
}


void protect_executable_memory(void *p, size_t size)
{
    mprotect(p, size, PROT_READ | PROT_EXEC);
}


void flush_instruction_cache(void *p, size_t size)
{
    __builtin___clear_cache((char*) p, (char*)(p) + size);
}
