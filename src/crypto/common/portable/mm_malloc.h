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

#ifndef XMRIG_MM_MALLOC_PORTABLE_H
#define XMRIG_MM_MALLOC_PORTABLE_H


#if (defined(XMRIG_ARM) || defined(XMRIG_RISCV)) && !defined(__clang__)
#include <stdlib.h>


#ifndef __cplusplus
extern
#else
extern "C"
#endif
int posix_memalign(void **__memptr, size_t __alignment, size_t __size);


static __inline__ void *__attribute__((__always_inline__, __malloc__)) _mm_malloc(size_t __size, size_t __align)
{
    if (__align == 1) {
        return malloc(__size);
    }

    if (!(__align & (__align - 1)) && __align < sizeof(void *)) {
        __align = sizeof(void *);
    }

    void *__mallocedMemory;
    if (posix_memalign(&__mallocedMemory, __align, __size)) {
        return nullptr;
    }

    return __mallocedMemory;
}


static __inline__ void __attribute__((__always_inline__)) _mm_free(void *__p)
{
    free(__p);
}
#elif defined(_WIN32) && !defined(__GNUC__)
#   include <malloc.h>
#else
#   include <mm_malloc.h>
#endif


#endif /* XMRIG_MM_MALLOC_PORTABLE_H */
