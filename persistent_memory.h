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

#ifndef XMRIG_PERSISTENT_MEMORY_H
#define XMRIG_PERSISTENT_MEMORY_H


#include <stddef.h>


#include "algo/cryptonight/cryptonight.h"


enum memory_flags {
    MEMORY_HUGEPAGES_AVAILABLE = 1,
    MEMORY_HUGEPAGES_ENABLED   = 2,
    MEMORY_LOCK                = 4
};


#define MEMORY 2097152


extern char *persistent_memory;
extern int persistent_memory_flags;


const char * persistent_memory_allocate();
void persistent_memory_free();
void * persistent_calloc(size_t num, size_t size);
void create_cryptonight_ctx(struct cryptonight_ctx **ctx, int thr_id);


#endif /* XMRIG_PERSISTENT_MEMORY_H */
