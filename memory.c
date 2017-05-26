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

#include <string.h>

#include "persistent_memory.h"
#include "algo/cryptonight/cryptonight.h"
#include "options.h"

static size_t offset = 0;


#ifndef XMRIG_NO_AEON
static void * create_persistent_ctx_lite(int thr_id) {
    struct cryptonight_ctx *ctx = NULL;

    if (!opt_double_hash) {
        const size_t offset = MEMORY * (thr_id + 1);

        ctx = (struct cryptonight_ctx *) &persistent_memory[offset + MEMORY_LITE];
        ctx->memory = (uint8_t*) &persistent_memory[offset];
        return ctx;
    }

    ctx = (struct cryptonight_ctx *) &persistent_memory[MEMORY - sizeof(struct cryptonight_ctx) * (thr_id + 1)];
    ctx->memory = (uint8_t*) &persistent_memory[MEMORY * (thr_id + 1)];

    return ctx;
}
#endif


void * persistent_calloc(size_t num, size_t size) {
    void *mem = &persistent_memory[offset];
    offset += (num * size);

    memset(mem, 0, num * size);

    return mem;
}


void * create_persistent_ctx(int thr_id) {
#   ifndef XMRIG_NO_AEON
    if (opt_algo == ALGO_CRYPTONIGHT_LITE) {
        return create_persistent_ctx_lite(thr_id);
    }
#   endif

    struct cryptonight_ctx *ctx = (struct cryptonight_ctx *) &persistent_memory[MEMORY - sizeof(struct cryptonight_ctx) * (thr_id + 1)];

    const int ratio = opt_double_hash ? 2 : 1;
    ctx->memory = (uint8_t*) &persistent_memory[MEMORY * (thr_id * ratio + 1)];

    return ctx;
}
