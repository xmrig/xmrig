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

#include <string.h>

#include "persistent_memory.h"
#include "options.h"


static size_t offset = 0;


void * persistent_calloc(size_t num, size_t size) {
    size += size % 16;

    void *mem = &persistent_memory[offset];
    offset += (num * size);

    memset(mem, 0, num * size);

    return mem;
}


void init_cn_r(struct cryptonight_ctx *ctx)
{
    uint8_t *p = allocate_executable_memory(0x4000);

    ctx->generated_code        = (cn_mainloop_fun_ms_abi) p;
    ctx->generated_code_double = (cn_mainloop_double_fun_ms_abi)(p + 0x2000);
    ctx->generated_code_height = ctx->generated_code_double_height = (uint64_t)(-1);
    ctx->height                = 0;
}


void create_cryptonight_ctx(struct cryptonight_ctx **ctx, int thr_id)
{
    const int ratio = (opt_double_hash && opt_algo == ALGO_CRYPTONIGHT) ? 2 : 1;
    ctx[0]          = persistent_calloc(1, sizeof(struct cryptonight_ctx));
    ctx[0]->memory  = &persistent_memory[MEMORY * (thr_id * ratio + 1)];

    init_cn_r(ctx[0]);

    if (opt_double_hash) {
        ctx[1]         = persistent_calloc(1, sizeof(struct cryptonight_ctx));
        ctx[1]->memory = ctx[0]->memory + (opt_algo == ALGO_CRYPTONIGHT ? MEMORY : MEMORY_LITE);

        init_cn_r(ctx[1]);
    }
}
