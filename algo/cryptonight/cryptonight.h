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

#ifndef __CRYPTONIGHT_H__
#define __CRYPTONIGHT_H__

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define MEMORY      2097152 /* 2 MiB */
#define MEMORY_LITE 1048576 /* 1 MiB */

struct cryptonight_ctx {
    uint8_t state0[200] __attribute__((aligned(16)));
    uint8_t state1[200] __attribute__((aligned(16)));
    uint8_t* memory     __attribute__((aligned(16)));
};


extern void (* const extra_hashes[4])(const void *, size_t, char *);

bool cryptonight_init(int variant);
int scanhash_cryptonight(int thr_id, uint32_t *hash, uint32_t *restrict blob, size_t blob_size, uint32_t target, uint32_t max_nonce, unsigned long *restrict hashes_done, struct cryptonight_ctx *restrict ctx);
int scanhash_cryptonight_double(int thr_id, uint32_t *hash, uint8_t *restrict blob, size_t blob_size, uint32_t target, uint32_t max_nonce, unsigned long *restrict hashes_done, struct cryptonight_ctx *restrict ctx);

#endif /* __CRYPTONIGHT_H__ */
