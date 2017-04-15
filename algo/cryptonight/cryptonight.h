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

#define MEMORY          (1 << 21) /* 2 MiB */
#define MEMORY_M128I    (MEMORY >> 4) // 2 MiB / 16 = 128 ki * __m128i
#define ITER            (1 << 20)
#define AES_BLOCK_SIZE  16
#define AES_KEY_SIZE    32 /*16*/
#define INIT_SIZE_BLK   8
#define INIT_SIZE_BYTE  (INIT_SIZE_BLK * AES_BLOCK_SIZE) // 128
#define INIT_SIZE_M128I (INIT_SIZE_BYTE >> 4) // 8

#pragma pack(push, 1)
union hash_state {
  uint8_t b[200];
  uint64_t w[25];
};
#pragma pack(pop)

#pragma pack(push, 1)
union cn_slow_hash_state {
    union hash_state hs;
    struct {
        uint8_t k[64];
        uint8_t init[INIT_SIZE_BYTE];
    };
};
#pragma pack(pop)


struct cryptonight_ctx {
    union cn_slow_hash_state state;
    uint8_t text[INIT_SIZE_BYTE] __attribute((aligned(16)));
    uint64_t a[2] __attribute__((aligned(16)));
    uint64_t b[2] __attribute__((aligned(16)));
    uint64_t c[2] __attribute__((aligned(16)));
};


extern void (* const extra_hashes[4])(const void *, size_t, char *);

void cryptonight_init(int variant);
void cryptonight_hash(void* output, const void* input, size_t input_len);
int scanhash_cryptonight(int thr_id, uint32_t *hash, uint32_t *restrict pdata, const uint32_t *restrict ptarget, uint32_t max_nonce, unsigned long *restrict hashes_done, const char *memory, struct cryptonight_ctx *persistentctx);

#endif /* __CRYPTONIGHT_H__ */
