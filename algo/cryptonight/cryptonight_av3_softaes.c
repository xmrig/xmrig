/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017      fireice-uk  <https://github.com/fireice-uk>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <x86intrin.h>
#include <string.h>

#include "crypto/c_keccak.h"
#include "cryptonight.h"
#include "cryptonight_monero.h"
#include "cryptonight_softaes.h"


void cryptonight_av3_softaes(const void *restrict input, size_t size, void *restrict output, struct cryptonight_ctx *restrict ctx, uint8_t version)
{
    keccak((const uint8_t *) input, size, ctx->state0, 200);

    VARIANT1_INIT(0);

    cn_explode_scratchpad((__m128i*) ctx->state0, (__m128i*) ctx->memory);

    const uint8_t* l0 = ctx->memory;
    uint64_t* h0 = (uint64_t*) ctx->state0;

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);

    uint64_t idx0 = h0[0] ^ h0[4];

    for (size_t i = 0; __builtin_expect(i < 0x80000, 1); i++) {
        __m128i cx;
        cx = _mm_load_si128((__m128i *)&l0[idx0 & 0x1FFFF0]);
        cx = soft_aesenc(cx, _mm_set_epi64x(ah0, al0));

        _mm_store_si128((__m128i *)&l0[idx0 & 0x1FFFF0], _mm_xor_si128(bx0, cx));
        VARIANT1_1(&l0[idx0 & 0x1FFFF0]);
        idx0 = EXTRACT64(cx);
        bx0 = cx;

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*)&l0[idx0 & 0x1FFFF0])[0];
        ch = ((uint64_t*)&l0[idx0 & 0x1FFFF0])[1];
        lo = _umul128(idx0, cl, &hi);

        al0 += hi;
        ah0 += lo;

        VARIANT1_2(ah0, 0);
        ((uint64_t*)&l0[idx0 & 0x1FFFF0])[0] = al0;
        ((uint64_t*)&l0[idx0 & 0x1FFFF0])[1] = ah0;
        VARIANT1_2(ah0, 0);

        ah0 ^= ch;
        al0 ^= cl;
        idx0 = al0;
    }

    cn_implode_scratchpad((__m128i*) ctx->memory, (__m128i*) ctx->state0);

    keccakf(h0, 24);
    extra_hashes[ctx->state0[0] & 3](ctx->state0, 200, output);
}
