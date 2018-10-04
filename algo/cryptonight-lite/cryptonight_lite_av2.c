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

#include "algo/cryptonight/cryptonight.h"
#include "algo/cryptonight/cryptonight_monero.h"
#include "cryptonight_lite_aesni.h"
#include "crypto/c_keccak.h"


void cryptonight_lite_av2_v0(const uint8_t *restrict input, size_t size, uint8_t *restrict output, struct cryptonight_ctx **restrict ctx)
{
    keccak(input,        size, ctx[0]->state, 200);
    keccak(input + size, size, ctx[1]->state, 200);

    const uint8_t* l0 = ctx[0]->memory;
    const uint8_t* l1 = ctx[1]->memory;
    uint64_t* h0 = (uint64_t*) ctx[0]->state;
    uint64_t* h1 = (uint64_t*) ctx[1]->state;

    cn_explode_scratchpad((__m128i*) h0, (__m128i*) l0);
    cn_explode_scratchpad((__m128i*) h1, (__m128i*) l1);

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t al1 = h1[0] ^ h1[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    uint64_t ah1 = h1[1] ^ h1[5];

    __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

    uint64_t idx0 = h0[0] ^ h0[4];
    uint64_t idx1 = h1[0] ^ h1[4];

    for (size_t i = 0; __builtin_expect(i < 0x40000, 1); i++) {
        __m128i cx0 = _mm_load_si128((__m128i *) &l0[idx0 & 0xFFFF0]);
        __m128i cx1 = _mm_load_si128((__m128i *) &l1[idx1 & 0xFFFF0]);

        cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
        cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));

        _mm_store_si128((__m128i *) &l0[idx0 & 0xFFFF0], _mm_xor_si128(bx0, cx0));
        _mm_store_si128((__m128i *) &l1[idx1 & 0xFFFF0], _mm_xor_si128(bx1, cx1));

        idx0 = EXTRACT64(cx0);
        idx1 = EXTRACT64(cx1);

        bx0 = cx0;
        bx1 = cx1;

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & 0xFFFF0])[0];
        ch = ((uint64_t*) &l0[idx0 & 0xFFFF0])[1];
        lo = _umul128(idx0, cl, &hi);

        al0 += hi;
        ah0 += lo;

        ((uint64_t*) &l0[idx0 & 0xFFFF0])[0] = al0;
        ((uint64_t*) &l0[idx0 & 0xFFFF0])[1] = ah0;

        ah0 ^= ch;
        al0 ^= cl;
        idx0 = al0;

        cl = ((uint64_t*) &l1[idx1 & 0xFFFF0])[0];
        ch = ((uint64_t*) &l1[idx1 & 0xFFFF0])[1];
        lo = _umul128(idx1, cl, &hi);

        al1 += hi;
        ah1 += lo;

        ((uint64_t*) &l1[idx1 & 0xFFFF0])[0] = al1;
        ((uint64_t*) &l1[idx1 & 0xFFFF0])[1] = ah1;

        ah1 ^= ch;
        al1 ^= cl;
        idx1 = al1;
    }

    cn_implode_scratchpad((__m128i*) l0, (__m128i*) h0);
    cn_implode_scratchpad((__m128i*) l1, (__m128i*) h1);

    keccakf(h0, 24);
    keccakf(h1, 24);

    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
    extra_hashes[ctx[1]->state[0] & 3](ctx[1]->state, 200, (char*) output + 32);
}


void cryptonight_lite_av2_v1(const uint8_t *restrict input, size_t size, uint8_t *restrict output, struct cryptonight_ctx **restrict ctx)
{
    if (size < 43) {
        memset(output, 0, 64);
        return;
    }

    keccak(input,        size, ctx[0]->state, 200);
    keccak(input + size, size, ctx[1]->state, 200);

    VARIANT1_INIT(0);
    VARIANT1_INIT(1);

    const uint8_t* l0 = ctx[0]->memory;
    const uint8_t* l1 = ctx[1]->memory;
    uint64_t* h0 = (uint64_t*) ctx[0]->state;
    uint64_t* h1 = (uint64_t*) ctx[1]->state;

    cn_explode_scratchpad((__m128i*) h0, (__m128i*) l0);
    cn_explode_scratchpad((__m128i*) h1, (__m128i*) l1);

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t al1 = h1[0] ^ h1[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    uint64_t ah1 = h1[1] ^ h1[5];

    __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

    uint64_t idx0 = h0[0] ^ h0[4];
    uint64_t idx1 = h1[0] ^ h1[4];

    for (size_t i = 0; __builtin_expect(i < 0x40000, 1); i++) {
        __m128i cx0 = _mm_load_si128((__m128i *) &l0[idx0 & 0xFFFF0]);
        __m128i cx1 = _mm_load_si128((__m128i *) &l1[idx1 & 0xFFFF0]);

        cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
        cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));

        cryptonight_monero_tweak((uint64_t*)&l0[idx0 & 0xFFFF0], _mm_xor_si128(bx0, cx0));
        cryptonight_monero_tweak((uint64_t*)&l1[idx1 & 0xFFFF0], _mm_xor_si128(bx1, cx1));

        idx0 = EXTRACT64(cx0);
        idx1 = EXTRACT64(cx1);

        bx0 = cx0;
        bx1 = cx1;

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & 0xFFFF0])[0];
        ch = ((uint64_t*) &l0[idx0 & 0xFFFF0])[1];
        lo = _umul128(idx0, cl, &hi);

        al0 += hi;
        ah0 += lo;

        ((uint64_t*) &l0[idx0 & 0xFFFF0])[0] = al0;
        ((uint64_t*) &l0[idx0 & 0xFFFF0])[1] = ah0 ^ tweak1_2_0;

        ah0 ^= ch;
        al0 ^= cl;
        idx0 = al0;

        cl = ((uint64_t*) &l1[idx1 & 0xFFFF0])[0];
        ch = ((uint64_t*) &l1[idx1 & 0xFFFF0])[1];
        lo = _umul128(idx1, cl, &hi);

        al1 += hi;
        ah1 += lo;

        ((uint64_t*) &l1[idx1 & 0xFFFF0])[0] = al1;
        ((uint64_t*) &l1[idx1 & 0xFFFF0])[1] = ah1 ^ tweak1_2_1;

        ah1 ^= ch;
        al1 ^= cl;
        idx1 = al1;
    }

    cn_implode_scratchpad((__m128i*) l0, (__m128i*) h0);
    cn_implode_scratchpad((__m128i*) l1, (__m128i*) h1);

    keccakf(h0, 24);
    keccakf(h1, 24);

    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
    extra_hashes[ctx[1]->state[0] & 3](ctx[1]->state, 200, (char*) output + 32);
}
