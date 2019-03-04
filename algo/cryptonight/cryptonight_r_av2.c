/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017      fireice-uk  <https://github.com/fireice-uk>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#include <x86intrin.h>
#include <string.h>

#include "crypto/c_keccak.h"
#include "cryptonight.h"
#include "cryptonight_aesni.h"
#include "cryptonight_monero.h"


void cryptonight_r_av2(const uint8_t *restrict input, size_t size, uint8_t *restrict output, struct cryptonight_ctx **restrict ctx)
{
    keccak(input,        size, ctx[0]->state, 200);
    keccak(input + size, size, ctx[1]->state, 200);

    const uint8_t* l0 = ctx[0]->memory;
    const uint8_t* l1 = ctx[1]->memory;
    uint64_t* h0 = (uint64_t*) ctx[0]->state;
    uint64_t* h1 = (uint64_t*) ctx[1]->state;

    VARIANT2_INIT(0);
    VARIANT2_INIT(1);
    VARIANT2_SET_ROUNDING_MODE();
    VARIANT4_RANDOM_MATH_INIT(0);
    VARIANT4_RANDOM_MATH_INIT(1);

    cn_explode_scratchpad((__m128i*) h0, (__m128i*) l0);
    cn_explode_scratchpad((__m128i*) h1, (__m128i*) l1);

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t al1 = h1[0] ^ h1[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    uint64_t ah1 = h1[1] ^ h1[5];

    __m128i bx00 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx01 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);
    __m128i bx10 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
    __m128i bx11 = _mm_set_epi64x(h1[9] ^ h1[11], h1[8] ^ h1[10]);

    uint64_t idx0 = al0;
    uint64_t idx1 = al1;

    for (size_t i = 0; __builtin_expect(i < 0x80000, 1); i++) {
        __m128i cx0       = _mm_load_si128((__m128i *) &l0[idx0 & 0x1FFFF0]);
        __m128i cx1       = _mm_load_si128((__m128i *) &l1[idx1 & 0x1FFFF0]);

        const __m128i ax0 = _mm_set_epi64x(ah0, al0);
        const __m128i ax1 = _mm_set_epi64x(ah1, al1);

        cx0 = _mm_aesenc_si128(cx0, ax0);
        cx1 = _mm_aesenc_si128(cx1, ax1);

        VARIANT4_SHUFFLE(l0, idx0 & 0x1FFFF0, ax0, bx00, bx01, cx0);
        _mm_store_si128((__m128i *) &l0[idx0 & 0x1FFFF0], _mm_xor_si128(bx00, cx0));

        VARIANT4_SHUFFLE(l1, idx1 & 0x1FFFF0, ax1, bx10, bx11, cx1);
        _mm_store_si128((__m128i *) &l1[idx1 & 0x1FFFF0], _mm_xor_si128(bx10, cx1));

        idx0 = _mm_cvtsi128_si64(cx0);
        idx1 = _mm_cvtsi128_si64(cx1);

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & 0x1FFFF0])[0];
        ch = ((uint64_t*) &l0[idx0 & 0x1FFFF0])[1];

        VARIANT4_RANDOM_MATH(0, al0, ah0, cl, bx00, bx01);
        al0 ^= r0[2] | ((uint64_t)(r0[3]) << 32);
        ah0 ^= r0[0] | ((uint64_t)(r0[1]) << 32);

        lo = _umul128(idx0, cl, &hi);
        VARIANT4_SHUFFLE(l0, idx0 & 0x1FFFF0, ax0, bx00, bx01, cx0);

        al0 += hi;
        ah0 += lo;

        ((uint64_t*)&l0[idx0 & 0x1FFFF0])[0] = al0;
        ((uint64_t*)&l0[idx0 & 0x1FFFF0])[1] = ah0;

        al0 ^= cl;
        ah0 ^= ch;
        idx0 = al0;

        cl = ((uint64_t*) &l1[idx1 & 0x1FFFF0])[0];
        ch = ((uint64_t*) &l1[idx1 & 0x1FFFF0])[1];

        VARIANT4_RANDOM_MATH(1, al1, ah1, cl, bx10, bx11);
        al1 ^= r1[2] | ((uint64_t)(r1[3]) << 32);
        ah1 ^= r1[0] | ((uint64_t)(r1[1]) << 32);

        lo = _umul128(idx1, cl, &hi);
        VARIANT4_SHUFFLE(l1, idx1 & 0x1FFFF0, ax1, bx10, bx11, cx1);

        al1 += hi;
        ah1 += lo;

        ((uint64_t*)&l1[idx1 & 0x1FFFF0])[0] = al1;
        ((uint64_t*)&l1[idx1 & 0x1FFFF0])[1] = ah1;

        al1 ^= cl;
        ah1 ^= ch;
        idx1 = al1;

        bx01 = bx00;
        bx11 = bx10;

        bx00 = cx0;
        bx10 = cx1;
    }

    cn_implode_scratchpad((__m128i*) l0, (__m128i*) h0);
    cn_implode_scratchpad((__m128i*) l1, (__m128i*) h1);

    keccakf(h0, 24);
    keccakf(h1, 24);

    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
    extra_hashes[ctx[1]->state[0] & 3](ctx[1]->state, 200, output + 32);
}


#ifndef XMRIG_NO_ASM
void v4_compile_code_double(const struct V4_Instruction* code, int code_size, void* machine_code, enum Assembly ASM);


void cryptonight_r_av2_asm_intel(const uint8_t *restrict input, size_t size, uint8_t *restrict output, struct cryptonight_ctx **restrict ctx)
{
    if (ctx[0]->generated_code_height != ctx[0]->height) {
        struct V4_Instruction code[256];
        const int code_size = v4_random_math_init(code, ctx[0]->height);
        v4_compile_code_double(code, code_size, (void*)(ctx[0]->generated_code_double), ASM_INTEL);
        ctx[0]->generated_code_height = ctx[0]->height;
    }

    keccak(input,        size, ctx[0]->state, 200);
    keccak(input + size, size, ctx[1]->state, 200);
    cn_explode_scratchpad((__m128i*) ctx[0]->state, (__m128i*) ctx[0]->memory);
    cn_explode_scratchpad((__m128i*) ctx[1]->state, (__m128i*) ctx[1]->memory);

    ctx[0]->generated_code_double(ctx[0], ctx[1]);

    cn_implode_scratchpad((__m128i*) ctx[0]->memory, (__m128i*) ctx[0]->state);
    cn_implode_scratchpad((__m128i*) ctx[1]->memory, (__m128i*) ctx[1]->state);

    keccakf((uint64_t *) ctx[0]->state, 24);
    keccakf((uint64_t *) ctx[1]->state, 24);

    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
    extra_hashes[ctx[1]->state[0] & 3](ctx[1]->state, 200, output + 32);
}


void cryptonight_r_av2_asm_bulldozer(const uint8_t *restrict input, size_t size, uint8_t *restrict output, struct cryptonight_ctx **restrict ctx)
{
    if (ctx[0]->generated_code_height != ctx[0]->height) {
        struct V4_Instruction code[256];
        const int code_size = v4_random_math_init(code, ctx[0]->height);
        v4_compile_code_double(code, code_size, (void*)(ctx[0]->generated_code_double), ASM_BULLDOZER);
        ctx[0]->generated_code_height = ctx[0]->height;
    }

    keccak(input,        size, ctx[0]->state, 200);
    keccak(input + size, size, ctx[1]->state, 200);
    cn_explode_scratchpad((__m128i*) ctx[0]->state, (__m128i*) ctx[0]->memory);
    cn_explode_scratchpad((__m128i*) ctx[1]->state, (__m128i*) ctx[1]->memory);

    ctx[0]->generated_code_double(ctx[0], ctx[1]);

    cn_implode_scratchpad((__m128i*) ctx[0]->memory, (__m128i*) ctx[0]->state);
    cn_implode_scratchpad((__m128i*) ctx[1]->memory, (__m128i*) ctx[1]->state);

    keccakf((uint64_t *) ctx[0]->state, 24);
    keccakf((uint64_t *) ctx[1]->state, 24);

    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
    extra_hashes[ctx[1]->state[0] & 3](ctx[1]->state, 200, output + 32);
}
#endif
