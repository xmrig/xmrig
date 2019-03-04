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
#include "cryptonight_monero.h"
#include "cryptonight_softaes.h"


#ifndef XMRIG_NO_ASM
void v4_soft_aes_compile_code(const struct V4_Instruction* code, int code_size, void* machine_code, enum Assembly ASM);
#endif


void cryptonight_r_av3(const uint8_t *restrict input, size_t size, uint8_t *restrict output, struct cryptonight_ctx **restrict ctx)
{
    keccak(input, size, ctx[0]->state, 200);
    cn_explode_scratchpad((__m128i*) ctx[0]->state, (__m128i*) ctx[0]->memory);

#   ifndef XMRIG_NO_ASM
    if (ctx[0]->generated_code_height != ctx[0]->height) {
        struct V4_Instruction code[256];
        const int code_size = v4_random_math_init(code, ctx[0]->height);

        v4_soft_aes_compile_code(code, code_size, (void*)(ctx[0]->generated_code), ASM_NONE);
        ctx[0]->generated_code_height = ctx[0]->height;
    }

    ctx[0]->saes_table = (const uint32_t*)saes_table;
    ctx[0]->generated_code(ctx[0]);
#   else
    const uint8_t* l0 = ctx[0]->memory;
    uint64_t* h0 = (uint64_t*) ctx[0]->state;

    VARIANT2_INIT(0);
    VARIANT2_SET_ROUNDING_MODE();
    VARIANT4_RANDOM_MATH_INIT(0);

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx1 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);

    uint64_t idx0 = al0;

    for (size_t i = 0; __builtin_expect(i < 0x80000, 1); i++) {
        __m128i cx        = _mm_load_si128((__m128i *) &l0[idx0 & 0x1FFFF0]);
        const __m128i ax0 = _mm_set_epi64x(ah0, al0);

        cx = soft_aesenc(cx, ax0);

        VARIANT4_SHUFFLE(l0, idx0 & 0x1FFFF0, ax0, bx0, bx1, cx);
        _mm_store_si128((__m128i *) &l0[idx0 & 0x1FFFF0], _mm_xor_si128(bx0, cx));

        idx0 = _mm_cvtsi128_si64(cx);

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & 0x1FFFF0])[0];
        ch = ((uint64_t*) &l0[idx0 & 0x1FFFF0])[1];

        VARIANT4_RANDOM_MATH(0, al0, ah0, cl, bx0, bx1);
        al0 ^= r0[2] | ((uint64_t)(r0[3]) << 32);
        ah0 ^= r0[0] | ((uint64_t)(r0[1]) << 32);

        lo = _umul128(idx0, cl, &hi);
        VARIANT4_SHUFFLE(l0, idx0 & 0x1FFFF0, ax0, bx0, bx1, cx);

        al0 += hi;
        ah0 += lo;

        ((uint64_t*)&l0[idx0 & 0x1FFFF0])[0] = al0;
        ((uint64_t*)&l0[idx0 & 0x1FFFF0])[1] = ah0;

        al0 ^= cl;
        ah0 ^= ch;
        idx0 = al0;

        bx1 = bx0;
        bx0 = cx;
    }
#   endif

    cn_implode_scratchpad((__m128i*) ctx[0]->memory, (__m128i*) ctx[0]->state);
    keccakf(ctx[0]->state, 24);
    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
}
