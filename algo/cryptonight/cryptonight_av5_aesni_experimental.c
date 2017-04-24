/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017      fireice-uk  <https://github.com/fireice-uk>
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

#include <x86intrin.h>
#include <string.h>

#include "cryptonight.h"
#include "cryptonight_p.h"
#include "crypto/c_keccak.h"


void cryptonight_av5_aesni_experimental(const void *restrict input, size_t size, void *restrict output, struct cryptonight_ctx *restrict ctx)
{
    const uint8_t* memory = ctx->memory;

    keccak((const uint8_t *) input, size, ctx->state, 200);
    cn_explode_scratchpad((__m128i*) ctx->state, (__m128i*) memory);

    uint64_t* state = (uint64_t*) ctx->state;

    uint64_t a[2] __attribute((aligned(16))) = { state[0] ^ state[4], state[1] ^ state[5] };
    uint64_t c    __attribute((aligned(16)));
    uint64_t d[2] __attribute((aligned(16)));

    __m128i a_x = _mm_load_si128((__m128i *) &memory[a[0] & 0x1FFFF0]);
    __m128i b_x = _mm_set_epi64x(state[3] ^ state[7], state[2] ^ state[6]);

    for (size_t i = 0; __builtin_expect(i < 0x80000, 1); i++) {
        __m128i c_x = _mm_aesenc_si128(a_x, _mm_load_si128((__m128i *) a));
        c = _mm_cvtsi128_si64(c_x);

        uint64_t *restrict d_ptr = (uint64_t *) &memory[c & 0x1FFFF0];
        _mm_store_si128((__m128i *) &memory[a[0] & 0x1FFFF0], _mm_xor_si128(b_x, c_x));
        b_x = c_x;

        d[0] = d_ptr[0];
        d[1] = d_ptr[1];

        {
            unsigned __int128 res = (unsigned __int128) c * d[0];

            d_ptr[0] = a[0] += res >> 64;
            d_ptr[1] = a[1] += (uint64_t) res;
        }

        a[0] ^= d[0];
        a[1] ^= d[1];

        a_x = _mm_load_si128((__m128i *) &memory[a[0] & 0x1FFFF0]);
    }

    cn_implode_scratchpad((__m128i*) memory, (__m128i*) state);

    keccakf(state, 24);
    extra_hashes[ctx->state[0] & 3](ctx->state, 200, output);
}
