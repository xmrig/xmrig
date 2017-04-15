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

#include <x86intrin.h>
#include <string.h>

#include "cryptonight.h"
#include "crypto/c_keccak.h"


static inline void ExpandAESKey256_sub1(__m128i *tmp1, __m128i *tmp2)
{
    __m128i tmp4;
    *tmp2 = _mm_shuffle_epi32(*tmp2, 0xFF);
    tmp4 = _mm_slli_si128(*tmp1, 0x04);
    *tmp1 = _mm_xor_si128(*tmp1, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    *tmp1 = _mm_xor_si128(*tmp1, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    *tmp1 = _mm_xor_si128(*tmp1, tmp4);
    *tmp1 = _mm_xor_si128(*tmp1, *tmp2);
}

static inline void ExpandAESKey256_sub2(__m128i *tmp1, __m128i *tmp3)
{
    __m128i tmp2, tmp4;

    tmp4 = _mm_aeskeygenassist_si128(*tmp1, 0x00);
    tmp2 = _mm_shuffle_epi32(tmp4, 0xAA);
    tmp4 = _mm_slli_si128(*tmp3, 0x04);
    *tmp3 = _mm_xor_si128(*tmp3, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    *tmp3 = _mm_xor_si128(*tmp3, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    *tmp3 = _mm_xor_si128(*tmp3, tmp4);
    *tmp3 = _mm_xor_si128(*tmp3, tmp2);
}

// Special thanks to Intel for helping me
// with ExpandAESKey256() and its subroutines
static inline void ExpandAESKey256(char *keybuf)
{
    __m128i tmp1, tmp2, tmp3, *keys;

    keys = (__m128i *)keybuf;

    tmp1 = _mm_load_si128((__m128i *)keybuf);
    tmp3 = _mm_load_si128((__m128i *)(keybuf+0x10));

    tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x01);
    ExpandAESKey256_sub1(&tmp1, &tmp2);
    keys[2] = tmp1;
    ExpandAESKey256_sub2(&tmp1, &tmp3);
    keys[3] = tmp3;

    tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x02);
    ExpandAESKey256_sub1(&tmp1, &tmp2);
    keys[4] = tmp1;
    ExpandAESKey256_sub2(&tmp1, &tmp3);
    keys[5] = tmp3;

    tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x04);
    ExpandAESKey256_sub1(&tmp1, &tmp2);
    keys[6] = tmp1;
    ExpandAESKey256_sub2(&tmp1, &tmp3);
    keys[7] = tmp3;

    tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x08);
    ExpandAESKey256_sub1(&tmp1, &tmp2);
    keys[8] = tmp1;
    ExpandAESKey256_sub2(&tmp1, &tmp3);
    keys[9] = tmp3;

    tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x10);
    ExpandAESKey256_sub1(&tmp1, &tmp2);
    keys[10] = tmp1;
    ExpandAESKey256_sub2(&tmp1, &tmp3);
    keys[11] = tmp3;

    tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x20);
    ExpandAESKey256_sub1(&tmp1, &tmp2);
    keys[12] = tmp1;
    ExpandAESKey256_sub2(&tmp1, &tmp3);
    keys[13] = tmp3;

    tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x40);
    ExpandAESKey256_sub1(&tmp1, &tmp2);
    keys[14] = tmp1;
}

void cryptonight_av1_aesni(void *restrict output, const void *restrict input, const char *restrict memory, struct cryptonight_ctx *restrict ctx)
{
    keccak((const uint8_t *)input, 76, (uint8_t *) &ctx->state.hs, 200);
    uint8_t ExpandedKey[256];
    size_t i, j;

    memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);
    memcpy(ExpandedKey, ctx->state.hs.b, AES_KEY_SIZE);
    ExpandAESKey256(ExpandedKey);

    __m128i *longoutput, *expkey, *xmminput;
    longoutput = (__m128i *) memory;
    expkey = (__m128i *)ExpandedKey;
    xmminput = (__m128i *)ctx->text;

    for (i = 0; __builtin_expect(i < MEMORY, 1); i += INIT_SIZE_BYTE)
    {
        for(j = 0; j < 10; j++)
        {
            xmminput[0] = _mm_aesenc_si128(xmminput[0], expkey[j]);
            xmminput[1] = _mm_aesenc_si128(xmminput[1], expkey[j]);
            xmminput[2] = _mm_aesenc_si128(xmminput[2], expkey[j]);
            xmminput[3] = _mm_aesenc_si128(xmminput[3], expkey[j]);
            xmminput[4] = _mm_aesenc_si128(xmminput[4], expkey[j]);
            xmminput[5] = _mm_aesenc_si128(xmminput[5], expkey[j]);
            xmminput[6] = _mm_aesenc_si128(xmminput[6], expkey[j]);
            xmminput[7] = _mm_aesenc_si128(xmminput[7], expkey[j]);
        }
        _mm_store_si128(&(longoutput[(i >> 4)]), xmminput[0]);
        _mm_store_si128(&(longoutput[(i >> 4) + 1]), xmminput[1]);
        _mm_store_si128(&(longoutput[(i >> 4) + 2]), xmminput[2]);
        _mm_store_si128(&(longoutput[(i >> 4) + 3]), xmminput[3]);
        _mm_store_si128(&(longoutput[(i >> 4) + 4]), xmminput[4]);
        _mm_store_si128(&(longoutput[(i >> 4) + 5]), xmminput[5]);
        _mm_store_si128(&(longoutput[(i >> 4) + 6]), xmminput[6]);
        _mm_store_si128(&(longoutput[(i >> 4) + 7]), xmminput[7]);
    }

    for (i = 0; i < 2; i++)
    {
        ctx->a[i] = ((uint64_t *)ctx->state.k)[i] ^  ((uint64_t *)ctx->state.k)[i+4];
        ctx->b[i] = ((uint64_t *)ctx->state.k)[i+2] ^  ((uint64_t *)ctx->state.k)[i+6];
    }

    __m128i a_x = _mm_load_si128((__m128i *) &memory[ctx->a[0] & 0x1FFFF0]);
    __m128i b_x = _mm_load_si128((__m128i *) ctx->b);

    uint64_t c[2] __attribute((aligned(16)));
    uint64_t d[2] __attribute((aligned(16)));

    for (i = 0; __builtin_expect(i < 0x80000, 1); i++) {
        __m128i c_x = _mm_aesenc_si128(a_x, _mm_load_si128((__m128i *) ctx->a));
        _mm_store_si128((__m128i *) c, c_x);

        uint64_t *restrict d_ptr = (uint64_t *) &memory[c[0] & 0x1FFFF0];
        _mm_store_si128((__m128i *) &memory[ctx->a[0] & 0x1FFFF0], _mm_xor_si128(b_x, c_x));
        b_x = c_x;

        d[0] = d_ptr[0];
        d[1] = d_ptr[1];

        {
            unsigned __int128 res = (unsigned __int128) c[0] * d[0];

            d_ptr[0] = ctx->a[0] += res >> 64;
            d_ptr[1] = ctx->a[1] += (uint64_t) res;
        }

        ctx->a[0] ^= d[0];
        ctx->a[1] ^= d[1];

        a_x = _mm_load_si128((__m128i *) &memory[ctx->a[0] & 0x1FFFF0]);
    }

    memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);
    memcpy(ExpandedKey, &ctx->state.hs.b[32], AES_KEY_SIZE);
    ExpandAESKey256(ExpandedKey);

    for (i = 0; __builtin_expect(i < MEMORY, 1); i += INIT_SIZE_BYTE) {
        xmminput[0] = _mm_xor_si128(longoutput[(i >> 4)], xmminput[0]);
        xmminput[1] = _mm_xor_si128(longoutput[(i >> 4) + 1], xmminput[1]);
        xmminput[2] = _mm_xor_si128(longoutput[(i >> 4) + 2], xmminput[2]);
        xmminput[3] = _mm_xor_si128(longoutput[(i >> 4) + 3], xmminput[3]);
        xmminput[4] = _mm_xor_si128(longoutput[(i >> 4) + 4], xmminput[4]);
        xmminput[5] = _mm_xor_si128(longoutput[(i >> 4) + 5], xmminput[5]);
        xmminput[6] = _mm_xor_si128(longoutput[(i >> 4) + 6], xmminput[6]);
        xmminput[7] = _mm_xor_si128(longoutput[(i >> 4) + 7], xmminput[7]);

        for(j = 0; j < 10; j++)
        {
            xmminput[0] = _mm_aesenc_si128(xmminput[0], expkey[j]);
            xmminput[1] = _mm_aesenc_si128(xmminput[1], expkey[j]);
            xmminput[2] = _mm_aesenc_si128(xmminput[2], expkey[j]);
            xmminput[3] = _mm_aesenc_si128(xmminput[3], expkey[j]);
            xmminput[4] = _mm_aesenc_si128(xmminput[4], expkey[j]);
            xmminput[5] = _mm_aesenc_si128(xmminput[5], expkey[j]);
            xmminput[6] = _mm_aesenc_si128(xmminput[6], expkey[j]);
            xmminput[7] = _mm_aesenc_si128(xmminput[7], expkey[j]);
        }

    }

    memcpy(ctx->state.init, ctx->text, INIT_SIZE_BYTE);
    keccakf((uint64_t *) &ctx->state.hs, 24);
    extra_hashes[ctx->state.hs.b[0] & 3](&ctx->state, 200, output);
}
