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
#include "compat.h"
#include "crypto/c_keccak.h"
#include "crypto/aesb.h"
#include "crypto/oaes_lib.h"


static inline uint64_t mul128(uint64_t multiplier, uint64_t multiplicand, uint64_t *product_hi) {
  // multiplier   = ab = a * 2^32 + b
  // multiplicand = cd = c * 2^32 + d
  // ab * cd = a * c * 2^64 + (a * d + b * c) * 2^32 + b * d
  uint64_t a = multiplier >> 32;
  uint64_t b = multiplier & 0xFFFFFFFF;
  uint64_t c = multiplicand >> 32;
  uint64_t d = multiplicand & 0xFFFFFFFF;

  //uint64_t ac = a * c;
  uint64_t ad = a * d;
  //uint64_t bc = b * c;
  uint64_t bd = b * d;

  uint64_t adbc = ad + (b * c);
  uint64_t adbc_carry = adbc < ad ? 1 : 0;

  // multiplier * multiplicand = product_hi * 2^64 + product_lo
  uint64_t product_lo = bd + (adbc << 32);
  uint64_t product_lo_carry = product_lo < bd ? 1 : 0;
  *product_hi = (a * c) + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;

  return product_lo;
}


static inline void mul_sum_xor_dst(const uint8_t* a, uint8_t* c, uint8_t* dst) {
    uint64_t hi, lo = mul128(((uint64_t*) a)[0], ((uint64_t*) dst)[0], &hi) + ((uint64_t*) c)[1];
    hi += ((uint64_t*) c)[0];

    ((uint64_t*) c)[0] = ((uint64_t*) dst)[0] ^ hi;
    ((uint64_t*) c)[1] = ((uint64_t*) dst)[1] ^ lo;
    ((uint64_t*) dst)[0] = hi;
    ((uint64_t*) dst)[1] = lo;
}


static inline void xor_blocks(uint8_t* a, const uint8_t* b) {
    ((uint64_t*) a)[0] ^= ((uint64_t*) b)[0];
    ((uint64_t*) a)[1] ^= ((uint64_t*) b)[1];
}


static inline void xor_blocks_dst(const uint8_t* a, const uint8_t* b, uint8_t* dst) {
    ((uint64_t*) dst)[0] = ((uint64_t*) a)[0] ^ ((uint64_t*) b)[0];
    ((uint64_t*) dst)[1] = ((uint64_t*) a)[1] ^ ((uint64_t*) b)[1];
}


void cryptonight_av4_legacy(void *restrict output, const void *restrict input, const char *restrict memory, struct cryptonight_ctx *restrict ctx) {
    oaes_ctx *aes_ctx = (oaes_ctx*) oaes_alloc();
    size_t i, j;
    keccak((const uint8_t *)input, 76, (uint8_t *) &ctx->state.hs, 200);
    memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);

    oaes_key_import_data(aes_ctx, ctx->state.hs.b, AES_KEY_SIZE);

   for (i = 0; likely(i < MEMORY); i += INIT_SIZE_BYTE) {
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 0], aes_ctx->key->exp_data);
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 1], aes_ctx->key->exp_data);
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 2], aes_ctx->key->exp_data);
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 3], aes_ctx->key->exp_data);
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 4], aes_ctx->key->exp_data);
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 5], aes_ctx->key->exp_data);
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 6], aes_ctx->key->exp_data);
        aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * 7], aes_ctx->key->exp_data);
        memcpy((void *) &memory[i], ctx->text, INIT_SIZE_BYTE);
    }

    xor_blocks_dst(&ctx->state.k[0],  &ctx->state.k[32], (uint8_t*) ctx->a);
    xor_blocks_dst(&ctx->state.k[16], &ctx->state.k[48], (uint8_t*) ctx->b);

    for (i = 0; likely(i < ITER / 4); ++i) {
        /* Dependency chain: address -> read value ------+
         * written value <-+ hard function (AES or MUL) <+
         * next address  <-+
         */
        /* Iteration 1 */
        j = ctx->a[0] & 0x1FFFF0;
        aesb_single_round((const uint8_t*) &memory[j], (uint8_t *) ctx->c, (const uint8_t *) ctx->a);
        xor_blocks_dst((const uint8_t*) ctx->c, (const uint8_t*) ctx->b, (uint8_t*) &memory[j]);
        /* Iteration 2 */
        mul_sum_xor_dst((const uint8_t*) ctx->c, (uint8_t*) ctx->a, (uint8_t*) &memory[ctx->c[0] & 0x1FFFF0]);
        /* Iteration 3 */
        j = ctx->a[0] & 0x1FFFF0;
        aesb_single_round(&memory[j], (uint8_t *) ctx->b, (uint8_t *) ctx->a);
        xor_blocks_dst((const uint8_t*) ctx->b, (const uint8_t*) ctx->c, (uint8_t*) &memory[j]);
        /* Iteration 4 */
        mul_sum_xor_dst((const uint8_t*) ctx->b, (uint8_t*) ctx->a, (uint8_t*) &memory[ctx->b[0] & 0x1FFFF0]);
    }

    memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);
    oaes_key_import_data(aes_ctx, &ctx->state.hs.b[32], AES_KEY_SIZE);

    for (i = 0; likely(i < MEMORY); i += INIT_SIZE_BYTE) {
        xor_blocks(&ctx->text[0 * AES_BLOCK_SIZE], &memory[i + 0 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[0 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
        xor_blocks(&ctx->text[1 * AES_BLOCK_SIZE], &memory[i + 1 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[1 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
        xor_blocks(&ctx->text[2 * AES_BLOCK_SIZE], &memory[i + 2 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[2 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
        xor_blocks(&ctx->text[3 * AES_BLOCK_SIZE], &memory[i + 3 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[3 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
        xor_blocks(&ctx->text[4 * AES_BLOCK_SIZE], &memory[i + 4 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[4 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
        xor_blocks(&ctx->text[5 * AES_BLOCK_SIZE], &memory[i + 5 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[5 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
        xor_blocks(&ctx->text[6 * AES_BLOCK_SIZE], &memory[i + 6 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[6 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
        xor_blocks(&ctx->text[7 * AES_BLOCK_SIZE], &memory[i + 7 * AES_BLOCK_SIZE]);
        aesb_pseudo_round_mut(&ctx->text[7 * AES_BLOCK_SIZE], aes_ctx->key->exp_data);
    }

    memcpy(ctx->state.init, ctx->text, INIT_SIZE_BYTE);
    keccakf((uint64_t *) &ctx->state.hs, 24);
    extra_hashes[ctx->state.hs.b[0] & 3](&ctx->state, 200, output);
    oaes_free((OAES_CTX **) &aes_ctx);
}
