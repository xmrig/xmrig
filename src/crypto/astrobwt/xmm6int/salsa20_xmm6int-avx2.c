/*
 * ISC License
 *
 * Copyright (c) 2013-2021
 * Frank Denis <j at pureftpd dot org>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>

#define ROUNDS 20

typedef struct salsa_ctx {
    uint32_t input[16];
} salsa_ctx;

static const int TR[16] = {
    0, 5, 10, 15, 12, 1, 6, 11, 8, 13, 2, 7, 4, 9, 14, 3
};

#define LOAD32_LE(p) *((uint32_t*)(p))
#define STORE32_LE(dst, src) memcpy((dst), &(src), sizeof(uint32_t))

static void
salsa_keysetup(salsa_ctx *ctx, const uint8_t *k)
{
    ctx->input[TR[1]]  = LOAD32_LE(k + 0);
    ctx->input[TR[2]]  = LOAD32_LE(k + 4);
    ctx->input[TR[3]]  = LOAD32_LE(k + 8);
    ctx->input[TR[4]]  = LOAD32_LE(k + 12);
    ctx->input[TR[11]] = LOAD32_LE(k + 16);
    ctx->input[TR[12]] = LOAD32_LE(k + 20);
    ctx->input[TR[13]] = LOAD32_LE(k + 24);
    ctx->input[TR[14]] = LOAD32_LE(k + 28);
    ctx->input[TR[0]]  = 0x61707865;
    ctx->input[TR[5]]  = 0x3320646e;
    ctx->input[TR[10]] = 0x79622d32;
    ctx->input[TR[15]] = 0x6b206574;
}

static void
salsa_ivsetup(salsa_ctx *ctx, const uint8_t *iv, const uint8_t *counter)
{
    ctx->input[TR[6]] = LOAD32_LE(iv + 0);
    ctx->input[TR[7]] = LOAD32_LE(iv + 4);
    ctx->input[TR[8]] = counter == NULL ? 0 : LOAD32_LE(counter + 0);
    ctx->input[TR[9]] = counter == NULL ? 0 : LOAD32_LE(counter + 4);
}

static void
salsa20_encrypt_bytes(salsa_ctx *ctx, const uint8_t *m, uint8_t *c,
                      unsigned long long bytes)
{
    uint32_t * const x = &ctx->input[0];

    if (!bytes) {
        return; /* LCOV_EXCL_LINE */
    }

#include "u8.h"
#include "u4.h"
#include "u1.h"
#include "u0.h"
}

int salsa20_stream_avx2(void* c, uint64_t clen, const void* iv, const void* key)
{
	struct salsa_ctx ctx;

	if (!clen) {
		return 0;
	}

	salsa_keysetup(&ctx, (const uint8_t*)key);
	salsa_ivsetup(&ctx, (const uint8_t*)iv, NULL);
	memset(c, 0, clen);
	salsa20_encrypt_bytes(&ctx, (const uint8_t*)c, (uint8_t*)c, clen);

	return 0;
}
