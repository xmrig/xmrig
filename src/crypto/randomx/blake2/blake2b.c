/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Original code from Argon2 reference source code package used under CC0 Licence
 * https://github.com/P-H-C/phc-winner-argon2
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
*/

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "crypto/randomx/blake2/blake2.h"
#include "crypto/randomx/blake2/blake2-impl.h"

#if defined(_M_X64) || defined(__x86_64__)

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <smmintrin.h>
#include "blake2b-round.h"

#endif

static const uint64_t blake2b_IV[8] = {
	UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
	UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
	UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
	UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179) };

#if defined(_M_X64) || defined(__x86_64__)
static const uint8_t blake2b_sigma_sse41[12][16] = {
	{0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15},
	{14, 4, 9, 13, 10, 8, 15, 6, 1, 0, 11, 5, 12, 2, 7, 3},
	{11, 12, 5, 15, 8, 0, 2, 13, 10, 3, 7, 9, 14, 6, 1, 4},
	{7, 3, 13, 11, 9, 1, 12, 14, 2, 5, 4, 15, 6, 10, 0, 8},
	{9, 5, 2, 10, 0, 7, 4, 15, 14, 11, 6, 3, 1, 12, 8, 13},
	{2, 6, 0, 8, 12, 10, 11, 3, 4, 7, 15, 1, 13, 5, 14, 9},
	{12, 1, 14, 4, 5, 15, 13, 10, 0, 6, 9, 8, 7, 3, 2, 11},
	{13, 7, 12, 3, 11, 14, 1, 9, 5, 15, 8, 2, 0, 4, 6, 10},
	{6, 14, 11, 0, 15, 9, 3, 8, 12, 13, 1, 10, 2, 7, 4, 5},
	{10, 8, 7, 1, 2, 4, 6, 5, 15, 9, 3, 13, 11, 14, 12, 0},
	{0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15},
	{14, 4, 9, 13, 10, 8, 15, 6, 1, 0, 11, 5, 12, 2, 7, 3},
};
#endif

static const uint8_t blake2b_sigma[12][16] = {
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

static FORCE_INLINE void blake2b_set_lastnode(blake2b_state *S) {
	S->f[1] = (uint64_t)-1;
}

static FORCE_INLINE void blake2b_set_lastblock(blake2b_state *S) {
	if (S->last_node) {
		blake2b_set_lastnode(S);
	}
	S->f[0] = (uint64_t)-1;
}

static FORCE_INLINE void blake2b_increment_counter(blake2b_state *S,
	uint64_t inc) {
	S->t[0] += inc;
	S->t[1] += (S->t[0] < inc);
}

static FORCE_INLINE void blake2b_invalidate_state(blake2b_state *S) {
	//clear_internal_memory(S, sizeof(*S));      /* wipe */
	blake2b_set_lastblock(S); /* invalidate for further use */
}

static FORCE_INLINE void blake2b_init0(blake2b_state *S) {
	memset(S, 0, sizeof(*S));
	memcpy(S->h, blake2b_IV, sizeof(S->h));
}

int rx_blake2b_init_param(blake2b_state *S, const blake2b_param *P) {
	const unsigned char *p = (const unsigned char *)P;
	unsigned int i;

	if (NULL == P || NULL == S) {
		return -1;
	}

	blake2b_init0(S);
	/* IV XOR Parameter Block */
	for (i = 0; i < 8; ++i) {
		S->h[i] ^= load64(&p[i * sizeof(S->h[i])]);
	}
	S->outlen = P->digest_length;
	return 0;
}

/* Sequential blake2b initialization */
int rx_blake2b_init(blake2b_state *S, size_t outlen) {
	blake2b_param P;

	if (S == NULL) {
		return -1;
	}

	if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	/* Setup Parameter Block for unkeyed BLAKE2 */
	P.digest_length = (uint8_t)outlen;
	P.key_length = 0;
	P.fanout = 1;
	P.depth = 1;
	P.leaf_length = 0;
	P.node_offset = 0;
	P.node_depth = 0;
	P.inner_length = 0;
	memset(P.reserved, 0, sizeof(P.reserved));
	memset(P.salt, 0, sizeof(P.salt));
	memset(P.personal, 0, sizeof(P.personal));

	return rx_blake2b_init_param(S, &P);
}

int rx_blake2b_init_key(blake2b_state *S, size_t outlen, const void *key, size_t keylen) {
	blake2b_param P;

	if (S == NULL) {
		return -1;
	}

	if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	if ((key == 0) || (keylen == 0) || (keylen > BLAKE2B_KEYBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	/* Setup Parameter Block for keyed BLAKE2 */
	P.digest_length = (uint8_t)outlen;
	P.key_length = (uint8_t)keylen;
	P.fanout = 1;
	P.depth = 1;
	P.leaf_length = 0;
	P.node_offset = 0;
	P.node_depth = 0;
	P.inner_length = 0;
	memset(P.reserved, 0, sizeof(P.reserved));
	memset(P.salt, 0, sizeof(P.salt));
	memset(P.personal, 0, sizeof(P.personal));

	if (rx_blake2b_init_param(S, &P) < 0) {
		blake2b_invalidate_state(S);
		return -1;
	}

	{
		uint8_t block[BLAKE2B_BLOCKBYTES];
		memset(block, 0, BLAKE2B_BLOCKBYTES);
		memcpy(block, key, keylen);
        rx_blake2b_update(S, block, BLAKE2B_BLOCKBYTES);
		/* Burn the key from stack */
		//clear_internal_memory(block, BLAKE2B_BLOCKBYTES);
	}
	return 0;
}

#if defined(_M_X64) || defined(__x86_64__)
static void rx_blake2b_compress_sse41(blake2b_state* S, const uint8_t *block)
{
	__m128i row1l, row1h;
	__m128i row2l, row2h;
	__m128i row3l, row3h;
	__m128i row4l, row4h;
	__m128i b0, b1;
	__m128i t0, t1;

	const __m128i r16 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9);
	const __m128i r24 = _mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10);

	row1l = LOADU(&S->h[0]);
	row1h = LOADU(&S->h[2]);
	row2l = LOADU(&S->h[4]);
	row2h = LOADU(&S->h[6]);
	row3l = LOADU(&blake2b_IV[0]);
	row3h = LOADU(&blake2b_IV[2]);
	row4l = _mm_xor_si128(LOADU(&blake2b_IV[4]), LOADU(&S->t[0]));
	row4h = _mm_xor_si128(LOADU(&blake2b_IV[6]), LOADU(&S->f[0]));

	const uint64_t* m = (const uint64_t*)(block);

	for (uint32_t r = 0; r < 12; ++r) {
		ROUND(r);
	}

	row1l = _mm_xor_si128(row3l, row1l);
	row1h = _mm_xor_si128(row3h, row1h);
	STOREU(&S->h[0], _mm_xor_si128(LOADU(&S->h[0]), row1l));
	STOREU(&S->h[2], _mm_xor_si128(LOADU(&S->h[2]), row1h));
	row2l = _mm_xor_si128(row4l, row2l);
	row2h = _mm_xor_si128(row4h, row2h);
	STOREU(&S->h[4], _mm_xor_si128(LOADU(&S->h[4]), row2l));
	STOREU(&S->h[6], _mm_xor_si128(LOADU(&S->h[6]), row2h));
}
#undef ROUND
#endif

static void rx_blake2b_compress_integer(blake2b_state *S, const uint8_t *block) {
	uint64_t m[16];
	uint64_t v[16];
	unsigned int i, r;

	for (i = 0; i < 16; ++i) {
		m[i] = load64(block + i * sizeof(m[i]));
	}

	for (i = 0; i < 8; ++i) {
		v[i] = S->h[i];
	}

	v[8] = blake2b_IV[0];
	v[9] = blake2b_IV[1];
	v[10] = blake2b_IV[2];
	v[11] = blake2b_IV[3];
	v[12] = blake2b_IV[4] ^ S->t[0];
	v[13] = blake2b_IV[5] ^ S->t[1];
	v[14] = blake2b_IV[6] ^ S->f[0];
	v[15] = blake2b_IV[7] ^ S->f[1];

#define G(r, i, a, b, c, d)                                                    \
    do {                                                                       \
        a = a + b + m[blake2b_sigma[r][2 * i + 0]];                            \
        d = rotr64(d ^ a, 32);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 24);                                                 \
        a = a + b + m[blake2b_sigma[r][2 * i + 1]];                            \
        d = rotr64(d ^ a, 16);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 63);                                                 \
    } while ((void)0, 0)

#define ROUND(r)                                                               \
    do {                                                                       \
        G(r, 0, v[0], v[4], v[8], v[12]);                                      \
        G(r, 1, v[1], v[5], v[9], v[13]);                                      \
        G(r, 2, v[2], v[6], v[10], v[14]);                                     \
        G(r, 3, v[3], v[7], v[11], v[15]);                                     \
        G(r, 4, v[0], v[5], v[10], v[15]);                                     \
        G(r, 5, v[1], v[6], v[11], v[12]);                                     \
        G(r, 6, v[2], v[7], v[8], v[13]);                                      \
        G(r, 7, v[3], v[4], v[9], v[14]);                                      \
    } while ((void)0, 0)

	for (r = 0; r < 12; ++r) {
		ROUND(r);
	}

	for (i = 0; i < 8; ++i) {
		S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
	}

#undef G
#undef ROUND
}

#if defined(_M_X64) || defined(__x86_64__)

uint32_t rx_blake2b_use_sse41 = 0;

#define rx_blake2b_compress(S, block) \
	if (rx_blake2b_use_sse41) \
		rx_blake2b_compress_sse41(S, block); \
	else \
		rx_blake2b_compress_integer(S, block);

#else
#define rx_blake2b_compress(S, block) rx_blake2b_compress_integer(S, block);
#endif

int rx_blake2b_update(blake2b_state *S, const void *in, size_t inlen) {
	const uint8_t *pin = (const uint8_t *)in;

	if (inlen == 0) {
		return 0;
	}

	/* Sanity check */
	if (S == NULL || in == NULL) {
		return -1;
	}

	/* Is this a reused state? */
	if (S->f[0] != 0) {
		return -1;
	}

	if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
		/* Complete current block */
		size_t left = S->buflen;
		size_t fill = BLAKE2B_BLOCKBYTES - left;
		memcpy(&S->buf[left], pin, fill);
		blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
		rx_blake2b_compress(S, S->buf);
		S->buflen = 0;
		inlen -= fill;
		pin += fill;
		/* Avoid buffer copies when possible */
		while (inlen > BLAKE2B_BLOCKBYTES) {
			blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
			rx_blake2b_compress(S, pin);
			inlen -= BLAKE2B_BLOCKBYTES;
			pin += BLAKE2B_BLOCKBYTES;
		}
	}
	memcpy(&S->buf[S->buflen], pin, inlen);
	S->buflen += (unsigned int)inlen;
	return 0;
}

int rx_blake2b_final(blake2b_state *S, void *out, size_t outlen) {
	uint8_t buffer[BLAKE2B_OUTBYTES] = { 0 };
	unsigned int i;

	/* Sanity checks */
	if (S == NULL || out == NULL || outlen < S->outlen) {
		return -1;
	}

	/* Is this a reused state? */
	if (S->f[0] != 0) {
		return -1;
	}

	blake2b_increment_counter(S, S->buflen);
	blake2b_set_lastblock(S);
	memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
	rx_blake2b_compress(S, S->buf);

	for (i = 0; i < 8; ++i) { /* Output full hash to temp buffer */
		store64(buffer + sizeof(S->h[i]) * i, S->h[i]);
	}

	memcpy(out, buffer, S->outlen);
	//clear_internal_memory(buffer, sizeof(buffer));
	//clear_internal_memory(S->buf, sizeof(S->buf));
	//clear_internal_memory(S->h, sizeof(S->h));
	return 0;
}

int rx_blake2b(void *out, size_t outlen, const void *in, size_t inlen) {
	blake2b_state S;
	int ret = -1;

	/* Verify parameters */
	if (NULL == in && inlen > 0) {
		goto fail;
	}

	if (NULL == out || outlen == 0 || outlen > BLAKE2B_OUTBYTES) {
		goto fail;
	}

	if (rx_blake2b_init(&S, outlen) < 0) {
		goto fail;
	}

	if (rx_blake2b_update(&S, in, inlen) < 0) {
		goto fail;
	}
	ret = rx_blake2b_final(&S, out, outlen);

fail:
	//clear_internal_memory(&S, sizeof(S));
	return ret;
}

/* Argon2 Team - Begin Code */
int rxa2_blake2b_long(void *pout, size_t outlen, const void *in, size_t inlen) {
	uint8_t *out = (uint8_t *)pout;
	blake2b_state blake_state;
	uint8_t outlen_bytes[sizeof(uint32_t)] = { 0 };
	int ret = -1;

	if (outlen > UINT32_MAX) {
		goto fail;
	}

	/* Ensure little-endian byte order! */
	store32(outlen_bytes, (uint32_t)outlen);

#define TRY(statement)                                                         \
	do {                                                                       \
		ret = statement;                                                       \
		if (ret < 0) {                                                         \
			goto fail;                                                         \
		}                                                                      \
	} while ((void)0, 0)

	if (outlen <= BLAKE2B_OUTBYTES) {
		TRY(rx_blake2b_init(&blake_state, outlen));
		TRY(rx_blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
		TRY(rx_blake2b_update(&blake_state, in, inlen));
		TRY(rx_blake2b_final(&blake_state, out, outlen));
	}
	else {
		uint32_t toproduce;
		uint8_t out_buffer[BLAKE2B_OUTBYTES];
		uint8_t in_buffer[BLAKE2B_OUTBYTES];
		TRY(rx_blake2b_init(&blake_state, BLAKE2B_OUTBYTES));
		TRY(rx_blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
		TRY(rx_blake2b_update(&blake_state, in, inlen));
		TRY(rx_blake2b_final(&blake_state, out_buffer, BLAKE2B_OUTBYTES));
		memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
		out += BLAKE2B_OUTBYTES / 2;
		toproduce = (uint32_t)outlen - BLAKE2B_OUTBYTES / 2;

		while (toproduce > BLAKE2B_OUTBYTES) {
			memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
			TRY(rx_blake2b(out_buffer, BLAKE2B_OUTBYTES, in_buffer,
				BLAKE2B_OUTBYTES));
			memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
			out += BLAKE2B_OUTBYTES / 2;
			toproduce -= BLAKE2B_OUTBYTES / 2;
		}

		memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
		TRY(rx_blake2b(out_buffer, toproduce, in_buffer, BLAKE2B_OUTBYTES));
		memcpy(out, out_buffer, toproduce);
	}
fail:
	//clear_internal_memory(&blake_state, sizeof(blake_state));
	return ret;
#undef TRY
}
/* Argon2 Team - End Code */

