/*-
 * Copyright 2009 Colin Percival
 * Copyright 2013-2019 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 *
 * This is a proof-of-work focused fork of yescrypt, including reference and
 * cut-down implementation of the obsolete yescrypt 0.5 (based off its first
 * submission to PHC back in 2014) and a new proof-of-work specific variation
 * known as yespower 1.0.  The former is intended as an upgrade for
 * cryptocurrencies that already use yescrypt 0.5 and the latter may be used
 * as a further upgrade (hard fork) by those and other cryptocurrencies.  The
 * version of algorithm to use is requested through parameters, allowing for
 * both algorithms to co-exist in client and miner implementations (such as in
 * preparation for a hard-fork).
 *
 * This is the reference implementation.  Its purpose is to provide a simple
 * human- and machine-readable specification that implementations intended
 * for actual use should be tested against.  It is deliberately mostly not
 * optimized, and it is not meant to be used in production.  Instead, use
 * yespower-opt.c.
 */

#if !defined(_MSC_VER)
#warning "This reference implementation is deliberately mostly not optimized. Use yespower-opt.c instead unless you're testing (against) the reference implementation on purpose."
#endif

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"
#include "sysendian.h"

#include "yespower.h"

static void blkcpy(uint32_t *dst, const uint32_t *src, size_t count)
{
	do {
		*dst++ = *src++;
	} while (--count);
}

static void blkxor(uint32_t *dst, const uint32_t *src, size_t count)
{
	do {
		*dst++ ^= *src++;
	} while (--count);
}

/**
 * salsa20(B):
 * Apply the Salsa20 core to the provided block.
 */
static void salsa20(uint32_t B[16], uint32_t rounds)
{
	uint32_t x[16];
	size_t i;

	/* SIMD unshuffle */
	for (i = 0; i < 16; i++)
		x[i * 5 % 16] = B[i];

	for (i = 0; i < rounds; i += 2) {
#define R(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
		/* Operate on columns */
		x[ 4] ^= R(x[ 0]+x[12], 7);  x[ 8] ^= R(x[ 4]+x[ 0], 9);
		x[12] ^= R(x[ 8]+x[ 4],13);  x[ 0] ^= R(x[12]+x[ 8],18);

		x[ 9] ^= R(x[ 5]+x[ 1], 7);  x[13] ^= R(x[ 9]+x[ 5], 9);
		x[ 1] ^= R(x[13]+x[ 9],13);  x[ 5] ^= R(x[ 1]+x[13],18);

		x[14] ^= R(x[10]+x[ 6], 7);  x[ 2] ^= R(x[14]+x[10], 9);
		x[ 6] ^= R(x[ 2]+x[14],13);  x[10] ^= R(x[ 6]+x[ 2],18);

		x[ 3] ^= R(x[15]+x[11], 7);  x[ 7] ^= R(x[ 3]+x[15], 9);
		x[11] ^= R(x[ 7]+x[ 3],13);  x[15] ^= R(x[11]+x[ 7],18);

		/* Operate on rows */
		x[ 1] ^= R(x[ 0]+x[ 3], 7);  x[ 2] ^= R(x[ 1]+x[ 0], 9);
		x[ 3] ^= R(x[ 2]+x[ 1],13);  x[ 0] ^= R(x[ 3]+x[ 2],18);

		x[ 6] ^= R(x[ 5]+x[ 4], 7);  x[ 7] ^= R(x[ 6]+x[ 5], 9);
		x[ 4] ^= R(x[ 7]+x[ 6],13);  x[ 5] ^= R(x[ 4]+x[ 7],18);

		x[11] ^= R(x[10]+x[ 9], 7);  x[ 8] ^= R(x[11]+x[10], 9);
		x[ 9] ^= R(x[ 8]+x[11],13);  x[10] ^= R(x[ 9]+x[ 8],18);

		x[12] ^= R(x[15]+x[14], 7);  x[13] ^= R(x[12]+x[15], 9);
		x[14] ^= R(x[13]+x[12],13);  x[15] ^= R(x[14]+x[13],18);
#undef R
	}

	/* SIMD shuffle */
	for (i = 0; i < 16; i++)
		B[i] += x[i * 5 % 16];
}

/**
 * blockmix_salsa(B):
 * Compute B = BlockMix_{salsa20, 1}(B).  The input B must be 128 bytes in
 * length.
 */
static void blockmix_salsa(uint32_t *B, uint32_t rounds)
{
	uint32_t X[16];
	size_t i;

	/* 1: X <-- B_{2r - 1} */
	blkcpy(X, &B[16], 16);

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < 2; i++) {
		/* 3: X <-- H(X xor B_i) */
		blkxor(X, &B[i * 16], 16);
		salsa20(X, rounds);

		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		blkcpy(&B[i * 16], X, 16);
	}
}

/*
 * These are tunable, but they must meet certain constraints and are part of
 * what defines a yespower version.
 */
#define PWXsimple 2
#define PWXgather 4
/* Version 0.5 */
#define PWXrounds_0_5 6
#define Swidth_0_5 8
/* Version 1.0 */
#define PWXrounds_1_0 3
#define Swidth_1_0 11

/* Derived values.  Not tunable on their own. */
#define PWXbytes (PWXgather * PWXsimple * 8)
#define PWXwords (PWXbytes / sizeof(uint32_t))
#define rmin ((PWXbytes + 127) / 128)

/* Runtime derived values.  Not tunable on their own. */
#define Swidth_to_Sbytes1(Swidth) ((1 << Swidth) * PWXsimple * 8)
#define Swidth_to_Smask(Swidth) (((1 << Swidth) - 1) * PWXsimple * 8)

typedef struct {
	yespower_version_t version;
	uint32_t salsa20_rounds;
	uint32_t PWXrounds, Swidth, Sbytes, Smask;
	uint32_t *S;
	uint32_t (*S0)[2], (*S1)[2], (*S2)[2];
	size_t w;
} pwxform_ctx_t;

/**
 * pwxform(B):
 * Transform the provided block using the provided S-boxes.
 */
static void pwxform(uint32_t *B, pwxform_ctx_t *ctx)
{
	uint32_t (*X)[PWXsimple][2] = (uint32_t (*)[PWXsimple][2])B;
	uint32_t (*S0)[2] = ctx->S0, (*S1)[2] = ctx->S1, (*S2)[2] = ctx->S2;
	uint32_t Smask = ctx->Smask;
	size_t w = ctx->w;
	size_t i, j, k;

	/* 1: for i = 0 to PWXrounds - 1 do */
	for (i = 0; i < ctx->PWXrounds; i++) {
		/* 2: for j = 0 to PWXgather - 1 do */
		for (j = 0; j < PWXgather; j++) {
			uint32_t xl = X[j][0][0];
			uint32_t xh = X[j][0][1];
			uint32_t (*p0)[2], (*p1)[2];

			/* 3: p0 <-- (lo(B_{j,0}) & Smask) / (PWXsimple * 8) */
			p0 = S0 + (xl & Smask) / sizeof(*S0);
			/* 4: p1 <-- (hi(B_{j,0}) & Smask) / (PWXsimple * 8) */
			p1 = S1 + (xh & Smask) / sizeof(*S1);

			/* 5: for k = 0 to PWXsimple - 1 do */
			for (k = 0; k < PWXsimple; k++) {
				uint64_t x, s0, s1;

				/* 6: B_{j,k} <-- (hi(B_{j,k}) * lo(B_{j,k}) + S0_{p0,k}) xor S1_{p1,k} */
				s0 = ((uint64_t)p0[k][1] << 32) + p0[k][0];
				s1 = ((uint64_t)p1[k][1] << 32) + p1[k][0];

				xl = X[j][k][0];
				xh = X[j][k][1];

				x = (uint64_t)xh * xl;
				x += s0;
				x ^= s1;

				X[j][k][0] = x;
				X[j][k][1] = x >> 32;
			}

			if (ctx->version != YESPOWER_0_5 &&
			    (i == 0 || j < PWXgather / 2)) {
				if (j & 1) {
					for (k = 0; k < PWXsimple; k++) {
						S1[w][0] = X[j][k][0];
						S1[w][1] = X[j][k][1];
						w++;
					}
				} else {
					for (k = 0; k < PWXsimple; k++) {
						S0[w + k][0] = X[j][k][0];
						S0[w + k][1] = X[j][k][1];
					}
				}
			}
		}
	}

	if (ctx->version != YESPOWER_0_5) {
		/* 14: (S0, S1, S2) <-- (S2, S0, S1) */
		ctx->S0 = S2;
		ctx->S1 = S0;
		ctx->S2 = S1;
		/* 15: w <-- w mod 2^Swidth */
		ctx->w = w & ((1 << ctx->Swidth) * PWXsimple - 1);
	}
}

/**
 * blockmix_pwxform(B, ctx, r):
 * Compute B = BlockMix_pwxform{salsa20, ctx, r}(B).  The input B must be
 * 128r bytes in length.
 */
static void blockmix_pwxform(uint32_t *B, pwxform_ctx_t *ctx, size_t r)
{
	uint32_t X[PWXwords];
	size_t r1, i;

	/* Convert 128-byte blocks to PWXbytes blocks */
	/* 1: r_1 <-- 128r / PWXbytes */
	r1 = 128 * r / PWXbytes;

	/* 2: X <-- B'_{r_1 - 1} */
	blkcpy(X, &B[(r1 - 1) * PWXwords], PWXwords);

	/* 3: for i = 0 to r_1 - 1 do */
	for (i = 0; i < r1; i++) {
		/* 4: if r_1 > 1 */
		if (r1 > 1) {
			/* 5: X <-- X xor B'_i */
			blkxor(X, &B[i * PWXwords], PWXwords);
		}

		/* 7: X <-- pwxform(X) */
		pwxform(X, ctx);

		/* 8: B'_i <-- X */
		blkcpy(&B[i * PWXwords], X, PWXwords);
	}

	/* 10: i <-- floor((r_1 - 1) * PWXbytes / 64) */
	i = (r1 - 1) * PWXbytes / 64;

	/* 11: B_i <-- H(B_i) */
	salsa20(&B[i * 16], ctx->salsa20_rounds);

#if 1 /* No-op with our current pwxform settings, but do it to make sure */
	/* 12: for i = i + 1 to 2r - 1 do */
	for (i++; i < 2 * r; i++) {
		/* 13: B_i <-- H(B_i xor B_{i-1}) */
		blkxor(&B[i * 16], &B[(i - 1) * 16], 16);
		salsa20(&B[i * 16], ctx->salsa20_rounds);
	}
#endif
}

/**
 * integerify(B, r):
 * Return the result of parsing B_{2r-1} as a little-endian integer.
 */
static uint32_t integerify(const uint32_t *B, size_t r)
{
/*
 * Our 32-bit words are in host byte order.  Also, they are SIMD-shuffled, but
 * we only care about the least significant 32 bits anyway.
 */
	const uint32_t *X = &B[(2 * r - 1) * 16];
	return X[0];
}

/**
 * p2floor(x):
 * Largest power of 2 not greater than argument.
 */
static uint32_t p2floor(uint32_t x)
{
	uint32_t y;
	while ((y = x & (x - 1)))
		x = y;
	return x;
}

/**
 * wrap(x, i):
 * Wrap x to the range 0 to i-1.
 */
static uint32_t wrap(uint32_t x, uint32_t i)
{
	uint32_t n = p2floor(i);
	return (x & (n - 1)) + (i - n);
}

/**
 * smix1(B, r, N, V, X, ctx):
 * Compute first loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage X must be 128r bytes in length.
 */
static void smix1(uint32_t *B, size_t r, uint32_t N,
    uint32_t *V, uint32_t *X, pwxform_ctx_t *ctx)
{
	size_t s = 32 * r;
	uint32_t i, j;
	size_t k;

	/* 1: X <-- B */
	for (k = 0; k < 2 * r; k++)
		for (i = 0; i < 16; i++)
			X[k * 16 + i] = le32dec(&B[k * 16 + (i * 5 % 16)]);

	if (ctx->version != YESPOWER_0_5) {
		for (k = 1; k < r; k++) {
			blkcpy(&X[k * 32], &X[(k - 1) * 32], 32);
			blockmix_pwxform(&X[k * 32], ctx, 1);
		}
	}

	/* 2: for i = 0 to N - 1 do */
	for (i = 0; i < N; i++) {
		/* 3: V_i <-- X */
		blkcpy(&V[i * s], X, s);

		if (i > 1) {
			/* j <-- Wrap(Integerify(X), i) */
			j = wrap(integerify(X, r), i);

			/* X <-- X xor V_j */
			blkxor(X, &V[j * s], s);
		}

		/* 4: X <-- H(X) */
		if (V != ctx->S)
			blockmix_pwxform(X, ctx, r);
		else
			blockmix_salsa(X, ctx->salsa20_rounds);
	}

	/* B' <-- X */
	for (k = 0; k < 2 * r; k++)
		for (i = 0; i < 16; i++)
			le32enc(&B[k * 16 + (i * 5 % 16)], X[k * 16 + i]);
}

/**
 * smix2(B, r, N, Nloop, V, X, ctx):
 * Compute second loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage X must be 128r bytes in length.  The value N must be a power of 2
 * greater than 1.
 */
static void smix2(uint32_t *B, size_t r, uint32_t N, uint32_t Nloop,
    uint32_t *V, uint32_t *X, pwxform_ctx_t *ctx)
{
	size_t s = 32 * r;
	uint32_t i, j;
	size_t k;

	/* X <-- B */
	for (k = 0; k < 2 * r; k++)
		for (i = 0; i < 16; i++)
			X[k * 16 + i] = le32dec(&B[k * 16 + (i * 5 % 16)]);

	/* 6: for i = 0 to N - 1 do */
	for (i = 0; i < Nloop; i++) {
		/* 7: j <-- Integerify(X) mod N */
		j = integerify(X, r) & (N - 1);

		/* 8.1: X <-- X xor V_j */
		blkxor(X, &V[j * s], s);
		/* V_j <-- X */
		if (Nloop != 2)
			blkcpy(&V[j * s], X, s);

		/* 8.2: X <-- H(X) */
		blockmix_pwxform(X, ctx, r);
	}

	/* 10: B' <-- X */
	for (k = 0; k < 2 * r; k++)
		for (i = 0; i < 16; i++)
			le32enc(&B[k * 16 + (i * 5 % 16)], X[k * 16 + i]);
}

/**
 * smix(B, r, N, p, t, V, X, ctx):
 * Compute B = SMix_r(B, N).  The input B must be 128rp bytes in length; the
 * temporary storage V must be 128rN bytes in length; the temporary storage
 * X must be 128r bytes in length.  The value N must be a power of 2 and at
 * least 16.
 */
static void smix(uint32_t *B, size_t r, uint32_t N,
    uint32_t *V, uint32_t *X, pwxform_ctx_t *ctx)
{
	uint32_t Nloop_all = (N + 2) / 3; /* 1/3, round up */
	uint32_t Nloop_rw = Nloop_all;

	Nloop_all++; Nloop_all &= ~(uint32_t)1; /* round up to even */
	if (ctx->version == YESPOWER_0_5) {
		Nloop_rw &= ~(uint32_t)1; /* round down to even */
	} else {
		Nloop_rw++; Nloop_rw &= ~(uint32_t)1; /* round up to even */
	}

	smix1(B, 1, ctx->Sbytes / 128, ctx->S, X, ctx);
	smix1(B, r, N, V, X, ctx);
	smix2(B, r, N, Nloop_rw /* must be > 2 */, V, X, ctx);
	smix2(B, r, N, Nloop_all - Nloop_rw /* 0 or 2 */, V, X, ctx);
}

/**
 * yespower(local, src, srclen, params, dst):
 * Compute yespower(src[0 .. srclen - 1], N, r), to be checked for "< target".
 *
 * Return 0 on success; or -1 on error.
 */
int yespower(yespower_local_t *local,
    const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst)
{
	yespower_version_t version = params->version;
	uint32_t N = params->N;
	uint32_t r = params->r;
	const uint8_t *pers = params->pers;
	size_t perslen = params->perslen;
	int retval = -1;
	size_t B_size, V_size;
	uint32_t *B, *V, *X, *S;
	pwxform_ctx_t ctx;
	uint32_t sha256[8];

	memset(dst, 0xff, sizeof(*dst));

	/* Sanity-check parameters */
	if ((version != YESPOWER_0_5 && version != YESPOWER_1_0) ||
	    N < 1024 || N > 512 * 1024 || r < 8 || r > 32 ||
	    (N & (N - 1)) != 0 || r < rmin ||
	    (!pers && perslen)) {
		errno = EINVAL;
		return -1;
	}

	/* Allocate memory */
	B_size = (size_t)128 * r;
	V_size = B_size * N;
	if ((V = malloc(V_size)) == NULL)
		return -1;
	if ((B = malloc(B_size)) == NULL)
		goto free_V;
	if ((X = malloc(B_size)) == NULL)
		goto free_B;
	ctx.version = version;
	if (version == YESPOWER_0_5) {
		ctx.salsa20_rounds = 8;
		ctx.PWXrounds = PWXrounds_0_5;
		ctx.Swidth = Swidth_0_5;
		ctx.Sbytes = 2 * Swidth_to_Sbytes1(ctx.Swidth);
	} else {
		ctx.salsa20_rounds = 2;
		ctx.PWXrounds = PWXrounds_1_0;
		ctx.Swidth = Swidth_1_0;
		ctx.Sbytes = 3 * Swidth_to_Sbytes1(ctx.Swidth);
	}
	if ((S = malloc(ctx.Sbytes)) == NULL)
		goto free_X;
	ctx.S = S;
	ctx.S0 = (uint32_t (*)[2])S;
	ctx.S1 = ctx.S0 + (1 << ctx.Swidth) * PWXsimple;
	ctx.S2 = ctx.S1 + (1 << ctx.Swidth) * PWXsimple;
	ctx.Smask = Swidth_to_Smask(ctx.Swidth);
	ctx.w = 0;

	SHA256_Buf(src, srclen, (uint8_t *)sha256);

	if (version != YESPOWER_0_5) {
		if (pers) {
			src = pers;
			srclen = perslen;
		} else {
			srclen = 0;
		}
	}

	/* 1: (B_0 ... B_{p-1}) <-- PBKDF2(P, S, 1, p * MFLen) */
	PBKDF2_SHA256((uint8_t *)sha256, sizeof(sha256),
	    src, srclen, 1, (uint8_t *)B, B_size);

	blkcpy(sha256, B, sizeof(sha256) / sizeof(sha256[0]));

	/* 3: B_i <-- MF(B_i, N) */
	smix(B, r, N, V, X, &ctx);

	if (version == YESPOWER_0_5) {
		/* 5: DK <-- PBKDF2(P, B, 1, dkLen) */
		PBKDF2_SHA256((uint8_t *)sha256, sizeof(sha256),
		    (uint8_t *)B, B_size, 1, (uint8_t *)dst, sizeof(*dst));

		if (pers) {
			HMAC_SHA256_Buf(dst, sizeof(*dst), pers, perslen,
			    (uint8_t *)sha256);
			SHA256_Buf(sha256, sizeof(sha256), (uint8_t *)dst);
		}
	} else {
		HMAC_SHA256_Buf((uint8_t *)B + B_size - 64, 64,
		    sha256, sizeof(sha256), (uint8_t *)dst);
	}

	/* Success! */
	retval = 0;

	/* Free memory */
	free(S);
free_X:
	free(X);
free_B:
	free(B);
free_V:
	free(V);

	return retval;
}

int yespower_tls(const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst)
{
/* The reference implementation doesn't use thread-local storage */
	return yespower(NULL, src, srclen, params, dst);
}

int yespower_init_local(yespower_local_t *local)
{
/* The reference implementation doesn't use the local structure */
	local->base = local->aligned = NULL;
	local->base_size = local->aligned_size = 0;
	return 0;
}

int yespower_free_local(yespower_local_t *local)
{
/* The reference implementation frees its memory in yespower() */
	(void)local; /* unused */
	return 0;
}
