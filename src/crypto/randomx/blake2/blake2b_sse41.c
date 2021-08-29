/*
 * Copyright (c) 2018-2019, tevador <tevador@gmail.com>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>

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

#if defined(_M_X64) || defined(__x86_64__)

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "crypto/randomx/blake2/blake2.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <smmintrin.h>
#include "blake2b-round.h"


extern const uint64_t blake2b_IV[8];


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


void rx_blake2b_compress_sse41(blake2b_state* S, const uint8_t *block)
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
#endif
