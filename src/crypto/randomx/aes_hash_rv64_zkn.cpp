/*
Copyright (c) 2025 SChernykh   <https://github.com/SChernykh>
Copyright (c) 2025 XMRig       <support@xmrig.com>

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

#include "crypto/randomx/aes_hash.hpp"
#include "crypto/randomx/randomx.h"
#include "crypto/rx/Profiler.h"

static FORCE_INLINE uint64_t aes64esm(uint64_t a, uint64_t b) { uint64_t t; asm("aes64esm %0,%1,%2" : "=r"(t) : "r"(a), "r"(b)); return t; }
static FORCE_INLINE uint64_t aes64dsm(uint64_t a, uint64_t b) { uint64_t t; asm("aes64dsm %0,%1,%2" : "=r"(t) : "r"(a), "r"(b)); return t; }

static FORCE_INLINE void aesenc_zkn(uint64_t& in0, uint64_t& in1, uint64_t key0, uint64_t key1)
{
	const uint64_t t0 = aes64esm(in0, in1);
	const uint64_t t1 = aes64esm(in1, in0);

	in0 = t0 ^ key0;
	in1 = t1 ^ key1;
}

static FORCE_INLINE void aesdec_zkn(uint64_t& in0, uint64_t& in1, uint64_t key0, uint64_t key1)
{
	const uint64_t t0 = aes64dsm(in0, in1);
	const uint64_t t1 = aes64dsm(in1, in0);

	in0 = t0 ^ key0;
	in1 = t1 ^ key1;
}

static const uint64_t AES_HASH_1R_STATE[4][2] = {
	{ 0x9fa856de92b52c0dull, 0xd7983aadcc82db47ull },
	{ 0x15c7b798338d996eull, 0xace78057f59e125aull },
	{ 0xae62c7d06a770017ull, 0xe8a07ce45079506bull },
	{ 0x07ad828d630a240cull, 0x7e99494879a10005ull },
};

static const uint64_t AES_HASH_1R_XKEY[2][2] = {
	{ 0x8b24949ff6fa8389ull, 0x0689020190dc56bfull },
	{ 0x51f4e03c61b263d1ull, 0xed18f99bee1043c6ull },
};

void hashAes1Rx4_zkn(const void *input, size_t inputSize, void *hash)
{
	const uint64_t* inptr = (uint64_t*)input;
	const uint64_t* inputEnd = inptr + inputSize / sizeof(uint64_t);

	uint64_t state[4][2];
	memcpy(state, AES_HASH_1R_STATE, sizeof(state));

	while (inptr < inputEnd) {
		aesenc_zkn(state[0][0], state[0][1], inptr[0], inptr[1]);
		aesdec_zkn(state[1][0], state[1][1], inptr[2], inptr[3]);
		aesenc_zkn(state[2][0], state[2][1], inptr[4], inptr[5]);
		aesdec_zkn(state[3][0], state[3][1], inptr[6], inptr[7]);

		inptr += 8;
	}

	for (int i = 0; i < 2; ++i) {
		const uint64_t xkey0 = AES_HASH_1R_XKEY[i][0];
		const uint64_t xkey1 = AES_HASH_1R_XKEY[i][1];

		aesenc_zkn(state[0][0], state[0][1], xkey0, xkey1);
		aesdec_zkn(state[1][0], state[1][1], xkey0, xkey1);
		aesenc_zkn(state[2][0], state[2][1], xkey0, xkey1);
		aesdec_zkn(state[3][0], state[3][1], xkey0, xkey1);
	}

	memcpy(hash, state, sizeof(state));
}

static const uint64_t AES_GEN_1R_KEY[4][2] = {
	{ 0x627166096daca553ull, 0xb4f44917dbb5552bull },
	{ 0x846a710d6d7caf07ull, 0x0da1dc4e1725d378ull },
	{ 0x9f947ec63f1262f1ull, 0x3e20e345f4c0794full },
	{ 0xb1ba317c6aef8135ull, 0x4916915416314c88ull },
};

void fillAes1Rx4_zkn(void *state, size_t outputSize, void *buffer)
{
	uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	uint64_t key[4][2];
	memcpy(key, AES_GEN_1R_KEY, sizeof(key));

	uint64_t cur_state[4][2];
	memcpy(cur_state, state, sizeof(cur_state));

	while (outptr < outputEnd) {
		aesdec_zkn(cur_state[0][0], cur_state[0][1], key[0][0], key[0][1]);
		aesenc_zkn(cur_state[1][0], cur_state[1][1], key[1][0], key[1][1]);
		aesdec_zkn(cur_state[2][0], cur_state[2][1], key[2][0], key[2][1]);
		aesenc_zkn(cur_state[3][0], cur_state[3][1], key[3][0], key[3][1]);

		memcpy(outptr, cur_state, sizeof(cur_state));
		outptr += 64;
	}

	memcpy(state, cur_state, sizeof(cur_state));
}

void fillAes4Rx4_zkn(void *state, size_t outputSize, void *buffer)
{
	uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	uint64_t key[8][2];
	memcpy(key, RandomX_CurrentConfig.fillAes4Rx4_Key, sizeof(key));

	uint64_t cur_state[4][2];
	memcpy(cur_state, state, sizeof(cur_state));

	while (outptr < outputEnd) {
		aesdec_zkn(cur_state[0][0], cur_state[0][1], key[0][0], key[0][1]);
		aesenc_zkn(cur_state[1][0], cur_state[1][1], key[0][0], key[0][1]);
		aesdec_zkn(cur_state[2][0], cur_state[2][1], key[4][0], key[4][1]);
		aesenc_zkn(cur_state[3][0], cur_state[3][1], key[4][0], key[4][1]);

		aesdec_zkn(cur_state[0][0], cur_state[0][1], key[1][0], key[1][1]);
		aesenc_zkn(cur_state[1][0], cur_state[1][1], key[1][0], key[1][1]);
		aesdec_zkn(cur_state[2][0], cur_state[2][1], key[5][0], key[5][1]);
		aesenc_zkn(cur_state[3][0], cur_state[3][1], key[5][0], key[5][1]);

		aesdec_zkn(cur_state[0][0], cur_state[0][1], key[2][0], key[2][1]);
		aesenc_zkn(cur_state[1][0], cur_state[1][1], key[2][0], key[2][1]);
		aesdec_zkn(cur_state[2][0], cur_state[2][1], key[6][0], key[6][1]);
		aesenc_zkn(cur_state[3][0], cur_state[3][1], key[6][0], key[6][1]);

		aesdec_zkn(cur_state[0][0], cur_state[0][1], key[3][0], key[3][1]);
		aesenc_zkn(cur_state[1][0], cur_state[1][1], key[3][0], key[3][1]);
		aesdec_zkn(cur_state[2][0], cur_state[2][1], key[7][0], key[7][1]);
		aesenc_zkn(cur_state[3][0], cur_state[3][1], key[7][0], key[7][1]);

		memcpy(outptr, cur_state, sizeof(cur_state));
		outptr += 64;
	}
}

void hashAndFillAes1Rx4_zkn(void *scratchpad, size_t scratchpadSize, void *hash, void* fill_state)
{
	PROFILE_SCOPE(RandomX_AES);

	uint64_t* scratchpadPtr = (uint64_t*)scratchpad;
	const uint64_t* scratchpadEnd = scratchpadPtr + scratchpadSize / sizeof(uint64_t);

	uint64_t cur_hash_state[4][2];
	memcpy(cur_hash_state, AES_HASH_1R_STATE, sizeof(cur_hash_state));

	uint64_t key[4][2];
	memcpy(key, AES_GEN_1R_KEY, sizeof(key));

	uint64_t cur_fill_state[4][2];
	memcpy(cur_fill_state, fill_state, sizeof(cur_fill_state));

	while (scratchpadPtr < scratchpadEnd) {
		aesenc_zkn(cur_hash_state[0][0], cur_hash_state[0][1], scratchpadPtr[0], scratchpadPtr[1]);
		aesdec_zkn(cur_hash_state[1][0], cur_hash_state[1][1], scratchpadPtr[2], scratchpadPtr[3]);
		aesenc_zkn(cur_hash_state[2][0], cur_hash_state[2][1], scratchpadPtr[4], scratchpadPtr[5]);
		aesdec_zkn(cur_hash_state[3][0], cur_hash_state[3][1], scratchpadPtr[6], scratchpadPtr[7]);

		aesdec_zkn(cur_fill_state[0][0], cur_fill_state[0][1], key[0][0], key[0][1]);
		aesenc_zkn(cur_fill_state[1][0], cur_fill_state[1][1], key[1][0], key[1][1]);
		aesdec_zkn(cur_fill_state[2][0], cur_fill_state[2][1], key[2][0], key[2][1]);
		aesenc_zkn(cur_fill_state[3][0], cur_fill_state[3][1], key[3][0], key[3][1]);

		memcpy(scratchpadPtr, cur_fill_state, sizeof(cur_fill_state));
		scratchpadPtr += 8;
	}

	memcpy(fill_state, cur_fill_state, sizeof(cur_fill_state));

	for (int i = 0; i < 2; ++i) {
		const uint64_t xkey0 = AES_HASH_1R_XKEY[i][0];
		const uint64_t xkey1 = AES_HASH_1R_XKEY[i][1];

		aesenc_zkn(cur_hash_state[0][0], cur_hash_state[0][1], xkey0, xkey1);
		aesdec_zkn(cur_hash_state[1][0], cur_hash_state[1][1], xkey0, xkey1);
		aesenc_zkn(cur_hash_state[2][0], cur_hash_state[2][1], xkey0, xkey1);
		aesdec_zkn(cur_hash_state[3][0], cur_hash_state[3][1], xkey0, xkey1);
	}

	memcpy(hash, cur_hash_state, sizeof(cur_hash_state));
}
