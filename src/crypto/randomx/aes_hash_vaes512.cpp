/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>
Copyright (c) 2026 XMRig       <support@xmrig.com>
Copyright (c) 2026 SChernykh   <https://github.com/SChernykh>

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

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#define REVERSE_4(A, B, C, D) D, C, B, A

alignas(64) static const uint32_t AES_HASH_1R_STATE[] = {
	REVERSE_4(0xd7983aad, 0xcc82db47, 0x9fa856de, 0x92b52c0d),
	REVERSE_4(0xace78057, 0xf59e125a, 0x15c7b798, 0x338d996e),
	REVERSE_4(0xe8a07ce4, 0x5079506b, 0xae62c7d0, 0x6a770017),
	REVERSE_4(0x7e994948, 0x79a10005, 0x07ad828d, 0x630a240c)
};

alignas(64) static const uint32_t AES_GEN_1R_KEY[] = {
	REVERSE_4(0xb4f44917, 0xdbb5552b, 0x62716609, 0x6daca553),
	REVERSE_4(0x0da1dc4e, 0x1725d378, 0x846a710d, 0x6d7caf07),
	REVERSE_4(0x3e20e345, 0xf4c0794f, 0x9f947ec6, 0x3f1262f1),
	REVERSE_4(0x49169154, 0x16314c88, 0xb1ba317c, 0x6aef8135)
};

alignas(64) static const uint32_t AES_HASH_1R_XKEY0[] = {
	REVERSE_4(0x06890201, 0x90dc56bf, 0x8b24949f, 0xf6fa8389),
	REVERSE_4(0x06890201, 0x90dc56bf, 0x8b24949f, 0xf6fa8389),
	REVERSE_4(0x06890201, 0x90dc56bf, 0x8b24949f, 0xf6fa8389),
	REVERSE_4(0x06890201, 0x90dc56bf, 0x8b24949f, 0xf6fa8389)
};

alignas(64) static const uint32_t AES_HASH_1R_XKEY1[] = {
	REVERSE_4(0xed18f99b, 0xee1043c6, 0x51f4e03c, 0x61b263d1),
	REVERSE_4(0xed18f99b, 0xee1043c6, 0x51f4e03c, 0x61b263d1),
	REVERSE_4(0xed18f99b, 0xee1043c6, 0x51f4e03c, 0x61b263d1),
	REVERSE_4(0xed18f99b, 0xee1043c6, 0x51f4e03c, 0x61b263d1)
};

void hashAndFillAes1Rx4_VAES512(void *scratchpad, size_t scratchpadSize, void *hash, void* fill_state)
{
	uint8_t* scratchpadPtr = (uint8_t*)scratchpad;
	const uint8_t* scratchpadEnd = scratchpadPtr + scratchpadSize;

	const __m512i fill_key = _mm512_load_si512(AES_GEN_1R_KEY);

	const __m512i initial_hash_state = _mm512_load_si512(AES_HASH_1R_STATE);
	const __m512i initial_fill_state = _mm512_load_si512(fill_state);

	constexpr uint8_t mask = 0b11001100;

	// enc_data[0] = hash_state[0]
	// enc_data[1] = fill_state[1]
	// enc_data[2] = hash_state[2]
	// enc_data[3] = fill_state[3]
	__m512i enc_data = _mm512_mask_blend_epi64(mask, initial_hash_state, initial_fill_state);

	// dec_data[0] = fill_state[0]
	// dec_data[1] = hash_state[1]
	// dec_data[2] = fill_state[2]
	// dec_data[3] = hash_state[3]
	__m512i dec_data = _mm512_mask_blend_epi64(mask, initial_fill_state, initial_hash_state);

	constexpr int PREFETCH_DISTANCE = 7168;

	const uint8_t* prefetchPtr = scratchpadPtr + PREFETCH_DISTANCE;
	scratchpadEnd -= PREFETCH_DISTANCE;

	for (const uint8_t* p = scratchpadPtr; p < prefetchPtr; p += 256) {
		_mm_prefetch((const char*)(p +   0), _MM_HINT_T0);
		_mm_prefetch((const char*)(p +  64), _MM_HINT_T0);
		_mm_prefetch((const char*)(p + 128), _MM_HINT_T0);
		_mm_prefetch((const char*)(p + 192), _MM_HINT_T0);
	}

	for (int i = 0; i < 2; ++i) {
		while (scratchpadPtr < scratchpadEnd) {
			const __m512i scratchpad_data = _mm512_load_si512(scratchpadPtr);

			// enc_key[0] = scratchpad_data[0]
			// enc_key[1] = fill_key[1]
			// enc_key[2] = scratchpad_data[2]
			// enc_key[3] = fill_key[3]
			enc_data = _mm512_aesenc_epi128(enc_data, _mm512_mask_blend_epi64(mask, scratchpad_data, fill_key));

			// dec_key[0] = fill_key[0]
			// dec_key[1] = scratchpad_data[1]
			// dec_key[2] = fill_key[2]
			// dec_key[3] = scratchpad_data[3]
			dec_data = _mm512_aesdec_epi128(dec_data, _mm512_mask_blend_epi64(mask, fill_key, scratchpad_data));

			// fill_state[0] = dec_data[0]
			// fill_state[1] = enc_data[1]
			// fill_state[2] = dec_data[2]
			// fill_state[3] = enc_data[3]
			_mm512_store_si512(scratchpadPtr, _mm512_mask_blend_epi64(mask, dec_data, enc_data));

			_mm_prefetch((const char*)prefetchPtr, _MM_HINT_T0);

			scratchpadPtr += 64;
			prefetchPtr += 64;
		}
		prefetchPtr = (const uint8_t*) scratchpad;
		scratchpadEnd += PREFETCH_DISTANCE;
	}

	_mm512_store_si512(fill_state, _mm512_mask_blend_epi64(mask, dec_data, enc_data));

	//two extra rounds to achieve full diffusion
	const __m512i xkey0 = _mm512_load_si512(AES_HASH_1R_XKEY0);
	const __m512i xkey1 = _mm512_load_si512(AES_HASH_1R_XKEY1);

	enc_data = _mm512_aesenc_epi128(enc_data, xkey0);
	dec_data = _mm512_aesdec_epi128(dec_data, xkey0);
	enc_data = _mm512_aesenc_epi128(enc_data, xkey1);
	dec_data = _mm512_aesdec_epi128(dec_data, xkey1);

	//output hash
	_mm512_store_si512(hash, _mm512_mask_blend_epi64(mask, enc_data, dec_data));

	// Just in case
	_mm256_zeroupper();
}
