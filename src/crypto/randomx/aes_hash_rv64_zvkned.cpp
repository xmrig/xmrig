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

#include <riscv_vector.h>

static FORCE_INLINE vuint32m1_t aesenc_zvkned(vuint32m1_t a, vuint32m1_t b) { return __riscv_vaesem_vv_u32m1(a, b, 8); }
static FORCE_INLINE vuint32m1_t aesdec_zvkned(vuint32m1_t a, vuint32m1_t b, vuint32m1_t zero) { return __riscv_vxor_vv_u32m1(__riscv_vaesdm_vv_u32m1(a, zero, 8), b, 8); }

static constexpr uint32_t AES_HASH_1R_STATE02[8] = { 0x92b52c0d, 0x9fa856de, 0xcc82db47, 0xd7983aad, 0x6a770017, 0xae62c7d0, 0x5079506b, 0xe8a07ce4 };
static constexpr uint32_t AES_HASH_1R_STATE13[8] = { 0x338d996e, 0x15c7b798, 0xf59e125a, 0xace78057, 0x630a240c, 0x07ad828d, 0x79a10005, 0x7e994948 };

static constexpr uint32_t AES_GEN_1R_KEY02[8] = { 0x6daca553, 0x62716609, 0xdbb5552b, 0xb4f44917, 0x3f1262f1, 0x9f947ec6, 0xf4c0794f, 0x3e20e345 };
static constexpr uint32_t AES_GEN_1R_KEY13[8] = { 0x6d7caf07, 0x846a710d, 0x1725d378, 0x0da1dc4e, 0x6aef8135, 0xb1ba317c, 0x16314c88, 0x49169154 };

static constexpr uint32_t AES_HASH_1R_XKEY00[8] = { 0xf6fa8389, 0x8b24949f, 0x90dc56bf, 0x06890201, 0xf6fa8389, 0x8b24949f, 0x90dc56bf, 0x06890201 };
static constexpr uint32_t AES_HASH_1R_XKEY11[8] = { 0x61b263d1, 0x51f4e03c, 0xee1043c6, 0xed18f99b, 0x61b263d1, 0x51f4e03c, 0xee1043c6, 0xed18f99b };

static constexpr uint32_t AES_HASH_STRIDE_X2[8] = { 0, 4, 8, 12, 32, 36, 40, 44 };
static constexpr uint32_t AES_HASH_STRIDE_X4[8] = { 0, 4, 8, 12, 64, 68, 72, 76 };

void hashAes1Rx4_zvkned(const void *input, size_t inputSize, void *hash)
{
	const uint8_t* inptr = (const uint8_t*)input;
	const uint8_t* inputEnd = inptr + inputSize;

	//intial state
	vuint32m1_t state02 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE02, 8);
	vuint32m1_t state13 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE13, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);
	const vuint32m1_t zero = {};

	//process 64 bytes at a time in 4 lanes
	while (inptr < inputEnd) {
		state02 = aesenc_zvkned(state02, __riscv_vluxei32_v_u32m1((uint32_t*)inptr + 0, stride, 8));
		state13 = aesdec_zvkned(state13, __riscv_vluxei32_v_u32m1((uint32_t*)inptr + 4, stride, 8), zero);

		inptr += 64;
	}

	//two extra rounds to achieve full diffusion
	const vuint32m1_t xkey00 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY00, 8);
	const vuint32m1_t xkey11 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY11, 8);

	state02 = aesenc_zvkned(state02, xkey00);
	state13 = aesdec_zvkned(state13, xkey00, zero);

	state02 = aesenc_zvkned(state02, xkey11);
	state13 = aesdec_zvkned(state13, xkey11, zero);

	//output hash
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 0, stride, state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 4, stride, state13, 8);
}

void fillAes1Rx4_zvkned(void *state, size_t outputSize, void *buffer)
{
	const uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	const vuint32m1_t key02 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY02, 8);
	const vuint32m1_t key13 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY13, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);
	const vuint32m1_t zero = {};

	vuint32m1_t state02 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 0, stride, 8);
	vuint32m1_t state13 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 4, stride, 8);

	while (outptr < outputEnd) {
		state02 = aesdec_zvkned(state02, key02, zero);
		state13 = aesenc_zvkned(state13, key13);

		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 0, stride, state02, 8);
		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 4, stride, state13, 8);

		outptr += 64;
	}

	__riscv_vsuxei32_v_u32m1((uint32_t*)state + 0, stride, state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)state + 4, stride, state13, 8);
}

void fillAes4Rx4_zvkned(void *state, size_t outputSize, void *buffer)
{
	const uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	const vuint32m1_t stride4 = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X4, 8);

	const vuint32m1_t key04 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 0), stride4, 8);
	const vuint32m1_t key15 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 1), stride4, 8);
	const vuint32m1_t key26 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 2), stride4, 8);
	const vuint32m1_t key37 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 3), stride4, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);
	const vuint32m1_t zero = {};

	vuint32m1_t state02 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 0, stride, 8);
	vuint32m1_t state13 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 4, stride, 8);

	while (outptr < outputEnd) {
		state02 = aesdec_zvkned(state02, key04, zero);
		state13 = aesenc_zvkned(state13, key04);

		state02 = aesdec_zvkned(state02, key15, zero);
		state13 = aesenc_zvkned(state13, key15);

		state02 = aesdec_zvkned(state02, key26, zero);
		state13 = aesenc_zvkned(state13, key26);

		state02 = aesdec_zvkned(state02, key37, zero);
		state13 = aesenc_zvkned(state13, key37);

		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 0, stride, state02, 8);
		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 4, stride, state13, 8);

		outptr += 64;
	}
}

void hashAndFillAes1Rx4_zvkned(void *scratchpad, size_t scratchpadSize, void *hash, void* fill_state)
{
	uint8_t* scratchpadPtr = (uint8_t*)scratchpad;
	const uint8_t* scratchpadEnd = scratchpadPtr + scratchpadSize;

	vuint32m1_t hash_state02 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE02, 8);
	vuint32m1_t hash_state13 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE13, 8);

	const vuint32m1_t key02 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY02, 8);
	const vuint32m1_t key13 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY13, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);
	const vuint32m1_t zero = {};

	vuint32m1_t fill_state02 = __riscv_vluxei32_v_u32m1((uint32_t*)fill_state + 0, stride, 8);
	vuint32m1_t fill_state13 = __riscv_vluxei32_v_u32m1((uint32_t*)fill_state + 4, stride, 8);

	//process 64 bytes at a time in 4 lanes
	while (scratchpadPtr < scratchpadEnd) {
		hash_state02 = aesenc_zvkned(hash_state02, __riscv_vluxei32_v_u32m1((uint32_t*)scratchpadPtr + 0, stride, 8));
		hash_state13 = aesdec_zvkned(hash_state13, __riscv_vluxei32_v_u32m1((uint32_t*)scratchpadPtr + 4, stride, 8), zero);

		fill_state02 = aesdec_zvkned(fill_state02, key02, zero);
		fill_state13 = aesenc_zvkned(fill_state13, key13);

		__riscv_vsuxei32_v_u32m1((uint32_t*)scratchpadPtr + 0, stride, fill_state02, 8);
		__riscv_vsuxei32_v_u32m1((uint32_t*)scratchpadPtr + 4, stride, fill_state13, 8);

		scratchpadPtr += 64;
	}

	__riscv_vsuxei32_v_u32m1((uint32_t*)fill_state + 0, stride, fill_state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)fill_state + 4, stride, fill_state13, 8);

	//two extra rounds to achieve full diffusion
	const vuint32m1_t xkey00 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY00, 8);
	const vuint32m1_t xkey11 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY11, 8);

	hash_state02 = aesenc_zvkned(hash_state02, xkey00);
	hash_state13 = aesdec_zvkned(hash_state13, xkey00, zero);

	hash_state02 = aesenc_zvkned(hash_state02, xkey11);
	hash_state13 = aesdec_zvkned(hash_state13, xkey11, zero);

	//output hash
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 0, stride, hash_state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 4, stride, hash_state13, 8);
}
