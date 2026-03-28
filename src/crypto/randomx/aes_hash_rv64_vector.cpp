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

#include <riscv_vector.h>

#include "crypto/randomx/soft_aes.h"
#include "crypto/randomx/randomx.h"

static FORCE_INLINE vuint32m1_t softaes_vector_double(
	vuint32m1_t in,
	vuint32m1_t key,
	vuint8m1_t i0, vuint8m1_t i1, vuint8m1_t i2, vuint8m1_t i3,
	const uint32_t* lut0, const uint32_t* lut1, const uint32_t *lut2, const uint32_t* lut3)
{
	const vuint8m1_t in8 = __riscv_vreinterpret_v_u32m1_u8m1(in);

	const vuint32m1_t index0 = __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vrgather_vv_u8m1(in8, i0, 32));
	const vuint32m1_t index1 = __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vrgather_vv_u8m1(in8, i1, 32));
	const vuint32m1_t index2 = __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vrgather_vv_u8m1(in8, i2, 32));
	const vuint32m1_t index3 = __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vrgather_vv_u8m1(in8, i3, 32));

	vuint32m1_t s0 = __riscv_vluxei32_v_u32m1(lut0, __riscv_vsll_vx_u32m1(index0, 2, 8), 8);
	vuint32m1_t s1 = __riscv_vluxei32_v_u32m1(lut1, __riscv_vsll_vx_u32m1(index1, 2, 8), 8);
	vuint32m1_t s2 = __riscv_vluxei32_v_u32m1(lut2, __riscv_vsll_vx_u32m1(index2, 2, 8), 8);
	vuint32m1_t s3 = __riscv_vluxei32_v_u32m1(lut3, __riscv_vsll_vx_u32m1(index3, 2, 8), 8);

	s0 = __riscv_vxor_vv_u32m1(s0, s1, 8);
	s2 = __riscv_vxor_vv_u32m1(s2, s3, 8);
	s0 = __riscv_vxor_vv_u32m1(s0, s2, 8);

	return __riscv_vxor_vv_u32m1(s0, key, 8);
}

static constexpr uint32_t AES_HASH_1R_STATE02[8] = { 0x92b52c0d, 0x9fa856de, 0xcc82db47, 0xd7983aad, 0x6a770017, 0xae62c7d0, 0x5079506b, 0xe8a07ce4 };
static constexpr uint32_t AES_HASH_1R_STATE13[8] = { 0x338d996e, 0x15c7b798, 0xf59e125a, 0xace78057, 0x630a240c, 0x07ad828d, 0x79a10005, 0x7e994948 };

static constexpr uint32_t AES_GEN_1R_KEY02[8] = { 0x6daca553, 0x62716609, 0xdbb5552b, 0xb4f44917, 0x3f1262f1, 0x9f947ec6, 0xf4c0794f, 0x3e20e345 };
static constexpr uint32_t AES_GEN_1R_KEY13[8] = { 0x6d7caf07, 0x846a710d, 0x1725d378, 0x0da1dc4e, 0x6aef8135, 0xb1ba317c, 0x16314c88, 0x49169154 };

static constexpr uint32_t AES_HASH_1R_XKEY00[8] = { 0xf6fa8389, 0x8b24949f, 0x90dc56bf, 0x06890201, 0xf6fa8389, 0x8b24949f, 0x90dc56bf, 0x06890201 };
static constexpr uint32_t AES_HASH_1R_XKEY11[8] = { 0x61b263d1, 0x51f4e03c, 0xee1043c6, 0xed18f99b, 0x61b263d1, 0x51f4e03c, 0xee1043c6, 0xed18f99b };

static constexpr uint32_t AES_HASH_STRIDE_X2[8] = { 0, 4, 8, 12, 32, 36, 40, 44 };
static constexpr uint32_t AES_HASH_STRIDE_X4[8] = { 0, 4, 8, 12, 64, 68, 72, 76 };

#define lutEnc0 lutEnc[0]
#define lutEnc1 lutEnc[1]
#define lutEnc2 lutEnc[2]
#define lutEnc3 lutEnc[3]

#define lutDec0 lutDec[0]
#define lutDec1 lutDec[1]
#define lutDec2 lutDec[2]
#define lutDec3 lutDec[3]

void hashAes1Rx4_RVV(const void *input, size_t inputSize, void *hash) {
	const uint8_t* inptr = (const uint8_t*)input;
	const uint8_t* inputEnd = inptr + inputSize;

	//intial state
	vuint32m1_t state02 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE02, 8);
	vuint32m1_t state13 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE13, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);

	const vuint8m1_t lutenc_index0 = __riscv_vle8_v_u8m1(lutEncIndex[0], 32);
	const vuint8m1_t lutenc_index1 = __riscv_vle8_v_u8m1(lutEncIndex[1], 32);
	const vuint8m1_t lutenc_index2 = __riscv_vle8_v_u8m1(lutEncIndex[2], 32);
	const vuint8m1_t lutenc_index3 = __riscv_vle8_v_u8m1(lutEncIndex[3], 32);

	const vuint8m1_t& lutdec_index0 = lutenc_index0;
	const vuint8m1_t lutdec_index1 = __riscv_vle8_v_u8m1(lutDecIndex[1], 32);
	const vuint8m1_t& lutdec_index2 = lutenc_index2;
	const vuint8m1_t lutdec_index3 = __riscv_vle8_v_u8m1(lutDecIndex[3], 32);

	//process 64 bytes at a time in 4 lanes
	while (inptr < inputEnd) {
		state02 = softaes_vector_double(state02, __riscv_vluxei32_v_u32m1((uint32_t*)inptr + 0, stride, 8), lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);
		state13 = softaes_vector_double(state13, __riscv_vluxei32_v_u32m1((uint32_t*)inptr + 4, stride, 8), lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);

		inptr += 64;
	}

	//two extra rounds to achieve full diffusion
	const vuint32m1_t xkey00 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY00, 8);
	const vuint32m1_t xkey11 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY11, 8);

	state02 = softaes_vector_double(state02, xkey00, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);
	state13 = softaes_vector_double(state13, xkey00, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);

	state02 = softaes_vector_double(state02, xkey11, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);
	state13 = softaes_vector_double(state13, xkey11, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);

	//output hash
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 0, stride, state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 4, stride, state13, 8);
}

void fillAes1Rx4_RVV(void *state, size_t outputSize, void *buffer) {
	const uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	const vuint32m1_t key02 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY02, 8);
	const vuint32m1_t key13 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY13, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);

	vuint32m1_t state02 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 0, stride, 8);
	vuint32m1_t state13 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 4, stride, 8);

	const vuint8m1_t lutenc_index0 = __riscv_vle8_v_u8m1(lutEncIndex[0], 32);
	const vuint8m1_t lutenc_index1 = __riscv_vle8_v_u8m1(lutEncIndex[1], 32);
	const vuint8m1_t lutenc_index2 = __riscv_vle8_v_u8m1(lutEncIndex[2], 32);
	const vuint8m1_t lutenc_index3 = __riscv_vle8_v_u8m1(lutEncIndex[3], 32);

	const vuint8m1_t& lutdec_index0 = lutenc_index0;
	const vuint8m1_t lutdec_index1 = __riscv_vle8_v_u8m1(lutDecIndex[1], 32);
	const vuint8m1_t& lutdec_index2 = lutenc_index2;
	const vuint8m1_t lutdec_index3 = __riscv_vle8_v_u8m1(lutDecIndex[3], 32);

	while (outptr < outputEnd) {
		state02 = softaes_vector_double(state02, key02, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);
		state13 = softaes_vector_double(state13, key13, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);

		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 0, stride, state02, 8);
		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 4, stride, state13, 8);

		outptr += 64;
	}

	__riscv_vsuxei32_v_u32m1((uint32_t*)state + 0, stride, state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)state + 4, stride, state13, 8);
}

void fillAes4Rx4_RVV(void *state, size_t outputSize, void *buffer) {
	const uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	const vuint32m1_t stride4 = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X4, 8);

	const vuint32m1_t key04 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 0), stride4, 8);
	const vuint32m1_t key15 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 1), stride4, 8);
	const vuint32m1_t key26 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 2), stride4, 8);
	const vuint32m1_t key37 = __riscv_vluxei32_v_u32m1((uint32_t*)(RandomX_CurrentConfig.fillAes4Rx4_Key + 3), stride4, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);

	vuint32m1_t state02 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 0, stride, 8);
	vuint32m1_t state13 = __riscv_vluxei32_v_u32m1((uint32_t*)state + 4, stride, 8);

	const vuint8m1_t lutenc_index0 = __riscv_vle8_v_u8m1(lutEncIndex[0], 32);
	const vuint8m1_t lutenc_index1 = __riscv_vle8_v_u8m1(lutEncIndex[1], 32);
	const vuint8m1_t lutenc_index2 = __riscv_vle8_v_u8m1(lutEncIndex[2], 32);
	const vuint8m1_t lutenc_index3 = __riscv_vle8_v_u8m1(lutEncIndex[3], 32);

	const vuint8m1_t& lutdec_index0 = lutenc_index0;
	const vuint8m1_t lutdec_index1 = __riscv_vle8_v_u8m1(lutDecIndex[1], 32);
	const vuint8m1_t& lutdec_index2 = lutenc_index2;
	const vuint8m1_t lutdec_index3 = __riscv_vle8_v_u8m1(lutDecIndex[3], 32);

	while (outptr < outputEnd) {
		state02 = softaes_vector_double(state02, key04, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);
		state13 = softaes_vector_double(state13, key04, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);

		state02 = softaes_vector_double(state02, key15, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);
		state13 = softaes_vector_double(state13, key15, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);

		state02 = softaes_vector_double(state02, key26, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);
		state13 = softaes_vector_double(state13, key26, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);

		state02 = softaes_vector_double(state02, key37, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);
		state13 = softaes_vector_double(state13, key37, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);

		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 0, stride, state02, 8);
		__riscv_vsuxei32_v_u32m1((uint32_t*)outptr + 4, stride, state13, 8);

		outptr += 64;
	}
}

void hashAndFillAes1Rx4_RVV(void *scratchpad, size_t scratchpadSize, void *hash, void* fill_state) {
	uint8_t* scratchpadPtr = (uint8_t*)scratchpad;
	const uint8_t* scratchpadEnd = scratchpadPtr + scratchpadSize;

	vuint32m1_t hash_state02 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE02, 8);
	vuint32m1_t hash_state13 = __riscv_vle32_v_u32m1(AES_HASH_1R_STATE13, 8);

	const vuint32m1_t key02 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY02, 8);
	const vuint32m1_t key13 = __riscv_vle32_v_u32m1(AES_GEN_1R_KEY13, 8);

	const vuint32m1_t stride = __riscv_vle32_v_u32m1(AES_HASH_STRIDE_X2, 8);

	vuint32m1_t fill_state02 = __riscv_vluxei32_v_u32m1((uint32_t*)fill_state + 0, stride, 8);
	vuint32m1_t fill_state13 = __riscv_vluxei32_v_u32m1((uint32_t*)fill_state + 4, stride, 8);

	const vuint8m1_t lutenc_index0 = __riscv_vle8_v_u8m1(lutEncIndex[0], 32);
	const vuint8m1_t lutenc_index1 = __riscv_vle8_v_u8m1(lutEncIndex[1], 32);
	const vuint8m1_t lutenc_index2 = __riscv_vle8_v_u8m1(lutEncIndex[2], 32);
	const vuint8m1_t lutenc_index3 = __riscv_vle8_v_u8m1(lutEncIndex[3], 32);

	const vuint8m1_t& lutdec_index0 = lutenc_index0;
	const vuint8m1_t lutdec_index1 = __riscv_vle8_v_u8m1(lutDecIndex[1], 32);
	const vuint8m1_t& lutdec_index2 = lutenc_index2;
	const vuint8m1_t lutdec_index3 = __riscv_vle8_v_u8m1(lutDecIndex[3], 32);

	//process 64 bytes at a time in 4 lanes
	while (scratchpadPtr < scratchpadEnd) {
#define HASH_STATE(k) \
		hash_state02 = softaes_vector_double(hash_state02, __riscv_vluxei32_v_u32m1((uint32_t*)scratchpadPtr + k * 16 + 0, stride, 8), lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3); \
		hash_state13 = softaes_vector_double(hash_state13, __riscv_vluxei32_v_u32m1((uint32_t*)scratchpadPtr + k * 16 + 4, stride, 8), lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);

#define FILL_STATE(k) \
		fill_state02 = softaes_vector_double(fill_state02, key02, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3); \
		fill_state13 = softaes_vector_double(fill_state13, key13, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3); \
		__riscv_vsuxei32_v_u32m1((uint32_t*)scratchpadPtr + k * 16 + 0, stride, fill_state02, 8); \
		__riscv_vsuxei32_v_u32m1((uint32_t*)scratchpadPtr + k * 16 + 4, stride, fill_state13, 8);

		HASH_STATE(0);
		HASH_STATE(1);

		FILL_STATE(0);
		FILL_STATE(1);

		scratchpadPtr += 128;
	}

#undef HASH_STATE
#undef FILL_STATE

	__riscv_vsuxei32_v_u32m1((uint32_t*)fill_state + 0, stride, fill_state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)fill_state + 4, stride, fill_state13, 8);

	//two extra rounds to achieve full diffusion
	const vuint32m1_t xkey00 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY00, 8);
	const vuint32m1_t xkey11 = __riscv_vle32_v_u32m1(AES_HASH_1R_XKEY11, 8);

	hash_state02 = softaes_vector_double(hash_state02, xkey00, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);
	hash_state13 = softaes_vector_double(hash_state13, xkey00, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);

	hash_state02 = softaes_vector_double(hash_state02, xkey11, lutenc_index0, lutenc_index1, lutenc_index2, lutenc_index3, lutEnc0, lutEnc1, lutEnc2, lutEnc3);
	hash_state13 = softaes_vector_double(hash_state13, xkey11, lutdec_index0, lutdec_index1, lutdec_index2, lutdec_index3, lutDec0, lutDec1, lutDec2, lutDec3);

	//output hash
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 0, stride, hash_state02, 8);
	__riscv_vsuxei32_v_u32m1((uint32_t*)hash + 4, stride, hash_state13, 8);
}
