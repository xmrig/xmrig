/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>
Copyright (c) 2019 SChernykh   <https://github.com/SChernykh>

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

#pragma once

#include <stdint.h>
#include "crypto/randomx/intrin_portable.h"

extern uint32_t lutEnc0[256];
extern uint32_t lutEnc1[256];
extern uint32_t lutEnc2[256];
extern uint32_t lutEnc3[256];
extern uint32_t lutDec0[256];
extern uint32_t lutDec1[256];
extern uint32_t lutDec2[256];
extern uint32_t lutDec3[256];

template<bool soft> rx_vec_i128 aesenc(rx_vec_i128 in, rx_vec_i128 key);
template<bool soft> rx_vec_i128 aesdec(rx_vec_i128 in, rx_vec_i128 key);

template<>
FORCE_INLINE rx_vec_i128 aesenc<true>(rx_vec_i128 in, rx_vec_i128 key) {
	volatile uint8_t s[16];
	memcpy((void*) s, &in, 16);

	uint32_t s0 = lutEnc0[s[ 0]];
	uint32_t s1 = lutEnc0[s[ 4]];
	uint32_t s2 = lutEnc0[s[ 8]];
	uint32_t s3 = lutEnc0[s[12]];

	s0 ^= lutEnc1[s[ 5]];
	s1 ^= lutEnc1[s[ 9]];
	s2 ^= lutEnc1[s[13]];
	s3 ^= lutEnc1[s[ 1]];

	s0 ^= lutEnc2[s[10]];
	s1 ^= lutEnc2[s[14]];
	s2 ^= lutEnc2[s[ 2]];
	s3 ^= lutEnc2[s[ 6]];

	s0 ^= lutEnc3[s[15]];
	s1 ^= lutEnc3[s[ 3]];
	s2 ^= lutEnc3[s[ 7]];
	s3 ^= lutEnc3[s[11]];

	return rx_xor_vec_i128(rx_set_int_vec_i128(s3, s2, s1, s0), key);
}

template<>
FORCE_INLINE rx_vec_i128 aesdec<true>(rx_vec_i128 in, rx_vec_i128 key) {
	volatile uint8_t s[16];
	memcpy((void*) s, &in, 16);

	uint32_t s0 = lutDec0[s[ 0]];
	uint32_t s1 = lutDec0[s[ 4]];
	uint32_t s2 = lutDec0[s[ 8]];
	uint32_t s3 = lutDec0[s[12]];

	s0 ^= lutDec1[s[13]];
	s1 ^= lutDec1[s[ 1]];
	s2 ^= lutDec1[s[ 5]];
	s3 ^= lutDec1[s[ 9]];

	s0 ^= lutDec2[s[10]];
	s1 ^= lutDec2[s[14]];
	s2 ^= lutDec2[s[ 2]];
	s3 ^= lutDec2[s[ 6]];

	s0 ^= lutDec3[s[ 7]];
	s1 ^= lutDec3[s[11]];
	s2 ^= lutDec3[s[15]];
	s3 ^= lutDec3[s[ 3]];

	return rx_xor_vec_i128(rx_set_int_vec_i128(s3, s2, s1, s0), key);
}

template<>
FORCE_INLINE rx_vec_i128 aesenc<false>(rx_vec_i128 in, rx_vec_i128 key) {
	return rx_aesenc_vec_i128(in, key);
}

template<>
FORCE_INLINE rx_vec_i128 aesdec<false>(rx_vec_i128 in, rx_vec_i128 key) {
	return rx_aesdec_vec_i128(in, key);
}
