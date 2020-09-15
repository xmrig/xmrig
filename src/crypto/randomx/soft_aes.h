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
	uint32_t s0, s1, s2, s3;

	s0 = rx_vec_i128_w(in);
	s1 = rx_vec_i128_z(in);
	s2 = rx_vec_i128_y(in);
	s3 = rx_vec_i128_x(in);

	rx_vec_i128 out = rx_set_int_vec_i128(
		(lutEnc0[s0 & 0xff] ^ lutEnc1[(s3 >> 8) & 0xff] ^ lutEnc2[(s2 >> 16) & 0xff] ^ lutEnc3[s1 >> 24]),
		(lutEnc0[s1 & 0xff] ^ lutEnc1[(s0 >> 8) & 0xff] ^ lutEnc2[(s3 >> 16) & 0xff] ^ lutEnc3[s2 >> 24]),
		(lutEnc0[s2 & 0xff] ^ lutEnc1[(s1 >> 8) & 0xff] ^ lutEnc2[(s0 >> 16) & 0xff] ^ lutEnc3[s3 >> 24]),
		(lutEnc0[s3 & 0xff] ^ lutEnc1[(s2 >> 8) & 0xff] ^ lutEnc2[(s1 >> 16) & 0xff] ^ lutEnc3[s0 >> 24])
	);

	return rx_xor_vec_i128(out, key);
}

template<>
FORCE_INLINE rx_vec_i128 aesdec<true>(rx_vec_i128 in, rx_vec_i128 key) {
	uint32_t s0, s1, s2, s3;

	s0 = rx_vec_i128_w(in);
	s1 = rx_vec_i128_z(in);
	s2 = rx_vec_i128_y(in);
	s3 = rx_vec_i128_x(in);

	rx_vec_i128 out = rx_set_int_vec_i128(
		(lutDec0[s0 & 0xff] ^ lutDec1[(s1 >> 8) & 0xff] ^ lutDec2[(s2 >> 16) & 0xff] ^ lutDec3[s3 >> 24]),
		(lutDec0[s1 & 0xff] ^ lutDec1[(s2 >> 8) & 0xff] ^ lutDec2[(s3 >> 16) & 0xff] ^ lutDec3[s0 >> 24]),
		(lutDec0[s2 & 0xff] ^ lutDec1[(s3 >> 8) & 0xff] ^ lutDec2[(s0 >> 16) & 0xff] ^ lutDec3[s1 >> 24]),
		(lutDec0[s3 & 0xff] ^ lutDec1[(s0 >> 8) & 0xff] ^ lutDec2[(s1 >> 16) & 0xff] ^ lutDec3[s2 >> 24])
	);

	return rx_xor_vec_i128(out, key);
}

template<>
FORCE_INLINE rx_vec_i128 aesenc<false>(rx_vec_i128 in, rx_vec_i128 key) {
	return rx_aesenc_vec_i128(in, key);
}

template<>
FORCE_INLINE rx_vec_i128 aesdec<false>(rx_vec_i128 in, rx_vec_i128 key) {
	return rx_aesdec_vec_i128(in, key);
}
