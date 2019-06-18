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

#pragma once

#include <cstdint>
#include "blake2/endian.h"

constexpr int32_t unsigned32ToSigned2sCompl(uint32_t x) {
	return (-1 == ~0) ? (int32_t)x : (x > INT32_MAX ? (-(int32_t)(UINT32_MAX - x) - 1) : (int32_t)x);
}

constexpr int64_t unsigned64ToSigned2sCompl(uint64_t x) {
	return (-1 == ~0) ? (int64_t)x : (x > INT64_MAX ? (-(int64_t)(UINT64_MAX - x) - 1) : (int64_t)x);
}

constexpr uint64_t signExtend2sCompl(uint32_t x) {
	return (-1 == ~0) ? (int64_t)(int32_t)(x) : (x > INT32_MAX ? (x | 0xffffffff00000000ULL) : (uint64_t)x);
}

constexpr int RoundToNearest = 0;
constexpr int RoundDown = 1;
constexpr int RoundUp = 2;
constexpr int RoundToZero = 3;

//MSVC doesn't define __SSE2__, so we have to define it manually if SSE2 is available
#if !defined(__SSE2__) && (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP == 2))
#define __SSE2__ 1
#endif

//the library "sqrt" function provided by MSVC for x86 targets doesn't give
//the correct results, so we have to use inline assembly to call x87 fsqrt directly
#if !defined(__SSE2__)
#if defined(_M_IX86)
inline double __cdecl rx_sqrt(double x) {
	__asm {
		fld x
		fsqrt
	}
}
#define rx_sqrt rx_sqrt

void rx_set_double_precision();
#define RANDOMX_USE_X87

#elif defined(__i386)

void rx_set_double_precision();
#define RANDOMX_USE_X87

#endif
#endif //__SSE2__

#if !defined(rx_sqrt)
#define rx_sqrt sqrt
#endif

#if !defined(RANDOMX_USE_X87)
#define rx_set_double_precision(x)
#endif

#ifdef __SSE2__
#ifdef __GNUC__
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

typedef __m128i rx_vec_i128;
typedef __m128d rx_vec_f128;

#define rx_aligned_alloc(a, b) _mm_malloc(a,b)
#define rx_aligned_free(a) _mm_free(a)
#define rx_prefetch_nta(x) _mm_prefetch((const char *)(x), _MM_HINT_NTA)

#define rx_load_vec_f128 _mm_load_pd
#define rx_store_vec_f128 _mm_store_pd
#define rx_add_vec_f128 _mm_add_pd
#define rx_sub_vec_f128 _mm_sub_pd
#define rx_mul_vec_f128 _mm_mul_pd
#define rx_div_vec_f128 _mm_div_pd
#define rx_sqrt_vec_f128 _mm_sqrt_pd

FORCE_INLINE rx_vec_f128 rx_swap_vec_f128(rx_vec_f128 a) {
	return _mm_shuffle_pd(a, a, 1);
}

FORCE_INLINE rx_vec_f128 rx_set_vec_f128(uint64_t x1, uint64_t x0) {
	return _mm_castsi128_pd(_mm_set_epi64x(x1, x0));
}

FORCE_INLINE rx_vec_f128 rx_set1_vec_f128(uint64_t x) {
	return _mm_castsi128_pd(_mm_set1_epi64x(x));
}

#define rx_xor_vec_f128 _mm_xor_pd
#define rx_and_vec_f128 _mm_and_pd
#define rx_or_vec_f128 _mm_or_pd
#define rx_aesenc_vec_i128 _mm_aesenc_si128
#define rx_aesdec_vec_i128 _mm_aesdec_si128

FORCE_INLINE int rx_vec_i128_x(rx_vec_i128 a) {
	return _mm_cvtsi128_si32(a);
}

FORCE_INLINE int rx_vec_i128_y(rx_vec_i128 a) {
	return _mm_cvtsi128_si32(_mm_shuffle_epi32(a, 0x55));
}

FORCE_INLINE int rx_vec_i128_z(rx_vec_i128 a) {
	return _mm_cvtsi128_si32(_mm_shuffle_epi32(a, 0xaa));
}

FORCE_INLINE int rx_vec_i128_w(rx_vec_i128 a) {
	return _mm_cvtsi128_si32(_mm_shuffle_epi32(a, 0xff));
}

#define rx_set_int_vec_i128 _mm_set_epi32
#define rx_xor_vec_i128 _mm_xor_si128
#define rx_load_vec_i128 _mm_load_si128
#define rx_store_vec_i128 _mm_store_si128

FORCE_INLINE rx_vec_f128 rx_cvt_packed_int_vec_f128(const void* addr) {
	__m128i ix = _mm_loadl_epi64((const __m128i*)addr);
	return _mm_cvtepi32_pd(ix);
}

constexpr uint32_t rx_mxcsr_default = 0x9FC0; //Flush to zero, denormals are zero, default rounding mode, all exceptions disabled

FORCE_INLINE void rx_reset_float_state() {
	_mm_setcsr(rx_mxcsr_default);
}

FORCE_INLINE void rx_set_rounding_mode(uint32_t mode) {
	_mm_setcsr(rx_mxcsr_default | (mode << 13));
}

#else
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
#include <cmath>

typedef union {
	uint64_t u64[2];
	uint32_t u32[4];
	uint16_t u16[8];
	uint8_t u8[16];
} rx_vec_i128;

typedef union {
	struct {
		double lo;
		double hi;
	};
	rx_vec_i128 i;
} rx_vec_f128;

#define rx_aligned_alloc(a, b) malloc(a)
#define rx_aligned_free(a) free(a)
#define rx_prefetch_nta(x)

FORCE_INLINE rx_vec_f128 rx_load_vec_f128(const double* pd) {
	rx_vec_f128 x;
	x.i.u64[0] = load64(pd + 0);
	x.i.u64[1] = load64(pd + 1);
	return x;
}

FORCE_INLINE void rx_store_vec_f128(double* mem_addr, rx_vec_f128 a) {
	store64(mem_addr + 0, a.i.u64[0]);
	store64(mem_addr + 1, a.i.u64[1]);
}

FORCE_INLINE rx_vec_f128 rx_swap_vec_f128(rx_vec_f128 a) {
	double temp = a.hi;
	a.hi = a.lo;
	a.lo = temp;
	return a;
}

FORCE_INLINE rx_vec_f128 rx_add_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.lo = a.lo + b.lo;
	x.hi = a.hi + b.hi;
	return x;
}

FORCE_INLINE rx_vec_f128 rx_sub_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.lo = a.lo - b.lo;
	x.hi = a.hi - b.hi;
	return x;
}

FORCE_INLINE rx_vec_f128 rx_mul_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.lo = a.lo * b.lo;
	x.hi = a.hi * b.hi;
	return x;
}

FORCE_INLINE rx_vec_f128 rx_div_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.lo = a.lo / b.lo;
	x.hi = a.hi / b.hi;
	return x;
}

FORCE_INLINE rx_vec_f128 rx_sqrt_vec_f128(rx_vec_f128 a) {
	rx_vec_f128 x;
	x.lo = rx_sqrt(a.lo);
	x.hi = rx_sqrt(a.hi);
	return x;
}

FORCE_INLINE rx_vec_i128 rx_set1_long_vec_i128(uint64_t a) {
	rx_vec_i128 x;
	x.u64[0] = a;
	x.u64[1] = a;
	return x;
}

FORCE_INLINE rx_vec_f128 rx_vec_i128_vec_f128(rx_vec_i128 a) {
	rx_vec_f128 x;
	x.i = a;
	return x;
}

FORCE_INLINE rx_vec_f128 rx_set_vec_f128(uint64_t x1, uint64_t x0) {
	rx_vec_f128 v;
	v.i.u64[0] = x0;
	v.i.u64[1] = x1;
	return v;
}

FORCE_INLINE rx_vec_f128 rx_set1_vec_f128(uint64_t x) {
	rx_vec_f128 v;
	v.i.u64[0] = x;
	v.i.u64[1] = x;
	return v;
}


FORCE_INLINE rx_vec_f128 rx_xor_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.i.u64[0] = a.i.u64[0] ^ b.i.u64[0];
	x.i.u64[1] = a.i.u64[1] ^ b.i.u64[1];
	return x;
}

FORCE_INLINE rx_vec_f128 rx_and_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.i.u64[0] = a.i.u64[0] & b.i.u64[0];
	x.i.u64[1] = a.i.u64[1] & b.i.u64[1];
	return x;
}

FORCE_INLINE rx_vec_f128 rx_or_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.i.u64[0] = a.i.u64[0] | b.i.u64[0];
	x.i.u64[1] = a.i.u64[1] | b.i.u64[1];
	return x;
}

static const char* platformError = "Platform doesn't support hardware AES";

FORCE_INLINE rx_vec_i128 rx_aesenc_vec_i128(rx_vec_i128 v, rx_vec_i128 rkey) {
	throw std::runtime_error(platformError);
}

FORCE_INLINE rx_vec_i128 rx_aesdec_vec_i128(rx_vec_i128 v, rx_vec_i128 rkey) {
	throw std::runtime_error(platformError);
}

FORCE_INLINE int rx_vec_i128_x(rx_vec_i128 a) {
	return a.u32[0];
}

FORCE_INLINE int rx_vec_i128_y(rx_vec_i128 a) {
	return a.u32[1];
}

FORCE_INLINE int rx_vec_i128_z(rx_vec_i128 a) {
	return a.u32[2];
}

FORCE_INLINE int rx_vec_i128_w(rx_vec_i128 a) {
	return a.u32[3];
}

FORCE_INLINE rx_vec_i128 rx_set_int_vec_i128(int _I3, int _I2, int _I1, int _I0) {
	rx_vec_i128 v;
	v.u32[0] = _I0;
	v.u32[1] = _I1;
	v.u32[2] = _I2;
	v.u32[3] = _I3;
	return v;
};

FORCE_INLINE rx_vec_i128 rx_xor_vec_i128(rx_vec_i128 _A, rx_vec_i128 _B) {
	rx_vec_i128 c;
	c.u32[0] = _A.u32[0] ^ _B.u32[0];
	c.u32[1] = _A.u32[1] ^ _B.u32[1];
	c.u32[2] = _A.u32[2] ^ _B.u32[2];
	c.u32[3] = _A.u32[3] ^ _B.u32[3];
	return c;
}

FORCE_INLINE rx_vec_i128 rx_load_vec_i128(rx_vec_i128 const*_P) {
#if defined(NATIVE_LITTLE_ENDIAN)
	return *_P;
#else
	uint32_t* ptr = (uint32_t*)_P;
	rx_vec_i128 c;
	c.u32[0] = load32(ptr + 0);
	c.u32[1] = load32(ptr + 1);
	c.u32[2] = load32(ptr + 2);
	c.u32[3] = load32(ptr + 3);
	return c;
#endif
}

FORCE_INLINE void rx_store_vec_i128(rx_vec_i128 *_P, rx_vec_i128 _B) {
#if defined(NATIVE_LITTLE_ENDIAN)
	*_P = _B;
#else
	uint32_t* ptr = (uint32_t*)_P;
	store32(ptr + 0, _B.u32[0]);
	store32(ptr + 1, _B.u32[1]);
	store32(ptr + 2, _B.u32[2]);
	store32(ptr + 3, _B.u32[3]);
#endif
}

FORCE_INLINE rx_vec_f128 rx_cvt_packed_int_vec_f128(const void* addr) {
	rx_vec_f128 x;
	x.lo = (double)unsigned32ToSigned2sCompl(load32((uint8_t*)addr + 0));
	x.hi = (double)unsigned32ToSigned2sCompl(load32((uint8_t*)addr + 4));
	return x;
}

#define RANDOMX_DEFAULT_FENV

void rx_reset_float_state();

void rx_set_rounding_mode(uint32_t mode);

#endif

double loadDoublePortable(const void* addr);
uint64_t mulh(uint64_t, uint64_t);
int64_t smulh(int64_t, int64_t);
uint64_t rotl(uint64_t, int);
uint64_t rotr(uint64_t, int);
