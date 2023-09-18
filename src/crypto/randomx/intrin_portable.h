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
#include "crypto/randomx/blake2/endian.h"

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

//MSVC doesn't define __AES__
#if defined(_MSC_VER) && defined(__SSE2__)
#define __AES__
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
#define rx_prefetch_t0(x) _mm_prefetch((const char *)(x), _MM_HINT_T0)

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
#define rx_and_vec_i128 _mm_and_si128
#define rx_or_vec_f128 _mm_or_pd

#ifdef __AES__

#define rx_aesenc_vec_i128 _mm_aesenc_si128
#define rx_aesdec_vec_i128 _mm_aesdec_si128

#define HAVE_AES

#endif //__AES__

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

#elif defined(__PPC64__) && defined(__ALTIVEC__) && defined(__VSX__) //sadly only POWER7 and newer will be able to use SIMD acceleration. Earlier processors cant use doubles or 64 bit integers with SIMD
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
#include <altivec.h>
#undef vector
#undef pixel
#undef bool

typedef __vector uint8_t __m128i;
typedef __vector uint32_t __m128l;
typedef __vector int      __m128li;
typedef __vector uint64_t __m128ll;
typedef __vector double __m128d;

typedef __m128i rx_vec_i128;
typedef __m128d rx_vec_f128;
typedef union{
	rx_vec_i128 i;
  rx_vec_f128 d;
  uint64_t u64[2];
  double   d64[2];
  uint32_t u32[4];
	int i32[4];
} vec_u;

#define rx_aligned_alloc(a, b) malloc(a)
#define rx_aligned_free(a) free(a)
#define rx_prefetch_nta(x)
#define rx_prefetch_t0(x)

/* Splat 64-bit long long to 2 64-bit long longs */
FORCE_INLINE __m128i vec_splat2sd (int64_t scalar)
{ return (__m128i) vec_splats (scalar); }

FORCE_INLINE rx_vec_f128 rx_load_vec_f128(const double* pd) {
#if defined(NATIVE_LITTLE_ENDIAN)
	return (rx_vec_f128)vec_vsx_ld(0,pd);
#else
	vec_u t;
	t.u64[0] = load64(pd + 0);
	t.u64[1] = load64(pd + 1);
	return (rx_vec_f128)t.d;
#endif
}

FORCE_INLINE void rx_store_vec_f128(double* mem_addr, rx_vec_f128 a) {
#if defined(NATIVE_LITTLE_ENDIAN)
	vec_vsx_st(a,0,(rx_vec_f128*)mem_addr);
#else
	vec_u _a;
	_a.d = a;
	store64(mem_addr + 0, _a.u64[0]);
	store64(mem_addr + 1, _a.u64[1]);
#endif
}

FORCE_INLINE rx_vec_f128 rx_swap_vec_f128(rx_vec_f128 a) {
	return (rx_vec_f128)vec_perm((__m128i)a,(__m128i)a,(__m128i){8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7});
}

FORCE_INLINE rx_vec_f128 rx_add_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return (rx_vec_f128)vec_add(a,b);
}

FORCE_INLINE rx_vec_f128 rx_sub_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return (rx_vec_f128)vec_sub(a,b);
}

FORCE_INLINE rx_vec_f128 rx_mul_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return (rx_vec_f128)vec_mul(a,b);
}

FORCE_INLINE rx_vec_f128 rx_div_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return (rx_vec_f128)vec_div(a,b);
}

FORCE_INLINE rx_vec_f128 rx_sqrt_vec_f128(rx_vec_f128 a) {
	return (rx_vec_f128)vec_sqrt(a);
}

FORCE_INLINE rx_vec_i128 rx_set1_long_vec_i128(uint64_t a) {
	return (rx_vec_i128)vec_splat2sd(a);
}

FORCE_INLINE rx_vec_f128 rx_vec_i128_vec_f128(rx_vec_i128 a) {
	return (rx_vec_f128)a;
}

FORCE_INLINE rx_vec_f128 rx_set_vec_f128(uint64_t x1, uint64_t x0) {
	return (rx_vec_f128)(__m128ll){x0,x1};
}

FORCE_INLINE rx_vec_f128 rx_set1_vec_f128(uint64_t x) {
	return (rx_vec_f128)vec_splat2sd(x);
}

FORCE_INLINE rx_vec_f128 rx_xor_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return (rx_vec_f128)vec_xor(a,b);
}

FORCE_INLINE rx_vec_f128 rx_and_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return (rx_vec_f128)vec_and(a,b);
}

FORCE_INLINE rx_vec_i128 rx_and_vec_i128(rx_vec_i128 a, rx_vec_i128 b) {
	return (rx_vec_i128)vec_and(a, b);
}

FORCE_INLINE rx_vec_f128 rx_or_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return (rx_vec_f128)vec_or(a,b);
}

#if defined(__CRYPTO__)

FORCE_INLINE __m128ll vrev(__m128i v){
#if defined(NATIVE_LITTLE_ENDIAN)
	return (__m128ll)vec_perm((__m128i)v,(__m128i){0},(__m128i){15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0});
#else
	return (__m128ll)vec_perm((__m128i)v,(__m128i){0},(__m128i){3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12});
#endif
}

FORCE_INLINE rx_vec_i128 rx_aesenc_vec_i128(rx_vec_i128 v, rx_vec_i128 rkey) {
	__m128ll _v = vrev(v);
	__m128ll _rkey = vrev(rkey);
	__m128ll result = vrev((__m128i)__builtin_crypto_vcipher(_v,_rkey));
	return (rx_vec_i128)result;
}

FORCE_INLINE rx_vec_i128 rx_aesdec_vec_i128(rx_vec_i128 v, rx_vec_i128 rkey) {
	__m128ll _v = vrev(v);
	__m128ll zero = (__m128ll){0};
	__m128ll out = vrev((__m128i)__builtin_crypto_vncipher(_v,zero));
	return (rx_vec_i128)vec_xor((__m128i)out,rkey);
}
#define HAVE_AES

#endif //__CRYPTO__

FORCE_INLINE int rx_vec_i128_x(rx_vec_i128 a) {
	vec_u _a;
	_a.i = a;
  return _a.i32[0];
}

FORCE_INLINE int rx_vec_i128_y(rx_vec_i128 a) {
	vec_u _a;
	_a.i = a;
	return _a.i32[1];
}

FORCE_INLINE int rx_vec_i128_z(rx_vec_i128 a) {
	vec_u _a;
	_a.i = a;
	return _a.i32[2];
}

FORCE_INLINE int rx_vec_i128_w(rx_vec_i128 a) {
	vec_u _a;
	_a.i = a;
	return _a.i32[3];
}

FORCE_INLINE rx_vec_i128 rx_set_int_vec_i128(int _I3, int _I2, int _I1, int _I0) {
	return (rx_vec_i128)((__m128li){_I0,_I1,_I2,_I3});
};

FORCE_INLINE rx_vec_i128 rx_xor_vec_i128(rx_vec_i128 _A, rx_vec_i128 _B) {
	return (rx_vec_i128)vec_xor(_A,_B);
}

FORCE_INLINE rx_vec_i128 rx_load_vec_i128(rx_vec_i128 const *_P) {
#if defined(NATIVE_LITTLE_ENDIAN)
	return *_P;
#else
	uint32_t* ptr = (uint32_t*)_P;
	vec_u c;
	c.u32[0] = load32(ptr + 0);
	c.u32[1] = load32(ptr + 1);
	c.u32[2] = load32(ptr + 2);
	c.u32[3] = load32(ptr + 3);
	return (rx_vec_i128)c.i;
#endif
}

FORCE_INLINE void rx_store_vec_i128(rx_vec_i128 *_P, rx_vec_i128 _B) {
#if defined(NATIVE_LITTLE_ENDIAN)
	*_P = _B;
#else
	uint32_t* ptr = (uint32_t*)_P;
	vec_u B;
	B.i = _B;
	store32(ptr + 0, B.u32[0]);
	store32(ptr + 1, B.u32[1]);
	store32(ptr + 2, B.u32[2]);
	store32(ptr + 3, B.u32[3]);
#endif
}

FORCE_INLINE rx_vec_f128 rx_cvt_packed_int_vec_f128(const void* addr) {
	vec_u x;
	x.d64[0] = (double)unsigned32ToSigned2sCompl(load32((uint8_t*)addr + 0));
	x.d64[1] = (double)unsigned32ToSigned2sCompl(load32((uint8_t*)addr + 4));
	return (rx_vec_f128)x.d;
}

#define RANDOMX_DEFAULT_FENV

#elif defined(__aarch64__)

#include <stdlib.h>
#include <arm_neon.h>
#include <arm_acle.h>

typedef uint8x16_t rx_vec_i128;
typedef float64x2_t rx_vec_f128;

inline void* rx_aligned_alloc(size_t size, size_t align) {
	void* p;
	if (posix_memalign(&p, align, size) == 0)
		return p;

	return 0;
};

#define rx_aligned_free(a) free(a)

inline void rx_prefetch_nta(void* ptr) {
	asm volatile ("prfm pldl1strm, [%0]\n" : : "r" (ptr));
}

inline void rx_prefetch_t0(const void* ptr) {
	asm volatile ("prfm pldl1strm, [%0]\n" : : "r" (ptr));
}

FORCE_INLINE rx_vec_f128 rx_load_vec_f128(const double* pd) {
	return vld1q_f64((const float64_t*)pd);
}

FORCE_INLINE void rx_store_vec_f128(double* mem_addr, rx_vec_f128 val) {
	vst1q_f64((float64_t*)mem_addr, val);
}

FORCE_INLINE rx_vec_f128 rx_swap_vec_f128(rx_vec_f128 a) {
	float64x2_t temp{};
	temp = vcopyq_laneq_f64(temp, 1, a, 1);
	a = vcopyq_laneq_f64(a, 1, a, 0);
	return vcopyq_laneq_f64(a, 0, temp, 1);
}

FORCE_INLINE rx_vec_f128 rx_set_vec_f128(uint64_t x1, uint64_t x0) {
	uint64x2_t temp0 = vdupq_n_u64(x0);
	uint64x2_t temp1 = vdupq_n_u64(x1);
	return vreinterpretq_f64_u64(vcopyq_laneq_u64(temp0, 1, temp1, 0));
}

FORCE_INLINE rx_vec_f128 rx_set1_vec_f128(uint64_t x) {
	return vreinterpretq_f64_u64(vdupq_n_u64(x));
}

#define rx_add_vec_f128 vaddq_f64
#define rx_sub_vec_f128 vsubq_f64
#define rx_mul_vec_f128 vmulq_f64
#define rx_div_vec_f128 vdivq_f64
#define rx_sqrt_vec_f128 vsqrtq_f64

FORCE_INLINE rx_vec_f128 rx_xor_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return vreinterpretq_f64_u8(veorq_u8(vreinterpretq_u8_f64(a), vreinterpretq_u8_f64(b)));
}

FORCE_INLINE rx_vec_f128 rx_and_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return vreinterpretq_f64_u8(vandq_u8(vreinterpretq_u8_f64(a), vreinterpretq_u8_f64(b)));
}

#define rx_and_vec_i128 vandq_u8

FORCE_INLINE rx_vec_f128 rx_or_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	return vreinterpretq_f64_u8(vorrq_u8(vreinterpretq_u8_f64(a), vreinterpretq_u8_f64(b)));
}

#ifdef __ARM_FEATURE_CRYPTO


FORCE_INLINE rx_vec_i128 rx_aesenc_vec_i128(rx_vec_i128 a, rx_vec_i128 key) {
	const uint8x16_t zero = { 0 };
	return vaesmcq_u8(vaeseq_u8(a, zero)) ^ key;
}

FORCE_INLINE rx_vec_i128 rx_aesdec_vec_i128(rx_vec_i128 a, rx_vec_i128 key) {
	const uint8x16_t zero = { 0 };
	return vaesimcq_u8(vaesdq_u8(a, zero)) ^ key;
}

#define HAVE_AES

#endif

#define rx_xor_vec_i128 veorq_u8

FORCE_INLINE int rx_vec_i128_x(rx_vec_i128 a) {
	return vgetq_lane_s32(vreinterpretq_s32_u8(a), 0);
}

FORCE_INLINE int rx_vec_i128_y(rx_vec_i128 a) {
	return vgetq_lane_s32(vreinterpretq_s32_u8(a), 1);
}

FORCE_INLINE int rx_vec_i128_z(rx_vec_i128 a) {
	return vgetq_lane_s32(vreinterpretq_s32_u8(a), 2);
}

FORCE_INLINE int rx_vec_i128_w(rx_vec_i128 a) {
	return vgetq_lane_s32(vreinterpretq_s32_u8(a), 3);
}

FORCE_INLINE rx_vec_i128 rx_set_int_vec_i128(int _I3, int _I2, int _I1, int _I0) {
	int32_t data[4];
	data[0] = _I0;
	data[1] = _I1;
	data[2] = _I2;
	data[3] = _I3;
	return vreinterpretq_u8_s32(vld1q_s32(data));
};

#define rx_xor_vec_i128 veorq_u8

FORCE_INLINE rx_vec_i128 rx_load_vec_i128(const rx_vec_i128* mem_addr) {
	return vld1q_u8((const uint8_t*)mem_addr);
}

FORCE_INLINE void rx_store_vec_i128(rx_vec_i128* mem_addr, rx_vec_i128 val) {
	vst1q_u8((uint8_t*)mem_addr, val);
}

FORCE_INLINE rx_vec_f128 rx_cvt_packed_int_vec_f128(const void* addr) {
	double lo = unsigned32ToSigned2sCompl(load32((uint8_t*)addr + 0));
	double hi = unsigned32ToSigned2sCompl(load32((uint8_t*)addr + 4));
	rx_vec_f128 x{};
	x = vsetq_lane_f64(lo, x, 0);
	x = vsetq_lane_f64(hi, x, 1);
	return x;
}

#define RANDOMX_DEFAULT_FENV

#else //portable fallback

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
#define rx_prefetch_t0(x)

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

FORCE_INLINE rx_vec_i128 rx_and_vec_i128(rx_vec_i128 a, rx_vec_i128 b) {
	rx_vec_i128 x;
	x.u64[0] = a.u64[0] & b.u64[0];
	x.u64[1] = a.u64[1] & b.u64[1];
	return x;
}

FORCE_INLINE rx_vec_f128 rx_or_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
	rx_vec_f128 x;
	x.i.u64[0] = a.i.u64[0] | b.i.u64[0];
	x.i.u64[1] = a.i.u64[1] | b.i.u64[1];
	return x;
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

#endif

#ifndef HAVE_AES
static const char* platformError = "Platform doesn't support hardware AES";

#include <stdexcept>

FORCE_INLINE rx_vec_i128 rx_aesenc_vec_i128(rx_vec_i128 v, rx_vec_i128 rkey) {
	throw std::runtime_error(platformError);
}

FORCE_INLINE rx_vec_i128 rx_aesdec_vec_i128(rx_vec_i128 v, rx_vec_i128 rkey) {
	throw std::runtime_error(platformError);
}
#endif

#ifdef RANDOMX_DEFAULT_FENV

void rx_reset_float_state();

void rx_set_rounding_mode(uint32_t mode);

#endif

double loadDoublePortable(const void* addr);
uint64_t mulh(uint64_t, uint64_t);
int64_t smulh(int64_t, int64_t);
uint64_t rotl64(uint64_t, unsigned int);
uint64_t rotr64(uint64_t, unsigned int);
