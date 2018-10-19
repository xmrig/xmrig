/* XMRig
 * Copyright 2010      Jeff Garzik  <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler       <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones  <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466     <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee    <jayddee246@gmail.com>
 * Copyright 2016      Imran Yusuff <https://github.com/imranyusuff>
 * Copyright 2016-2017 XMRig        <support@xmrig.com>
 * Copyright 2018      Sebastian Stolzenberg <https://github.com/sebastianstolzenberg>
 * Copyright 2018      BenDroid    <ben@graef.in>
 *
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __CRYPTONIGHT_ARM_H__
#define __CRYPTONIGHT_ARM_H__


#if defined(XMRIG_ARM) && !defined(__clang__)
#   include "aligned_malloc.h"
#else

#   include <mm_malloc.h>

#endif

#include <math.h>
#include <signal.h>

#include "crypto/CryptoNight.h"
#include "crypto/soft_aes.h"


extern "C"
{
#include "crypto/c_keccak.h"
#include "crypto/c_groestl.h"
#include "crypto/c_blake256.h"
#include "crypto/c_jh.h"
#include "crypto/c_skein.h"
}

static inline void do_blake_hash(const uint8_t* input, size_t len, uint8_t* output)
{
    blake256_hash(output, input, len);
}


static inline void do_groestl_hash(const uint8_t* input, size_t len, uint8_t* output)
{
    groestl(input, len * 8, output);
}


static inline void do_jh_hash(const uint8_t* input, size_t len, uint8_t* output)
{
    jh_hash(32 * 8, input, 8 * len, output);
}


static inline void do_skein_hash(const uint8_t* input, size_t len, uint8_t* output)
{
    xmr_skein(input, output);
}


void (* const extra_hashes[4])(const uint8_t*, size_t, uint8_t*) = {do_blake_hash, do_groestl_hash, do_jh_hash,
                                                                    do_skein_hash};


static inline __attribute__((always_inline)) __m128i _mm_set_epi64x(const uint64_t a, const uint64_t b)
{
    return vcombine_u64(vcreate_u64(b), vcreate_u64(a));
}

#if __ARM_FEATURE_CRYPTO
static inline __attribute__((always_inline)) __m128i _mm_aesenc_si128(__m128i v, __m128i rkey)
{
    alignas(16) const __m128i zero = { 0 };
    return veorq_u8(vaesmcq_u8(vaeseq_u8(v, zero)), rkey );
}
#else

static inline __attribute__((always_inline)) __m128i _mm_aesenc_si128(__m128i v, __m128i rkey)
{
    alignas(16) const __m128i zero = {0};
    return zero;
}

#endif

/* this one was not implemented yet so here it is */
static inline __attribute__((always_inline)) uint64_t _mm_cvtsi128_si64(__m128i a)
{
    return vgetq_lane_u64(a, 0);
}


#define EXTRACT64(X) _mm_cvtsi128_si64(X)


# define SHUFFLE_PHASE_1(l, idx, bx0, bx1, ax) \
{ \
    const uint64x2_t chunk1 = vld1q_u64((uint64_t*)((l) + ((idx) ^ 0x10))); \
    const uint64x2_t chunk2 = vld1q_u64((uint64_t*)((l) + ((idx) ^ 0x20))); \
    const uint64x2_t chunk3 = vld1q_u64((uint64_t*)((l) + ((idx) ^ 0x30))); \
    vst1q_u64((uint64_t*)((l) + ((idx) ^ 0x10)), vaddq_u64(chunk3, vreinterpretq_u64_u8(bx1))); \
    vst1q_u64((uint64_t*)((l) + ((idx) ^ 0x20)), vaddq_u64(chunk1, vreinterpretq_u64_u8(bx0))); \
    vst1q_u64((uint64_t*)((l) + ((idx) ^ 0x30)), vaddq_u64(chunk2, vreinterpretq_u64_u8(ax))); \
}

# define INTEGER_MATH_V2(idx, cl, cx) \
{ \
    const uint64_t cx_0 = _mm_cvtsi128_si64(cx); \
    cl ^= division_result_xmm##idx ^ (sqrt_result##idx << 32); \
    const uint32_t d = static_cast<uint32_t>(cx_0 + (sqrt_result##idx << 1)) | 0x80000001UL; \
    const uint64_t cx_1 = _mm_cvtsi128_si64(_mm_srli_si128(cx, 8)); \
    division_result_xmm##idx = static_cast<uint32_t>(cx_1 / d) + ((cx_1 % d) << 32); \
    const uint64_t sqrt_input = cx_0 + division_result_xmm##idx; \
    sqrt_result##idx = sqrt(sqrt_input + 18446744073709551616.0) * 2.0 - 8589934592.0; \
    const uint64_t s = sqrt_result##idx >> 1; \
    const uint64_t b = sqrt_result##idx & 1; \
    const uint64_t r2 = (uint64_t)(s) * (s + b) + (sqrt_result##idx << 32); \
    sqrt_result##idx += ((r2 + b > sqrt_input) ? -1 : 0) + ((r2 + (1ULL << 32) < sqrt_input - s) ? 1 : 0); \
}

# define SHUFFLE_PHASE_2(l, idx, bx0, bx1, ax, lo, hi) \
{ \
    const uint64x2_t chunk1 = veorq_u64(vld1q_u64((uint64_t*)((l) + ((idx) ^ 0x10))), vcombine_u64(vcreate_u64(hi), vcreate_u64(lo))); \
    const uint64x2_t chunk2 = vld1q_u64((uint64_t*)((l) + ((idx) ^ 0x20))); \
    const uint64x2_t chunk3 = vld1q_u64((uint64_t*)((l) + ((idx) ^ 0x30))); \
    hi ^= ((uint64_t*)((l) + ((idx) ^ 0x20)))[0]; \
    lo ^= ((uint64_t*)((l) + ((idx) ^ 0x20)))[1]; \
    vst1q_u64((uint64_t*)((l) + ((idx) ^ 0x10)), vaddq_u64(chunk3, vreinterpretq_u64_u8(bx1))); \
    vst1q_u64((uint64_t*)((l) + ((idx) ^ 0x20)), vaddq_u64(chunk1, vreinterpretq_u64_u8(bx0))); \
    vst1q_u64((uint64_t*)((l) + ((idx) ^ 0x30)), vaddq_u64(chunk2, vreinterpretq_u64_u8(ax))); \
}


#if defined (__arm64__) || defined (__aarch64__)
static inline uint64_t __umul128(uint64_t a, uint64_t b, uint64_t* hi)
{
    unsigned __int128 r = (unsigned __int128) a * (unsigned __int128) b;
    *hi = r >> 64;
    return (uint64_t) r;
}
#else

static inline uint64_t __umul128(uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi)
{
    uint64_t a = multiplier >> 32;
    uint64_t b = multiplier & 0xFFFFFFFF;
    uint64_t c = multiplicand >> 32;
    uint64_t d = multiplicand & 0xFFFFFFFF;

    uint64_t ad = a * d;
    uint64_t bd = b * d;

    uint64_t adbc = ad + (b * c);
    uint64_t adbc_carry = adbc < ad ? 1 : 0;

    uint64_t product_lo = bd + (adbc << 32);
    uint64_t product_lo_carry = product_lo < bd ? 1 : 0;
    *product_hi = (a * c) + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;

    return product_lo;
}

#endif


// This will shift and xor tmp1 into itself as 4 32-bit vals such as
// sl_xor(a1 a2 a3 a4) = a1 (a2^a1) (a3^a2^a1) (a4^a3^a2^a1)
static inline __m128i sl_xor(__m128i tmp1)
{
    __m128i tmp4;
    tmp4 = _mm_slli_si128(tmp1, 0x04);
    tmp1 = _mm_xor_si128(tmp1, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    tmp1 = _mm_xor_si128(tmp1, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    tmp1 = _mm_xor_si128(tmp1, tmp4);
    return tmp1;
}


template<uint8_t rcon>
static inline void soft_aes_genkey_sub(__m128i* xout0, __m128i* xout2)
{
    __m128i xout1 = soft_aeskeygenassist<rcon>(*xout2);
    xout1 = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
    *xout0 = sl_xor(*xout0);
    *xout0 = _mm_xor_si128(*xout0, xout1);
    xout1 = soft_aeskeygenassist<0x00>(*xout0);
    xout1 = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
    *xout2 = sl_xor(*xout2);
    *xout2 = _mm_xor_si128(*xout2, xout1);
}


template<bool SOFT_AES>
static inline void
aes_genkey(const __m128i* memory, __m128i* k0, __m128i* k1, __m128i* k2, __m128i* k3, __m128i* k4, __m128i* k5,
           __m128i* k6, __m128i* k7, __m128i* k8, __m128i* k9)
{
    __m128i xout0 = _mm_load_si128(memory);
    __m128i xout2 = _mm_load_si128(memory + 1);
    *k0 = xout0;
    *k1 = xout2;

    soft_aes_genkey_sub<0x01>(&xout0, &xout2);
    *k2 = xout0;
    *k3 = xout2;

    soft_aes_genkey_sub<0x02>(&xout0, &xout2);
    *k4 = xout0;
    *k5 = xout2;

    soft_aes_genkey_sub<0x04>(&xout0, &xout2);
    *k6 = xout0;
    *k7 = xout2;

    soft_aes_genkey_sub<0x08>(&xout0, &xout2);
    *k8 = xout0;
    *k9 = xout2;
}


template<bool SOFT_AES>
static inline void
aes_round(__m128i key, __m128i* x0, __m128i* x1, __m128i* x2, __m128i* x3, __m128i* x4, __m128i* x5, __m128i* x6,
          __m128i* x7)
{
    if (SOFT_AES) {
        *x0 = soft_aesenc((uint32_t*)x0, key);
        *x1 = soft_aesenc((uint32_t*)x1, key);
        *x2 = soft_aesenc((uint32_t*)x2, key);
        *x3 = soft_aesenc((uint32_t*)x3, key);
        *x4 = soft_aesenc((uint32_t*)x4, key);
        *x5 = soft_aesenc((uint32_t*)x5, key);
        *x6 = soft_aesenc((uint32_t*)x6, key);
        *x7 = soft_aesenc((uint32_t*)x7, key);
    }
#   ifndef XMRIG_ARMv7
    else {
        *x0 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x0), key));
        *x1 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x1), key));
        *x2 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x2), key));
        *x3 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x3), key));
        *x4 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x4), key));
        *x5 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x5), key));
        *x6 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x6), key));
        *x7 = vaesmcq_u8(vaeseq_u8(*((uint8x16_t *) x7), key));
    }
#   else
    else {
        *x0 = _mm_aesenc_si128(*x0, key);
        *x1 = _mm_aesenc_si128(*x1, key);
        *x2 = _mm_aesenc_si128(*x2, key);
        *x3 = _mm_aesenc_si128(*x3, key);
        *x4 = _mm_aesenc_si128(*x4, key);
        *x5 = _mm_aesenc_si128(*x5, key);
        *x6 = _mm_aesenc_si128(*x6, key);
        *x7 = _mm_aesenc_si128(*x7, key);
    }
#   endif
}


inline void mix_and_propagate(__m128i& x0, __m128i& x1, __m128i& x2, __m128i& x3, __m128i& x4, __m128i& x5, __m128i& x6,
                              __m128i& x7)
{
    __m128i tmp0 = x0;
    x0 = _mm_xor_si128(x0, x1);
    x1 = _mm_xor_si128(x1, x2);
    x2 = _mm_xor_si128(x2, x3);
    x3 = _mm_xor_si128(x3, x4);
    x4 = _mm_xor_si128(x4, x5);
    x5 = _mm_xor_si128(x5, x6);
    x6 = _mm_xor_si128(x6, x7);
    x7 = _mm_xor_si128(x7, tmp0);
}

template<size_t MEM, bool SOFT_AES>
static inline void cn_explode_scratchpad(const __m128i* input, __m128i* output)
{
    __m128i xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7;
    __m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    aes_genkey<SOFT_AES>(input, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xin0 = _mm_load_si128(input + 4);
    xin1 = _mm_load_si128(input + 5);
    xin2 = _mm_load_si128(input + 6);
    xin3 = _mm_load_si128(input + 7);
    xin4 = _mm_load_si128(input + 8);
    xin5 = _mm_load_si128(input + 9);
    xin6 = _mm_load_si128(input + 10);
    xin7 = _mm_load_si128(input + 11);

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8) {
        if (!SOFT_AES) {
            aes_round<SOFT_AES>(_mm_setzero_si128(), &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        }

        aes_round<SOFT_AES>(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);

        if (!SOFT_AES) {
            xin0 ^= k9;
            xin1 ^= k9;
            xin2 ^= k9;
            xin3 ^= k9;
            xin4 ^= k9;
            xin5 ^= k9;
            xin6 ^= k9;
            xin7 ^= k9;
        } else {
            aes_round<SOFT_AES>(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        }

        _mm_store_si128(output + i + 0, xin0);
        _mm_store_si128(output + i + 1, xin1);
        _mm_store_si128(output + i + 2, xin2);
        _mm_store_si128(output + i + 3, xin3);
        _mm_store_si128(output + i + 4, xin4);
        _mm_store_si128(output + i + 5, xin5);
        _mm_store_si128(output + i + 6, xin6);
        _mm_store_si128(output + i + 7, xin7);
    }
}

template<size_t MEM, bool SOFT_AES>
static inline void cn_explode_scratchpad_heavy(const __m128i* input, __m128i* output)
{
    __m128i xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7;
    __m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    aes_genkey<SOFT_AES>(input, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xin0 = _mm_load_si128(input + 4);
    xin1 = _mm_load_si128(input + 5);
    xin2 = _mm_load_si128(input + 6);
    xin3 = _mm_load_si128(input + 7);
    xin4 = _mm_load_si128(input + 8);
    xin5 = _mm_load_si128(input + 9);
    xin6 = _mm_load_si128(input + 10);
    xin7 = _mm_load_si128(input + 11);

    for (size_t i = 0; i < 16; i++) {

        if (!SOFT_AES) {
            aes_round<SOFT_AES>(_mm_setzero_si128(), &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        }

        aes_round<SOFT_AES>(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);

        if (!SOFT_AES) {
            xin0 ^= k9;
            xin1 ^= k9;
            xin2 ^= k9;
            xin3 ^= k9;
            xin4 ^= k9;
            xin5 ^= k9;
            xin6 ^= k9;
            xin7 ^= k9;
        } else {
            aes_round<SOFT_AES>(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        }

        mix_and_propagate(xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
    }

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8) {
        if (!SOFT_AES) {
            aes_round<SOFT_AES>(_mm_setzero_si128(), &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        }

        aes_round<SOFT_AES>(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);

        if (!SOFT_AES) {
            xin0 ^= k9;
            xin1 ^= k9;
            xin2 ^= k9;
            xin3 ^= k9;
            xin4 ^= k9;
            xin5 ^= k9;
            xin6 ^= k9;
            xin7 ^= k9;
        } else {
            aes_round<SOFT_AES>(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        }

        _mm_store_si128(output + i + 0, xin0);
        _mm_store_si128(output + i + 1, xin1);
        _mm_store_si128(output + i + 2, xin2);
        _mm_store_si128(output + i + 3, xin3);
        _mm_store_si128(output + i + 4, xin4);
        _mm_store_si128(output + i + 5, xin5);
        _mm_store_si128(output + i + 6, xin6);
        _mm_store_si128(output + i + 7, xin7);
    }
}

template<size_t MEM, bool SOFT_AES>
static inline void cn_implode_scratchpad(const __m128i* input, __m128i* output)
{
    __m128i xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7;
    __m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    aes_genkey<SOFT_AES>(output + 2, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xout0 = _mm_load_si128(output + 4);
    xout1 = _mm_load_si128(output + 5);
    xout2 = _mm_load_si128(output + 6);
    xout3 = _mm_load_si128(output + 7);
    xout4 = _mm_load_si128(output + 8);
    xout5 = _mm_load_si128(output + 9);
    xout6 = _mm_load_si128(output + 10);
    xout7 = _mm_load_si128(output + 11);

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8) {
        xout0 = _mm_xor_si128(_mm_load_si128(input + i + 0), xout0);
        xout1 = _mm_xor_si128(_mm_load_si128(input + i + 1), xout1);
        xout2 = _mm_xor_si128(_mm_load_si128(input + i + 2), xout2);
        xout3 = _mm_xor_si128(_mm_load_si128(input + i + 3), xout3);
        xout4 = _mm_xor_si128(_mm_load_si128(input + i + 4), xout4);
        xout5 = _mm_xor_si128(_mm_load_si128(input + i + 5), xout5);
        xout6 = _mm_xor_si128(_mm_load_si128(input + i + 6), xout6);
        xout7 = _mm_xor_si128(_mm_load_si128(input + i + 7), xout7);

        if (!SOFT_AES) {
            aes_round<SOFT_AES>(_mm_setzero_si128(), &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);

        if (!SOFT_AES) {
            xout0 ^= k9;
            xout1 ^= k9;
            xout2 ^= k9;
            xout3 ^= k9;
            xout4 ^= k9;
            xout5 ^= k9;
            xout6 ^= k9;
            xout7 ^= k9;
        } else {
            aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }
    }

    _mm_store_si128(output + 4, xout0);
    _mm_store_si128(output + 5, xout1);
    _mm_store_si128(output + 6, xout2);
    _mm_store_si128(output + 7, xout3);
    _mm_store_si128(output + 8, xout4);
    _mm_store_si128(output + 9, xout5);
    _mm_store_si128(output + 10, xout6);
    _mm_store_si128(output + 11, xout7);
}

template<size_t MEM, bool SOFT_AES>
static inline void cn_implode_scratchpad_heavy(const __m128i* input, __m128i* output)
{
    __m128i xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7;
    __m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    aes_genkey<SOFT_AES>(output + 2, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xout0 = _mm_load_si128(output + 4);
    xout1 = _mm_load_si128(output + 5);
    xout2 = _mm_load_si128(output + 6);
    xout3 = _mm_load_si128(output + 7);
    xout4 = _mm_load_si128(output + 8);
    xout5 = _mm_load_si128(output + 9);
    xout6 = _mm_load_si128(output + 10);
    xout7 = _mm_load_si128(output + 11);

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8) {
        xout0 = _mm_xor_si128(_mm_load_si128(input + i + 0), xout0);
        xout1 = _mm_xor_si128(_mm_load_si128(input + i + 1), xout1);
        xout2 = _mm_xor_si128(_mm_load_si128(input + i + 2), xout2);
        xout3 = _mm_xor_si128(_mm_load_si128(input + i + 3), xout3);
        xout4 = _mm_xor_si128(_mm_load_si128(input + i + 4), xout4);
        xout5 = _mm_xor_si128(_mm_load_si128(input + i + 5), xout5);
        xout6 = _mm_xor_si128(_mm_load_si128(input + i + 6), xout6);
        xout7 = _mm_xor_si128(_mm_load_si128(input + i + 7), xout7);

        if (!SOFT_AES) {
            aes_round<SOFT_AES>(_mm_setzero_si128(), &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);

        if (!SOFT_AES) {
            xout0 ^= k9;
            xout1 ^= k9;
            xout2 ^= k9;
            xout3 ^= k9;
            xout4 ^= k9;
            xout5 ^= k9;
            xout6 ^= k9;
            xout7 ^= k9;
        } else {
            aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }

        mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
    }

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8) {
        xout0 = _mm_xor_si128(_mm_load_si128(input + i + 0), xout0);
        xout1 = _mm_xor_si128(_mm_load_si128(input + i + 1), xout1);
        xout2 = _mm_xor_si128(_mm_load_si128(input + i + 2), xout2);
        xout3 = _mm_xor_si128(_mm_load_si128(input + i + 3), xout3);
        xout4 = _mm_xor_si128(_mm_load_si128(input + i + 4), xout4);
        xout5 = _mm_xor_si128(_mm_load_si128(input + i + 5), xout5);
        xout6 = _mm_xor_si128(_mm_load_si128(input + i + 6), xout6);
        xout7 = _mm_xor_si128(_mm_load_si128(input + i + 7), xout7);

        if (!SOFT_AES) {
            aes_round<SOFT_AES>(_mm_setzero_si128(), &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);

        if (!SOFT_AES) {
            xout0 ^= k9;
            xout1 ^= k9;
            xout2 ^= k9;
            xout3 ^= k9;
            xout4 ^= k9;
            xout5 ^= k9;
            xout6 ^= k9;
            xout7 ^= k9;
        } else {
            aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }

        mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
    }

    for (size_t i = 0; i < 16; i++) {
        if (!SOFT_AES) {
            aes_round<SOFT_AES>(_mm_setzero_si128(), &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);

        if (!SOFT_AES) {
            xout0 ^= k9;
            xout1 ^= k9;
            xout2 ^= k9;
            xout3 ^= k9;
            xout4 ^= k9;
            xout5 ^= k9;
            xout6 ^= k9;
            xout7 ^= k9;
        } else {
            aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        }

        mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
    }

    _mm_store_si128(output + 4, xout0);
    _mm_store_si128(output + 5, xout1);
    _mm_store_si128(output + 6, xout2);
    _mm_store_si128(output + 7, xout3);
    _mm_store_si128(output + 8, xout4);
    _mm_store_si128(output + 9, xout5);
    _mm_store_si128(output + 10, xout6);
    _mm_store_si128(output + 11, xout7);
}

// n-Loop version. Seems to be little bit slower then the hardcoded one.
template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES, size_t NUM_HASH_BLOCKS>
class CryptoNightMultiHash
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        __m128i cx[NUM_HASH_BLOCKS];
        __m128i ax[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size,
                   scratchPad[hashBlock]->state, 200);
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = scratchPad[hashBlock]->memory;
            h[hashBlock] = reinterpret_cast<uint64_t*>(scratchPad[hashBlock]->state);

            cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx[hashBlock] = _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
        }

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                ax[hashBlock] = _mm_set_epi64x(ah[hashBlock], al[hashBlock]);

                if (SOFT_AES) {
                    cx[hashBlock] = soft_aesenc((uint32_t *) &l[hashBlock][idx[hashBlock] & MASK], ax[hashBlock]);
                } else {
                    cx[hashBlock] = _mm_load_si128((__m128i *) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx[hashBlock] = _mm_aesenc_si128(cx[hashBlock], ax[hashBlock]);
                }
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK],
                                _mm_xor_si128(bx[hashBlock], cx[hashBlock]));
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                idx[hashBlock] = EXTRACT64(cx[hashBlock]);
            }

            uint64_t hi, lo, cl, ch;
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                cl = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0];
                ch = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1];
                lo = __umul128(idx[hashBlock], cl, &hi);

                al[hashBlock] += hi;
                ah[hashBlock] += lo;

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = al[hashBlock];
                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] = ah[hashBlock];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];

                bx[hashBlock] = cx[hashBlock];
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[scratchPad[hashBlock]->state[0] & 3](scratchPad[hashBlock]->state, 200,
                                                              output + hashBlock * 32);
        }
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];
        uint64_t tweak1_2[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        __m128i cx[NUM_HASH_BLOCKS];
        __m128i ax[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, scratchPad[hashBlock]->state,
                   200);
            tweak1_2[hashBlock] = (*reinterpret_cast<const uint64_t*>(input + 35 + hashBlock * size) ^
                                   *(reinterpret_cast<const uint64_t*>(scratchPad[hashBlock]->state) + 24));
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = scratchPad[hashBlock]->memory;
            h[hashBlock] = reinterpret_cast<uint64_t*>(scratchPad[hashBlock]->state);

            cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx[hashBlock] = _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
        }

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                ax[hashBlock] = _mm_set_epi64x(ah[hashBlock], al[hashBlock]);

                if (SOFT_AES) {
                    cx[hashBlock] = soft_aesenc((uint32_t *) &l[hashBlock][idx[hashBlock] & MASK], ax[hashBlock]);
                } else {
                    cx[hashBlock] = _mm_load_si128((__m128i *) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx[hashBlock] = _mm_aesenc_si128(cx[hashBlock], ax[hashBlock]);
                }
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                _mm_store_si128((__m128i *) &l[hashBlock][idx[hashBlock] & MASK],
                                _mm_xor_si128(bx[hashBlock], cx[hashBlock]));
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                const uint8_t tmp = reinterpret_cast<const uint8_t *>(&l[hashBlock][idx[hashBlock] & MASK])[11];
                static const uint32_t table = 0x75310;
                const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
                ((uint8_t *) (&l[hashBlock][idx[hashBlock] & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                idx[hashBlock] = EXTRACT64(cx[hashBlock]);
            }

            uint64_t hi, lo, cl, ch;
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                cl = ((uint64_t *) &l[hashBlock][idx[hashBlock] & MASK])[0];
                ch = ((uint64_t *) &l[hashBlock][idx[hashBlock] & MASK])[1];
                lo = __umul128(idx[hashBlock], cl, &hi);

                al[hashBlock] += hi;
                ah[hashBlock] += lo;

                ah[hashBlock] ^= tweak1_2[hashBlock];

                ((uint64_t *) &l[hashBlock][idx[hashBlock] & MASK])[0] = al[hashBlock];
                ((uint64_t *) &l[hashBlock][idx[hashBlock] & MASK])[1] = ah[hashBlock];

                ah[hashBlock] ^= tweak1_2[hashBlock];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];

                bx[hashBlock] = cx[hashBlock];
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[scratchPad[hashBlock]->state[0] & 3](scratchPad[hashBlock]->state, 200,
                                                              output + hashBlock * 32);
        }
    }

    // multi
    inline static void hashPowV3(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];
        uint64_t sqrt_result[NUM_HASH_BLOCKS];
        uint64_t division_result_xmm[NUM_HASH_BLOCKS];
        __m128i bx0[NUM_HASH_BLOCKS];
        __m128i bx1[NUM_HASH_BLOCKS];
        __m128i cx[NUM_HASH_BLOCKS];
        __m128i ax[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size,
                   scratchPad[hashBlock]->state, 200);
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = scratchPad[hashBlock]->memory;
            h[hashBlock] = reinterpret_cast<uint64_t*>(scratchPad[hashBlock]->state);

            cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx0[hashBlock] = _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            bx1[hashBlock] = _mm_set_epi64x(h[hashBlock][9] ^ h[hashBlock][11], h[hashBlock][8] ^ h[hashBlock][10]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];

            division_result_xmm[hashBlock] = h[hashBlock][12];
            sqrt_result[hashBlock] = h[hashBlock][13];
        }

        uint64_t sqrt_result0;
        uint64_t division_result_xmm0;

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                ax[hashBlock] = _mm_set_epi64x(ah[hashBlock], al[hashBlock]);

                if (SOFT_AES) {
                    cx[hashBlock] = soft_aesenc((uint32_t *) &l[hashBlock][idx[hashBlock] & MASK], ax[hashBlock]);
                } else {
                    cx[hashBlock] = _mm_load_si128((__m128i *) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx[hashBlock] = _mm_aesenc_si128(cx[hashBlock], ax[hashBlock]);
                }
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                SHUFFLE_PHASE_1(l[hashBlock], idx[hashBlock] & MASK, bx0[hashBlock], bx1[hashBlock], ax[hashBlock])
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK],
                                _mm_xor_si128(bx0[hashBlock], cx[hashBlock]));
            }

            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                idx[hashBlock] = EXTRACT64(cx[hashBlock]);
            }

            uint64_t hi, lo, cl, ch;
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                cl = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0];
                ch = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1];

                sqrt_result0 = sqrt_result[hashBlock];
                division_result_xmm0 = division_result_xmm[hashBlock];

                INTEGER_MATH_V2(0, cl, cx[hashBlock])

                sqrt_result[hashBlock] = sqrt_result0;
                division_result_xmm[hashBlock] = division_result_xmm0;

                lo = __umul128(idx[hashBlock], cl, &hi);

                SHUFFLE_PHASE_2(l[hashBlock], idx[hashBlock] & MASK, bx0[hashBlock], bx1[hashBlock], ax[hashBlock], lo, hi)

                al[hashBlock] += hi;
                ah[hashBlock] += lo;

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = al[hashBlock];
                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] = ah[hashBlock];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];

                bx1[hashBlock] = bx0[hashBlock];
                bx0[hashBlock] = cx[hashBlock];
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[scratchPad[hashBlock]->state[0] & 3](scratchPad[hashBlock]->state, 200,
                                                              output + hashBlock * 32);
        }
    }

    inline static void hashLiteTube(const uint8_t* __restrict__ input,
                                    size_t size,
                                    uint8_t* __restrict__ output,
                                    ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];
        uint64_t tweak1_2[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, scratchPad[hashBlock]->state,
                   200);
            tweak1_2[hashBlock] = (*reinterpret_cast<const uint64_t*>(input + 35 + hashBlock * size) ^
                                   *(reinterpret_cast<const uint64_t*>(scratchPad[hashBlock]->state) + 24));
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = scratchPad[hashBlock]->memory;
            h[hashBlock] = reinterpret_cast<uint64_t*>(scratchPad[hashBlock]->state);

            cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx[hashBlock] =
                    _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
        }

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                __m128i cx;

                if (SOFT_AES) {
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK],
                                     _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                } else {
                    cx = _mm_load_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                }

                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK],
                                _mm_xor_si128(bx[hashBlock], cx));

                const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[hashBlock][idx[hashBlock] & MASK])[11];
                static const uint32_t table = 0x75310;
                const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
                ((uint8_t*) (&l[hashBlock][idx[hashBlock] & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

                idx[hashBlock] = EXTRACT64(cx);
                bx[hashBlock] = cx;

                uint64_t hi, lo, cl, ch;
                cl = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0];
                ch = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1];
                lo = __umul128(idx[hashBlock], cl, &hi);

                al[hashBlock] += hi;
                ah[hashBlock] += lo;

                ah[hashBlock] ^= tweak1_2[hashBlock];

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = al[hashBlock];
                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] = ah[hashBlock];

                ah[hashBlock] ^= tweak1_2[hashBlock];

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] ^= ((uint64_t*) &l[hashBlock][idx[hashBlock] &
                                                                                                    MASK])[0];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[scratchPad[hashBlock]->state[0] & 3](scratchPad[hashBlock]->state, 200,
                                                              output + hashBlock * 32);
        }
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size,
                   scratchPad[hashBlock]->state, 200);
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = scratchPad[hashBlock]->memory;
            h[hashBlock] = reinterpret_cast<uint64_t*>(scratchPad[hashBlock]->state);

            cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx[hashBlock] = _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
        }

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                __m128i cx;

                if (SOFT_AES) {
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK],
                                     _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                } else {
                    cx = _mm_load_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                }

                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK],
                                _mm_xor_si128(bx[hashBlock], cx));

                idx[hashBlock] = EXTRACT64(cx);
                bx[hashBlock] = cx;

                uint64_t hi, lo, cl, ch;
                cl = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0];
                ch = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1];
                lo = __umul128(idx[hashBlock], cl, &hi);

                al[hashBlock] += hi;
                ah[hashBlock] += lo;

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = al[hashBlock];
                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] = ah[hashBlock];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];

                const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t*>(&l[hashBlock][idx[hashBlock] & MASK]));
                const int64_t n = vgetq_lane_s64(x, 0);
                const int32_t d = vgetq_lane_s32(x, 2);
                const int64_t q = n / (d | 0x5);

                ((int64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = n ^ q;

                idx[hashBlock] = d ^ q;
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[scratchPad[hashBlock]->state[0] & 3](scratchPad[hashBlock]->state, 200,
                                                              output + hashBlock * 32);
        }
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                      size_t size,
                                      uint8_t* __restrict__ output,
                                      ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size,
                   scratchPad[hashBlock]->state, 200);
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = scratchPad[hashBlock]->memory;
            h[hashBlock] = reinterpret_cast<uint64_t*>(scratchPad[hashBlock]->state);

            cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx[hashBlock] = _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
        }

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                __m128i cx;

                if (SOFT_AES) {
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK],
                                     _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                } else {
                    cx = _mm_load_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                }

                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK],
                                _mm_xor_si128(bx[hashBlock], cx));

                idx[hashBlock] = EXTRACT64(cx);
                bx[hashBlock] = cx;

                uint64_t hi, lo, cl, ch;
                cl = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0];
                ch = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1];
                lo = __umul128(idx[hashBlock], cl, &hi);

                al[hashBlock] += hi;
                ah[hashBlock] += lo;

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = al[hashBlock];
                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] = ah[hashBlock];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];

                const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t*>(&l[hashBlock][idx[hashBlock] & MASK]));
                const int64_t n = vgetq_lane_s64(x, 0);
                const int32_t d = vgetq_lane_s32(x, 2);
                const int64_t q = n / (d | 0x5);

                ((int64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = n ^ q;

                idx[hashBlock] = (~d) ^ q;
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[scratchPad[hashBlock]->state[0] & 3](scratchPad[hashBlock]->state, 200,
                                                              output + hashBlock * 32);
        }
    }

    inline static void hashHeavyTube(const uint8_t* __restrict__ input,
                                     size_t size,
                                     uint8_t* __restrict__ output,
                                     ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];
        uint64_t tweak1_2[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, scratchPad[hashBlock]->state,
                   200);
            tweak1_2[hashBlock] = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 +
                                                                      hashBlock * size) ^
                                   *(reinterpret_cast<const uint64_t*>(scratchPad[hashBlock]->state) + 24));
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = scratchPad[hashBlock]->memory;
            h[hashBlock] = reinterpret_cast<uint64_t*>(scratchPad[hashBlock]->state);

            cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx[hashBlock] = _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
        }

        union alignas(16)
        {
            uint32_t k[4];
            uint64_t v64[2];
        };
        alignas(16) uint32_t x[4];

#define BYTE(p, i) ((unsigned char*)&p)[i]

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                __m128i cx;

                cx = _mm_load_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK]);

                const __m128i& key = _mm_set_epi64x(ah[hashBlock], al[hashBlock]);

                _mm_store_si128((__m128i*) k, key);
                cx = _mm_xor_si128(cx, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
                _mm_store_si128((__m128i*) x, cx);

                k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^
                        saes_table[3][BYTE(x[3], 3)];
                x[0] ^= k[0];
                k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^
                        saes_table[3][BYTE(x[0], 3)];
                x[1] ^= k[1];
                k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^
                        saes_table[3][BYTE(x[1], 3)];
                x[2] ^= k[2];
                k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^
                        saes_table[3][BYTE(x[2], 3)];

                cx = _mm_load_si128((__m128i*) k);

                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK], _mm_xor_si128(bx[hashBlock], cx));

                const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[hashBlock][idx[hashBlock] & MASK])[11];
                static const uint32_t table = 0x75310;
                const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
                ((uint8_t*) (&l[hashBlock][idx[hashBlock] & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

                idx[hashBlock] = EXTRACT64(cx);
                bx[hashBlock] = cx;

                uint64_t hi, lo, cl, ch;
                cl = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0];
                ch = ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1];
                lo = __umul128(idx[hashBlock], cl, &hi);

                al[hashBlock] += hi;
                ah[hashBlock] += lo;

                ah[hashBlock] ^= tweak1_2[hashBlock];

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = al[hashBlock];
                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] = ah[hashBlock];

                ah[hashBlock] ^= tweak1_2[hashBlock];

                ((uint64_t*) &l[hashBlock][idx[hashBlock] & MASK])[1] ^= ((uint64_t*) &l[hashBlock][idx[hashBlock] &
                                                                                                    MASK])[0];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];

                const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t*>(&l[hashBlock][idx[hashBlock] & MASK]));
                const int64_t n = vgetq_lane_s64(x, 0);
                const int32_t d = vgetq_lane_s32(x, 2);
                const int64_t q = n / (d | 0x5);

                ((int64_t*) &l[hashBlock][idx[hashBlock] & MASK])[0] = n ^ q;

                idx[hashBlock] = d ^ q;
            }
        }

#undef BYTE

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[scratchPad[hashBlock]->state[0] & 3](scratchPad[hashBlock]->state, 200,
                                                              output + hashBlock * 32);
        }
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 1>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, scratchPad[0]->state, 200);

        l = scratchPad[0]->memory;
        h = reinterpret_cast<uint64_t*>(scratchPad[0]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*) &l[idx & MASK], _mm_set_epi64x(ah, al));
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah, al));
            }

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            idx = EXTRACT64(cx);
            bx = cx;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l[idx & MASK])[0];
            ch = ((uint64_t*) &l[idx & MASK])[1];
            lo = __umul128(idx, cl, &hi);

            al += hi;
            ah += lo;

            ((uint64_t*) &l[idx & MASK])[0] = al;
            ((uint64_t*) &l[idx & MASK])[1] = ah;

            ah ^= ch;
            al ^= cl;
            idx = al;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, scratchPad[0]->state, 200);

        uint64_t tweak1_2 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                             *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        l = scratchPad[0]->memory;
        h = reinterpret_cast<uint64_t*>(scratchPad[0]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*) &l[idx & MASK], _mm_set_epi64x(ah, al));
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah, al));
            }

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[idx & MASK])[11];
            static const uint32_t table = 0x75310;
            const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l[idx & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            idx = EXTRACT64(cx);
            bx = cx;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l[idx & MASK])[0];
            ch = ((uint64_t*) &l[idx & MASK])[1];
            lo = __umul128(idx, cl, &hi);

            al += hi;
            ah += lo;

            ah ^= tweak1_2;
            ((uint64_t*) &l[idx & MASK])[0] = al;
            ((uint64_t*) &l[idx & MASK])[1] = ah;
            ah ^= tweak1_2;

            ah ^= ch;
            al ^= cl;
            idx = al;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
    }

    // single
    inline static void hashPowV3(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        uint64_t idx;
        __m128i bx0;
        __m128i bx1;

        keccak(static_cast<const uint8_t*>(input), (int) size, scratchPad[0]->state, 200);

        l = scratchPad[0]->memory;
        h = reinterpret_cast<uint64_t*>(scratchPad[0]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx0 = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        bx1 = _mm_set_epi64x(h[9] ^ h[11], h[8] ^ h[10]);
        idx = h[0] ^ h[4];

        uint64_t division_result_xmm0 = h[12];
        uint64_t sqrt_result0 =  h[13];

        for (size_t i = 0; i < ITERATIONS; i++) {
            const __m128i ax = _mm_set_epi64x(ah, al);

            __m128i cx;
            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*) &l[idx & MASK], ax);
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, ax);
            }

            SHUFFLE_PHASE_1(l, (idx&MASK), bx0, bx1, ax)

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx0, cx));

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l[idx & MASK])[0];
            ch = ((uint64_t*) &l[idx & MASK])[1];

            INTEGER_MATH_V2(0, cl, cx)

            lo = __umul128(idx, cl, &hi);

            SHUFFLE_PHASE_2(l, (idx&MASK), bx0, bx1, ax, lo, hi)

            al += hi;
            ah += lo;

            ((uint64_t*) &l[idx & MASK])[0] = al;
            ((uint64_t*) &l[idx & MASK])[1] = ah;

            ah ^= ch;
            al ^= cl;
            idx = al;

            bx0 = cx;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
    }

    inline static void hashLiteTube(const uint8_t* __restrict__ input,
                                    size_t size,
                                    uint8_t* __restrict__ output,
                                    ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, scratchPad[0]->state, 200);

        uint64_t tweak1_2 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                             *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        l = scratchPad[0]->memory;
        h = reinterpret_cast<uint64_t*>(scratchPad[0]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*) &l[idx & MASK], _mm_set_epi64x(ah, al));
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah, al));
            }

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[idx & MASK])[11];
            static const uint32_t table = 0x75310;
            const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l[idx & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            idx = EXTRACT64(cx);
            bx = cx;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l[idx & MASK])[0];
            ch = ((uint64_t*) &l[idx & MASK])[1];
            lo = __umul128(idx, cl, &hi);

            al += hi;
            ah += lo;

            ah ^= tweak1_2;
            ((uint64_t*) &l[idx & MASK])[0] = al;
            ((uint64_t*) &l[idx & MASK])[1] = ah;
            ah ^= tweak1_2;

            ((uint64_t*) &l[idx & MASK])[1] ^= ((uint64_t*) &l[idx & MASK])[0];

            ah ^= ch;
            al ^= cl;
            idx = al;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, scratchPad[0]->state, 200);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) scratchPad[0]->state, (__m128i*) scratchPad[0]->memory);

        l = scratchPad[0]->memory;
        h = reinterpret_cast<uint64_t*>(scratchPad[0]->state);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*) &l[idx & MASK], _mm_set_epi64x(ah, al));
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah, al));
            }

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            idx = EXTRACT64(cx);
            bx = cx;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l[idx & MASK])[0];
            ch = ((uint64_t*) &l[idx & MASK])[1];
            lo = __umul128(idx, cl, &hi);

            al += hi;
            ah += lo;

            ((uint64_t*) &l[idx & MASK])[0] = al;
            ((uint64_t*) &l[idx & MASK])[1] = ah;

            ah ^= ch;
            al ^= cl;
            idx = al;

            const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t*>(&l[idx & MASK]));
            const int64_t n = vgetq_lane_s64(x, 0);
            const int32_t d = vgetq_lane_s32(x, 2);
            const int64_t q = n / (d | 0x5);

            ((int64_t*) &l[idx & MASK])[0] = n ^ q;

            idx = d ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) scratchPad[0]->memory, (__m128i*) scratchPad[0]->state);
        keccakf(h, 24);
        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                      size_t size,
                                      uint8_t* __restrict__ output,
                                      ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, scratchPad[0]->state, 200);

        l = scratchPad[0]->memory;
        h = reinterpret_cast<uint64_t*>(scratchPad[0]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*) &l[idx & MASK], _mm_set_epi64x(ah, al));
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah, al));
            }

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            idx = EXTRACT64(cx);
            bx = cx;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l[idx & MASK])[0];
            ch = ((uint64_t*) &l[idx & MASK])[1];
            lo = __umul128(idx, cl, &hi);

            al += hi;
            ah += lo;

            ((uint64_t*) &l[idx & MASK])[0] = al;
            ((uint64_t*) &l[idx & MASK])[1] = ah;

            ah ^= ch;
            al ^= cl;
            idx = al;

            const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t*>(&l[idx & MASK]));
            const int64_t n = vgetq_lane_s64(x, 0);
            const int32_t d = vgetq_lane_s32(x, 2);
            const int64_t q = n / (d | 0x5);

            ((int64_t*) &l[idx & MASK])[0] = n ^ q;

            idx = (~d) ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
    }


    inline static void hashHeavyTube(const uint8_t* __restrict__ input,
                                     size_t size,
                                     uint8_t* __restrict__ output,
                                     ScratchPad** __restrict__ scratchPad)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, scratchPad[0]->state, 200);

        uint64_t tweak1_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                             *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));

        l = scratchPad[0]->memory;
        h = reinterpret_cast<uint64_t*>(scratchPad[0]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        union alignas(16)
        {
            uint32_t k[4];
            uint64_t v64[2];
        };
        alignas(16) uint32_t x[4];

#define BYTE(p, i) ((unsigned char*)&p)[i]
        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx = _mm_load_si128((__m128i*) &l[idx & MASK]);

            const __m128i& key = _mm_set_epi64x(ah, al);

            _mm_store_si128((__m128i*) k, key);
            cx = _mm_xor_si128(cx, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
            _mm_store_si128((__m128i*) x, cx);

            k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^
                    saes_table[3][BYTE(x[3], 3)];
            x[0] ^= k[0];
            k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^
                    saes_table[3][BYTE(x[0], 3)];
            x[1] ^= k[1];
            k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^
                    saes_table[3][BYTE(x[1], 3)];
            x[2] ^= k[2];
            k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^
                    saes_table[3][BYTE(x[2], 3)];

            cx = _mm_load_si128((__m128i*) k);

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[idx & MASK])[11];
            static const uint32_t table = 0x75310;
            const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l[idx & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx = EXTRACT64(cx);
            bx = cx;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l[idx & MASK])[0];
            ch = ((uint64_t*) &l[idx & MASK])[1];
            lo = __umul128(idx, cl, &hi);

            al += hi;
            ah += lo;

            ah ^= tweak1_2;
            ((uint64_t*) &l[idx & MASK])[0] = al;
            ((uint64_t*) &l[idx & MASK])[1] = ah;
            ah ^= tweak1_2;

            ((uint64_t*) &l[idx & MASK])[1] ^= ((uint64_t*) &l[idx & MASK])[0];

            ah ^= ch;
            al ^= cl;
            idx = al;

            const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t*>(&l[idx & MASK]));
            const int64_t n = vgetq_lane_s64(x, 0);
            const int32_t d = vgetq_lane_s32(x, 2);
            const int64_t q = n / (d | 0x5);

            ((int64_t*) &l[idx & MASK])[0] = n ^ q;

            idx = d ^ q;
        }
#undef BYTE

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
    }
};


template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 2>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);

            bx0 = cx0;
            bx1 = cx1;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);

            bx0 = cx0;
            bx1 = cx1;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
    }

    // double
    inline static void hashPowV3(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];

        __m128i bx00 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx10 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);

        __m128i bx01 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx11 = _mm_set_epi64x(h1[9] ^ h1[11], h1[8] ^ h1[10]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];

        uint64_t division_result_xmm0 = h0[12];
        uint64_t division_result_xmm1 = h1[12];

        uint64_t sqrt_result0 = h0[13];
        uint64_t sqrt_result1 = h1[13];

        for (size_t i = 0; i < ITERATIONS; i++) {
            const __m128i ax0 = _mm_set_epi64x(ah0, al0);
            const __m128i ax1 = _mm_set_epi64x(ah1, al1);

            __m128i cx0;
            __m128i cx1;
            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], ax0);
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], ax1);
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, ax0);
                cx1 = _mm_aesenc_si128(cx1, ax1);
            }

            SHUFFLE_PHASE_1(l0, (idx0&MASK), bx00, bx10, ax0)
            SHUFFLE_PHASE_1(l1, (idx1&MASK), bx01, bx11, ax1)

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx00, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx01, cx1));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx00, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx01, cx1));

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];

            INTEGER_MATH_V2(0, cl, cx0);

            lo = __umul128(idx0, cl, &hi);

            SHUFFLE_PHASE_2(l0, (idx0&MASK), bx00, bx10, ax0, lo, hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            bx10 = bx00;
            bx00 = cx0;

            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];

            INTEGER_MATH_V2(1, cl, cx1);

            lo = __umul128(idx1, cl, &hi);

            SHUFFLE_PHASE_2(l1, (idx1&MASK), bx01, bx11, ax1, lo, hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            bx11 = bx01;
            bx01 = cx1;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
    }

    inline static void hashLiteTube(const uint8_t* __restrict__ input,
                                    size_t size,
                                    uint8_t* __restrict__ output,
                                    ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);

            bx0 = cx0;
            bx1 = cx1;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ((uint64_t*) &l0[idx0 & MASK])[1] ^= ((uint64_t*) &l0[idx0 & MASK])[0];

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ((uint64_t*) &l1[idx1 & MASK])[1] ^= ((uint64_t*) &l1[idx1 & MASK])[0];

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);

            bx0 = cx0;
            bx1 = cx1;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            const int64x2_t x0 = vld1q_s64(reinterpret_cast<const int64_t*>(&l0[idx0 & MASK]));
            const int64_t n0 = vgetq_lane_s64(x0, 0);
            const int32_t d0 = vgetq_lane_s32(x0, 2);
            const int64_t q0 = n0 / (d0 | 0x5);

            ((int64_t*) &l0[idx0 & MASK])[0] = n0 ^ q0;

            idx0 = d0 ^ q0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            const int64x2_t x1 = vld1q_s64(reinterpret_cast<const int64_t*>(&l1[idx1 & MASK]));
            const int64_t n1 = vgetq_lane_s64(x1, 0);
            const int32_t d1 = vgetq_lane_s32(x1, 2);
            const int64_t q1 = n1 / (d1 | 0x5);

            ((int64_t*) &l1[idx1 & MASK])[0] = n1 ^ q1;

            idx1 = d1 ^ q1;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                      size_t size,
                                      uint8_t* __restrict__ output,
                                      ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);

            bx0 = cx0;
            bx1 = cx1;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            const int64x2_t x0 = vld1q_s64(reinterpret_cast<const int64_t*>(&l0[idx0 & MASK]));
            const int64_t n0 = vgetq_lane_s64(x0, 0);
            const int32_t d0 = vgetq_lane_s32(x0, 2);
            const int64_t q0 = n0 / (d0 | 0x5);

            ((int64_t*) &l0[idx0 & MASK])[0] = n0 ^ q0;

            idx0 = (~d0) ^ q0;

            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            const int64x2_t x1 = vld1q_s64(reinterpret_cast<const int64_t*>(&l1[idx1 & MASK]));
            const int64_t n1 = vgetq_lane_s64(x1, 0);
            const int32_t d1 = vgetq_lane_s32(x1, 2);
            const int64_t q1 = n1 / (d1 | 0x5);

            ((int64_t*) &l1[idx1 & MASK])[0] = n1 ^ q1;

            idx1 = (~d1) ^ q1;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
    }

    inline static void hashHeavyTube(const uint8_t* __restrict__ input,
                                     size_t size,
                                     uint8_t* __restrict__ output,
                                     ScratchPad** __restrict__ scratchPad)
    {
        keccak((const uint8_t*) input, (int) size, scratchPad[0]->state, 200);
        keccak((const uint8_t*) input + size, (int) size, scratchPad[1]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];

        union alignas(16)
        {
            uint32_t k[4];
            uint64_t v64[2];
        };
        alignas(16) uint32_t x[4];

#define BYTE(p, i) ((unsigned char*)&p)[i]
        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
            __m128i cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);

            const __m128i& key0 = _mm_set_epi64x(ah0, al0);

            _mm_store_si128((__m128i*) k, key0);
            cx0 = _mm_xor_si128(cx0, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
            _mm_store_si128((__m128i*) x, cx0);

            k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^
                    saes_table[3][BYTE(x[3], 3)];
            x[0] ^= k[0];
            k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^
                    saes_table[3][BYTE(x[0], 3)];
            x[1] ^= k[1];
            k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^
                    saes_table[3][BYTE(x[1], 3)];
            x[2] ^= k[2];
            k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^
                    saes_table[3][BYTE(x[2], 3)];

            cx0 = _mm_load_si128((__m128i*) k);

            const __m128i& key1 = _mm_set_epi64x(ah1, al1);

            _mm_store_si128((__m128i*) k, key1);
            cx1 = _mm_xor_si128(cx1, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
            _mm_store_si128((__m128i*) x, cx1);

            k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^
                    saes_table[3][BYTE(x[3], 3)];
            x[0] ^= k[0];
            k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^
                    saes_table[3][BYTE(x[0], 3)];
            x[1] ^= k[1];
            k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^
                    saes_table[3][BYTE(x[1], 3)];
            x[2] ^= k[2];
            k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^
                    saes_table[3][BYTE(x[2], 3)];

            cx1 = _mm_load_si128((__m128i*) k);

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);

            bx0 = cx0;
            bx1 = cx1;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ((uint64_t*) &l0[idx0 & MASK])[1] ^= ((uint64_t*) &l0[idx0 & MASK])[0];

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            const int64x2_t x0 = vld1q_s64(reinterpret_cast<const int64_t*>(&l0[idx0 & MASK]));
            const int64_t n0 = vgetq_lane_s64(x0, 0);
            const int32_t d0 = vgetq_lane_s32(x0, 2);
            const int64_t q0 = n0 / (d0 | 0x5);

            ((int64_t*) &l0[idx0 & MASK])[0] = n0 ^ q0;

            idx0 = d0 ^ q0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ((uint64_t*) &l1[idx1 & MASK])[1] ^= ((uint64_t*) &l1[idx1 & MASK])[0];

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            const int64x2_t x1 = vld1q_s64(reinterpret_cast<const int64_t*>(&l1[idx1 & MASK]));
            const int64_t n1 = vgetq_lane_s64(x1, 0);
            const int32_t d1 = vgetq_lane_s32(x1, 2);
            const int64_t q1 = n1 / (d1 | 0x5);

            ((int64_t*) &l1[idx1 & MASK])[0] = n1 ^ q1;

            idx1 = d1 ^ q1;
        }
#undef BYTE

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 3>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(input + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[2]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ah2 ^= tweak1_2_2;
            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;
            ah2 ^= tweak1_2_2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
    }

    // triple
    inline static void hashPowV3(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];

        __m128i bx00 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx01 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx02 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

        __m128i bx10 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);
        __m128i bx11 = _mm_set_epi64x(h1[9] ^ h1[11], h1[8] ^ h1[10]);
        __m128i bx12 = _mm_set_epi64x(h2[9] ^ h2[11], h2[8] ^ h2[10]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];

        uint64_t division_result_xmm0 = h0[12];
        uint64_t division_result_xmm1 = h1[12];
        uint64_t division_result_xmm2 = h2[12];

        uint64_t sqrt_result0 =  h0[13];
        uint64_t sqrt_result1 =  h1[13];
        uint64_t sqrt_result2 =  h2[13];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;

            const __m128i ax0 = _mm_set_epi64x(ah0, al0);
            const __m128i ax1 = _mm_set_epi64x(ah1, al1);
            const __m128i ax2 = _mm_set_epi64x(ah2, al2);

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], ax0);
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], ax1);
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], ax2);
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
            }

            SHUFFLE_PHASE_1(l0, (idx0&MASK), bx00, bx10, ax0)
            SHUFFLE_PHASE_1(l1, (idx1&MASK), bx01, bx11, ax1)
            SHUFFLE_PHASE_1(l2, (idx2&MASK), bx02, bx12, ax2)

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx00, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx01, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx02, cx2));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];

            INTEGER_MATH_V2(0, cl, cx0);

            lo = __umul128(idx0, cl, &hi);

            SHUFFLE_PHASE_2(l0, (idx0&MASK), bx00, bx10, ax0, lo, hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            bx10 = bx00;
            bx00 = cx0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];

            INTEGER_MATH_V2(1, cl, cx1);

            lo = __umul128(idx1, cl, &hi);

            SHUFFLE_PHASE_2(l1, (idx1&MASK), bx01, bx11, ax1, lo, hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            bx11 = bx01;
            bx01 = cx1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];

            INTEGER_MATH_V2(2, cl, cx2);

            lo = __umul128(idx2, cl, &hi);

            SHUFFLE_PHASE_2(l2, (idx2&MASK), bx02, bx12, ax2, lo, hi)

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;

            bx12 = bx02;
            bx02 = cx2;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
    }

    inline static void hashLiteTube(const uint8_t* __restrict__ input,
                                    size_t size,
                                    uint8_t* __restrict__ output,
                                    ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(input + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[2]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ((uint64_t*) &l0[idx0 & MASK])[1] ^= ((uint64_t*) &l0[idx0 & MASK])[0];

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ((uint64_t*) &l1[idx1 & MASK])[1] ^= ((uint64_t*) &l1[idx1 & MASK])[0];

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ah2 ^= tweak1_2_2;
            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;
            ah2 ^= tweak1_2_2;

            ((uint64_t*) &l2[idx2 & MASK])[1] ^= ((uint64_t*) &l2[idx2 & MASK])[0];

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            const int64x2_t x0 = vld1q_s64(reinterpret_cast<const int64_t*>(&l0[idx0 & MASK]));
            const int64_t n0 = vgetq_lane_s64(x0, 0);
            const int32_t d0 = vgetq_lane_s32(x0, 2);
            const int64_t q0 = n0 / (d0 | 0x5);

            ((int64_t*) &l0[idx0 & MASK])[0] = n0 ^ q0;

            idx0 = d0 ^ q0;

            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            const int64x2_t x1 = vld1q_s64(reinterpret_cast<const int64_t*>(&l1[idx1 & MASK]));
            const int64_t n1 = vgetq_lane_s64(x1, 0);
            const int32_t d1 = vgetq_lane_s32(x1, 2);
            const int64_t q1 = n1 / (d1 | 0x5);

            ((int64_t*) &l1[idx1 & MASK])[0] = n1 ^ q1;

            idx1 = d1 ^ q1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;


            const int64x2_t x2 = vld1q_s64(reinterpret_cast<const int64_t*>(&l2[idx2 & MASK]));
            const int64_t n2 = vgetq_lane_s64(x2, 0);
            const int32_t d2 = vgetq_lane_s32(x2, 2);
            const int64_t q2 = n2 / (d2 | 0x5);

            ((int64_t*) &l2[idx2 & MASK])[0] = n2 ^ q2;

            idx2 = d2 ^ q2;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                      size_t size,
                                      uint8_t* __restrict__ output,
                                      ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            const int64x2_t x0 = vld1q_s64(reinterpret_cast<const int64_t*>(&l0[idx0 & MASK]));
            const int64_t n0 = vgetq_lane_s64(x0, 0);
            const int32_t d0 = vgetq_lane_s32(x0, 2);
            const int64_t q0 = n0 / (d0 | 0x5);

            ((int64_t*) &l0[idx0 & MASK])[0] = n0 ^ q0;

            idx0 = (~d0) ^ q0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            const int64x2_t x1 = vld1q_s64(reinterpret_cast<const int64_t*>(&l1[idx1 & MASK]));
            const int64_t n1 = vgetq_lane_s64(x1, 0);
            const int32_t d1 = vgetq_lane_s32(x1, 2);
            const int64_t q1 = n1 / (d1 | 0x5);

            ((int64_t*) &l1[idx1 & MASK])[0] = n1 ^ q1;

            idx1 = (~d1) ^ q1;

            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;

            const int64x2_t x2 = vld1q_s64(reinterpret_cast<const int64_t*>(&l2[idx2 & MASK]));
            const int64_t n2 = vgetq_lane_s64(x2, 0);
            const int32_t d2 = vgetq_lane_s32(x2, 2);
            const int64_t q2 = n2 / (d2 | 0x5);

            ((int64_t*) &l2[idx2 & MASK])[0] = n2 ^ q2;

            idx2 = (~d2) ^ q2;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
    }

    inline static void hashHeavyTube(const uint8_t* __restrict__ input,
                                     size_t size,
                                     uint8_t* __restrict__ output,
                                     ScratchPad** __restrict__ scratchPad)
    {
        keccak((const uint8_t*) input, (int) size, scratchPad[0]->state, 200);
        keccak((const uint8_t*) input + size, (int) size, scratchPad[1]->state, 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, scratchPad[2]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[2]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];

        union alignas(16)
        {
            uint32_t k[4];
            uint64_t v64[2];
        };
        alignas(16) uint32_t x[4];

#define BYTE(p, i) ((unsigned char*)&p)[i]
        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
            __m128i cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
            __m128i cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);

            const __m128i& key0 = _mm_set_epi64x(ah0, al0);

            _mm_store_si128((__m128i*) k, key0);
            cx0 = _mm_xor_si128(cx0, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
            _mm_store_si128((__m128i*) x, cx0);

            k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^
                    saes_table[3][BYTE(x[3], 3)];
            x[0] ^= k[0];
            k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^
                    saes_table[3][BYTE(x[0], 3)];
            x[1] ^= k[1];
            k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^
                    saes_table[3][BYTE(x[1], 3)];
            x[2] ^= k[2];
            k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^
                    saes_table[3][BYTE(x[2], 3)];

            cx0 = _mm_load_si128((__m128i*) k);

            const __m128i& key1 = _mm_set_epi64x(ah1, al1);

            _mm_store_si128((__m128i*) k, key1);
            cx1 = _mm_xor_si128(cx1, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
            _mm_store_si128((__m128i*) x, cx1);

            k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^
                    saes_table[3][BYTE(x[3], 3)];
            x[0] ^= k[0];
            k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^
                    saes_table[3][BYTE(x[0], 3)];
            x[1] ^= k[1];
            k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^
                    saes_table[3][BYTE(x[1], 3)];
            x[2] ^= k[2];
            k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^
                    saes_table[3][BYTE(x[2], 3)];

            cx1 = _mm_load_si128((__m128i*) k);

            const __m128i& key2 = _mm_set_epi64x(ah2, al2);

            _mm_store_si128((__m128i*) k, key2);
            cx2 = _mm_xor_si128(cx2, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
            _mm_store_si128((__m128i*) x, cx2);

            k[0] ^= saes_table[0][BYTE(x[0], 0)] ^ saes_table[1][BYTE(x[1], 1)] ^ saes_table[2][BYTE(x[2], 2)] ^
                    saes_table[3][BYTE(x[3], 3)];
            x[0] ^= k[0];
            k[1] ^= saes_table[0][BYTE(x[1], 0)] ^ saes_table[1][BYTE(x[2], 1)] ^ saes_table[2][BYTE(x[3], 2)] ^
                    saes_table[3][BYTE(x[0], 3)];
            x[1] ^= k[1];
            k[2] ^= saes_table[0][BYTE(x[2], 0)] ^ saes_table[1][BYTE(x[3], 1)] ^ saes_table[2][BYTE(x[0], 2)] ^
                    saes_table[3][BYTE(x[1], 3)];
            x[2] ^= k[2];
            k[3] ^= saes_table[0][BYTE(x[3], 0)] ^ saes_table[1][BYTE(x[0], 1)] ^ saes_table[2][BYTE(x[1], 2)] ^
                    saes_table[3][BYTE(x[2], 3)];

            cx2 = _mm_load_si128((__m128i*) k);

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ((uint64_t*) &l0[idx0 & MASK])[1] ^= ((uint64_t*) &l0[idx0 & MASK])[0];

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            const int64x2_t x0 = vld1q_s64(reinterpret_cast<const int64_t*>(&l0[idx0 & MASK]));
            const int64_t n0 = vgetq_lane_s64(x0, 0);
            const int32_t d0 = vgetq_lane_s32(x0, 2);
            const int64_t q0 = n0 / (d0 | 0x5);

            ((int64_t*) &l0[idx0 & MASK])[0] = n0 ^ q0;

            idx0 = d0 ^ q0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ((uint64_t*) &l1[idx1 & MASK])[1] ^= ((uint64_t*) &l1[idx1 & MASK])[0];

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            const int64x2_t x1 = vld1q_s64(reinterpret_cast<const int64_t*>(&l1[idx1 & MASK]));
            const int64_t n1 = vgetq_lane_s64(x1, 0);
            const int32_t d1 = vgetq_lane_s32(x1, 2);
            const int64_t q1 = n1 / (d1 | 0x5);

            ((int64_t*) &l1[idx1 & MASK])[0] = n1 ^ q1;

            idx1 = d1 ^ q1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ah2 ^= tweak1_2_2;
            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;
            ah2 ^= tweak1_2_2;

            ((uint64_t*) &l2[idx2 & MASK])[1] ^= ((uint64_t*) &l2[idx2 & MASK])[0];

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;

            const int64x2_t x2 = vld1q_s64(reinterpret_cast<const int64_t*>(&l2[idx2 & MASK]));
            const int64_t n2 = vgetq_lane_s64(x2, 0);
            const int32_t d2 = vgetq_lane_s32(x2, 2);
            const int64_t q2 = n2 / (d2 | 0x5);

            ((int64_t*) &l2[idx2 & MASK])[0] = n2 ^ q2;

            idx2 = d2 ^ q2;
        }
#undef BYTE

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 4>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx3 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
                cx3 = _mm_aesenc_si128(cx3, _mm_set_epi64x(ah3, al3));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx3, cx3));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;
            bx3 = cx3;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];
            lo = __umul128(idx3, cl, &hi);

            al3 += hi;
            ah3 += lo;

            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(input + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[2]->state) + 24));
        uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(input + 35 + 3 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[3]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx3 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
                cx3 = _mm_aesenc_si128(cx3, _mm_set_epi64x(ah3, al3));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx3, cx3));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;
            bx3 = cx3;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ah2 ^= tweak1_2_2;
            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;
            ah2 ^= tweak1_2_2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];
            lo = __umul128(idx3, cl, &hi);

            al3 += hi;
            ah3 += lo;

            ah3 ^= tweak1_2_3;
            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;
            ah3 ^= tweak1_2_3;

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
    }

    // quadruple
    inline static void hashPowV3(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];

        __m128i bx00 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx01 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx02 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx03 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);

        __m128i bx10 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);
        __m128i bx11 = _mm_set_epi64x(h1[9] ^ h1[11], h1[8] ^ h1[10]);
        __m128i bx12 = _mm_set_epi64x(h2[9] ^ h2[11], h2[8] ^ h2[10]);
        __m128i bx13 = _mm_set_epi64x(h3[9] ^ h3[11], h3[8] ^ h3[10]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];

        uint64_t division_result_xmm0 = h0[12];
        uint64_t division_result_xmm1 = h1[12];
        uint64_t division_result_xmm2 = h2[12];
        uint64_t division_result_xmm3 = h3[12];

        uint64_t sqrt_result0 =  h0[13];
        uint64_t sqrt_result1 =  h1[13];
        uint64_t sqrt_result2 =  h2[13];
        uint64_t sqrt_result3 =  h3[13];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;

            const __m128i ax0 = _mm_set_epi64x(ah0, al0);
            const __m128i ax1 = _mm_set_epi64x(ah1, al1);
            const __m128i ax2 = _mm_set_epi64x(ah2, al2);
            const __m128i ax3 = _mm_set_epi64x(ah3, al3);

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], ax0);
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], ax1);
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], ax2);
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], ax3);
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, ax0);
                cx1 = _mm_aesenc_si128(cx1, ax1);
                cx2 = _mm_aesenc_si128(cx2, ax2);
                cx3 = _mm_aesenc_si128(cx3, ax3);
            }

            SHUFFLE_PHASE_1(l0, (idx0&MASK), bx00, bx10, ax0)
            SHUFFLE_PHASE_1(l1, (idx1&MASK), bx01, bx11, ax1)
            SHUFFLE_PHASE_1(l2, (idx2&MASK), bx02, bx12, ax2)
            SHUFFLE_PHASE_1(l3, (idx3&MASK), bx03, bx13, ax3)

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx00, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx01, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx02, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx03, cx3));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];

            INTEGER_MATH_V2(0, cl, cx0);

            lo = __umul128(idx0, cl, &hi);

            SHUFFLE_PHASE_2(l0, (idx0&MASK), bx00, bx10, ax0, lo, hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            bx10 = bx00;
            bx00 = cx0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];

            INTEGER_MATH_V2(1, cl, cx1);

            lo = __umul128(idx1, cl, &hi);

            SHUFFLE_PHASE_2(l1, (idx1&MASK), bx01, bx11, ax1, lo, hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            bx11 = bx01;
            bx01 = cx1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];

            INTEGER_MATH_V2(2, cl, cx2);

            lo = __umul128(idx2, cl, &hi);

            SHUFFLE_PHASE_2(l2, (idx2&MASK), bx02, bx12, ax2, lo, hi);

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;

            bx12 = bx02;
            bx02 = cx2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];

            INTEGER_MATH_V2(3, cl, cx3);

            lo = __umul128(idx3, cl, &hi);

            SHUFFLE_PHASE_2(l3, (idx3&MASK), bx03, bx13, ax3, lo, hi);

            al3 += hi;
            ah3 += lo;

            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;

            bx13 = bx03;
            bx03 = cx3;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
    }

    inline static void hashLiteTube(const uint8_t* __restrict__ input,
                                    size_t size,
                                    uint8_t* __restrict__ output,
                                    ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(input + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[2]->state) + 24));
        uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(input + 35 + 3 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[3]->state) + 24));

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx3 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
                cx3 = _mm_aesenc_si128(cx3, _mm_set_epi64x(ah3, al3));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx3, cx3));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;
            bx3 = cx3;


            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ((uint64_t*) &l0[idx0 & MASK])[1] ^= ((uint64_t*) &l0[idx0 & MASK])[0];

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ((uint64_t*) &l1[idx1 & MASK])[1] ^= ((uint64_t*) &l1[idx1 & MASK])[0];

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ah2 ^= tweak1_2_2;
            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;
            ah2 ^= tweak1_2_2;

            ((uint64_t*) &l2[idx2 & MASK])[1] ^= ((uint64_t*) &l2[idx2 & MASK])[0];

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];
            lo = __umul128(idx3, cl, &hi);

            al3 += hi;
            ah3 += lo;

            ah3 ^= tweak1_2_3;
            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;
            ah3 ^= tweak1_2_3;

            ((uint64_t*) &l3[idx3 & MASK])[1] ^= ((uint64_t*) &l3[idx3 & MASK])[0];

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        // not supported
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                      size_t size,
                                      uint8_t* __restrict__ output,
                                      ScratchPad** __restrict__ scratchPad)
    {
        // not supported
    }

    inline static void hashHeavyTube(const uint8_t* __restrict__ input,
                                     size_t size,
                                     uint8_t* __restrict__ output,
                                     ScratchPad** __restrict__ scratchPad)
    {
        // not supported
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 5>
{//
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);
        keccak(input + 4 * size, (int) size, scratchPad[4]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        const uint8_t* l4 = scratchPad[4]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);
        uint64_t* h4 = reinterpret_cast<uint64_t*>(scratchPad[4]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h4, (__m128i*) l4);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t al4 = h4[0] ^h4[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];
        uint64_t ah4 = h4[1] ^h4[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx3 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);
        __m128i bx4 = _mm_set_epi64x(h4[3] ^ h4[7], h4[2] ^ h4[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];
        uint64_t idx4 = h4[0] ^h4[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;
            __m128i cx4;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
                cx4 = soft_aesenc((uint32_t*) &l4[idx4 & MASK], _mm_set_epi64x(ah4, al4));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);
                cx4 = _mm_load_si128((__m128i*) &l4[idx4 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
                cx3 = _mm_aesenc_si128(cx3, _mm_set_epi64x(ah3, al3));
                cx4 = _mm_aesenc_si128(cx4, _mm_set_epi64x(ah4, al4));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx3, cx3));
            _mm_store_si128((__m128i*) &l4[idx4 & MASK], _mm_xor_si128(bx4, cx4));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);
            idx4 = EXTRACT64(cx4);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;
            bx3 = cx3;
            bx4 = cx4;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];
            lo = __umul128(idx3, cl, &hi);

            al3 += hi;
            ah3 += lo;

            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;


            cl = ((uint64_t*) &l4[idx4 & MASK])[0];
            ch = ((uint64_t*) &l4[idx4 & MASK])[1];
            lo = __umul128(idx4, cl, &hi);

            al4 += hi;
            ah4 += lo;

            ((uint64_t*) &l4[idx4 & MASK])[0] = al4;
            ((uint64_t*) &l4[idx4 & MASK])[1] = ah4;

            ah4 ^= ch;
            al4 ^= cl;
            idx4 = al4;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l4, (__m128i*) h4);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);
        keccakf(h4, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
        extra_hashes[scratchPad[4]->state[0] & 3](scratchPad[4]->state, 200, output + 128);
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);
        keccak(input + 4 * size, (int) size, scratchPad[4]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(input + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[2]->state) + 24));
        uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(input + 35 + 3 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[3]->state) + 24));
        uint64_t tweak1_2_4 = (*reinterpret_cast<const uint64_t*>(input + 35 + 4 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[4]->state) + 24));


        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        const uint8_t* l4 = scratchPad[4]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);
        uint64_t* h4 = reinterpret_cast<uint64_t*>(scratchPad[4]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h4, (__m128i*) l4);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t al4 = h4[0] ^h4[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];
        uint64_t ah4 = h4[1] ^h4[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx3 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);
        __m128i bx4 = _mm_set_epi64x(h4[3] ^ h4[7], h4[2] ^ h4[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];
        uint64_t idx4 = h4[0] ^h4[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;
            __m128i cx4;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
                cx4 = soft_aesenc((uint32_t*) &l4[idx4 & MASK], _mm_set_epi64x(ah4, al4));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);
                cx4 = _mm_load_si128((__m128i*) &l4[idx4 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
                cx3 = _mm_aesenc_si128(cx3, _mm_set_epi64x(ah3, al3));
                cx4 = _mm_aesenc_si128(cx4, _mm_set_epi64x(ah4, al4));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx3, cx3));
            _mm_store_si128((__m128i*) &l4[idx4 & MASK], _mm_xor_si128(bx4, cx4));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l4[idx4 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l4[idx4 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);
            idx4 = EXTRACT64(cx4);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;
            bx3 = cx3;
            bx4 = cx4;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ah2 ^= tweak1_2_2;
            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;
            ah2 ^= tweak1_2_2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];
            lo = __umul128(idx3, cl, &hi);

            al3 += hi;
            ah3 += lo;

            ah3 ^= tweak1_2_3;
            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;
            ah3 ^= tweak1_2_3;

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;


            cl = ((uint64_t*) &l4[idx4 & MASK])[0];
            ch = ((uint64_t*) &l4[idx4 & MASK])[1];
            lo = __umul128(idx4, cl, &hi);

            al4 += hi;
            ah4 += lo;

            ah4 ^= tweak1_2_4;
            ((uint64_t*) &l4[idx4 & MASK])[0] = al4;
            ((uint64_t*) &l4[idx4 & MASK])[1] = ah4;
            ah4 ^= tweak1_2_4;

            ah4 ^= ch;
            al4 ^= cl;
            idx4 = al4;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l4, (__m128i*) h4);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);
        keccakf(h4, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
        extra_hashes[scratchPad[4]->state[0] & 3](scratchPad[4]->state, 200, output + 128);
    }

    // quintuple
    inline static void hashPowV3(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);
        keccak(input + 4 * size, (int) size, scratchPad[4]->state, 200);

        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        const uint8_t* l4 = scratchPad[4]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);
        uint64_t* h4 = reinterpret_cast<uint64_t*>(scratchPad[4]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h4, (__m128i*) l4);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t al4 = h4[0] ^h4[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];
        uint64_t ah4 = h4[1] ^h4[5];

        __m128i bx00 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx01 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx02 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx03 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);
        __m128i bx04 = _mm_set_epi64x(h4[3] ^ h4[7], h4[2] ^ h4[6]);

        __m128i bx10 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);
        __m128i bx11 = _mm_set_epi64x(h1[9] ^ h1[11], h1[8] ^ h1[10]);
        __m128i bx12 = _mm_set_epi64x(h2[9] ^ h2[11], h2[8] ^ h2[10]);
        __m128i bx13 = _mm_set_epi64x(h3[9] ^ h3[11], h3[8] ^ h3[10]);
        __m128i bx14 = _mm_set_epi64x(h4[9] ^ h4[11], h4[8] ^ h4[10]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];
        uint64_t idx4 = h4[0] ^h4[4];

        uint64_t division_result_xmm0 = h0[12];
        uint64_t division_result_xmm1 = h1[12];
        uint64_t division_result_xmm2 = h2[12];
        uint64_t division_result_xmm3 = h3[12];
        uint64_t division_result_xmm4 = h4[12];

        uint64_t sqrt_result0 =  h0[13];
        uint64_t sqrt_result1 =  h1[13];
        uint64_t sqrt_result2 =  h2[13];
        uint64_t sqrt_result3 =  h3[13];
        uint64_t sqrt_result4 =  h4[13];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;
            __m128i cx4;

            const __m128i ax0 = _mm_set_epi64x(ah0, al0);
            const __m128i ax1 = _mm_set_epi64x(ah1, al1);
            const __m128i ax2 = _mm_set_epi64x(ah2, al2);
            const __m128i ax3 = _mm_set_epi64x(ah3, al3);
            const __m128i ax4 = _mm_set_epi64x(ah4, al4);

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], ax0);
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], ax1);
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], ax2);
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], ax3);
                cx4 = soft_aesenc((uint32_t*) &l4[idx4 & MASK], ax4);
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);
                cx4 = _mm_load_si128((__m128i*) &l4[idx4 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, ax0);
                cx1 = _mm_aesenc_si128(cx1, ax1);
                cx2 = _mm_aesenc_si128(cx2, ax2);
                cx3 = _mm_aesenc_si128(cx3, ax3);
                cx4 = _mm_aesenc_si128(cx4, ax4);
            }

            SHUFFLE_PHASE_1(l0, (idx0&MASK), bx00, bx10, ax0)
            SHUFFLE_PHASE_1(l1, (idx1&MASK), bx01, bx11, ax1)
            SHUFFLE_PHASE_1(l2, (idx2&MASK), bx02, bx12, ax2)
            SHUFFLE_PHASE_1(l3, (idx3&MASK), bx03, bx13, ax3)
            SHUFFLE_PHASE_1(l4, (idx4&MASK), bx04, bx14, ax4)

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx00, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx01, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx02, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx03, cx3));
            _mm_store_si128((__m128i*) &l4[idx4 & MASK], _mm_xor_si128(bx04, cx4));

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);
            idx4 = EXTRACT64(cx4);

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];

            INTEGER_MATH_V2(0, cl, cx0);

            lo = __umul128(idx0, cl, &hi);

            SHUFFLE_PHASE_2(l0, (idx0&MASK), bx00, bx10, ax0, lo, hi);

            al0 += hi;
            ah0 += lo;

            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;

            bx10 = bx00;
            bx00 = cx0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];

            INTEGER_MATH_V2(1, cl, cx1);

            lo = __umul128(idx1, cl, &hi);

            SHUFFLE_PHASE_2(l1, (idx1&MASK), bx01, bx11, ax1, lo, hi);

            al1 += hi;
            ah1 += lo;

            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;

            bx11 = bx01;
            bx01 = cx1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];

            INTEGER_MATH_V2(2, cl, cx2);

            lo = __umul128(idx2, cl, &hi);

            SHUFFLE_PHASE_2(l2, (idx2&MASK), bx02, bx12, ax2, lo, hi);

            al2 += hi;
            ah2 += lo;

            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;

            bx12 = bx02;
            bx02 = cx2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];

            INTEGER_MATH_V2(3, cl, cx3);

            lo = __umul128(idx3, cl, &hi);

            SHUFFLE_PHASE_2(l3, (idx3&MASK), bx03, bx13, ax3, lo, hi);

            al3 += hi;
            ah3 += lo;

            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;

            bx13 = bx03;
            bx03 = cx3;


            cl = ((uint64_t*) &l4[idx4 & MASK])[0];
            ch = ((uint64_t*) &l4[idx4 & MASK])[1];

            INTEGER_MATH_V2(4, cl, cx4);

            lo = __umul128(idx4, cl, &hi);

            SHUFFLE_PHASE_2(l4, (idx4&MASK), bx04, bx14, ax4, lo, hi);

            al4 += hi;
            ah4 += lo;

            ((uint64_t*) &l4[idx4 & MASK])[0] = al4;
            ((uint64_t*) &l4[idx4 & MASK])[1] = ah4;

            ah4 ^= ch;
            al4 ^= cl;
            idx4 = al4;

            bx14 = bx04;
            bx04 = cx4;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l4, (__m128i*) h4);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);
        keccakf(h4, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
        extra_hashes[scratchPad[4]->state[0] & 3](scratchPad[4]->state, 200, output + 128);
    }

    inline static void hashLiteTube(const uint8_t* __restrict__ input,
                                    size_t size,
                                    uint8_t* __restrict__ output,
                                    ScratchPad** __restrict__ scratchPad)
    {
        keccak(input, (int) size, scratchPad[0]->state, 200);
        keccak(input + size, (int) size, scratchPad[1]->state, 200);
        keccak(input + 2 * size, (int) size, scratchPad[2]->state, 200);
        keccak(input + 3 * size, (int) size, scratchPad[3]->state, 200);
        keccak(input + 4 * size, (int) size, scratchPad[4]->state, 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(input + 35) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[0]->state) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(input + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[1]->state) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(input + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[2]->state) + 24));
        uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(input + 35 + 3 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[3]->state) + 24));
        uint64_t tweak1_2_4 = (*reinterpret_cast<const uint64_t*>(input + 35 + 4 * size) ^
                               *(reinterpret_cast<const uint64_t*>(scratchPad[4]->state) + 24));


        const uint8_t* l0 = scratchPad[0]->memory;
        const uint8_t* l1 = scratchPad[1]->memory;
        const uint8_t* l2 = scratchPad[2]->memory;
        const uint8_t* l3 = scratchPad[3]->memory;
        const uint8_t* l4 = scratchPad[4]->memory;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(scratchPad[0]->state);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(scratchPad[1]->state);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(scratchPad[2]->state);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(scratchPad[3]->state);
        uint64_t* h4 = reinterpret_cast<uint64_t*>(scratchPad[4]->state);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h3, (__m128i*) l3);
        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h4, (__m128i*) l4);

        uint64_t al0 = h0[0] ^h0[4];
        uint64_t al1 = h1[0] ^h1[4];
        uint64_t al2 = h2[0] ^h2[4];
        uint64_t al3 = h3[0] ^h3[4];
        uint64_t al4 = h4[0] ^h4[4];
        uint64_t ah0 = h0[1] ^h0[5];
        uint64_t ah1 = h1[1] ^h1[5];
        uint64_t ah2 = h2[1] ^h2[5];
        uint64_t ah3 = h3[1] ^h3[5];
        uint64_t ah4 = h4[1] ^h4[5];

        __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
        __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
        __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);
        __m128i bx3 = _mm_set_epi64x(h3[3] ^ h3[7], h3[2] ^ h3[6]);
        __m128i bx4 = _mm_set_epi64x(h4[3] ^ h4[7], h4[2] ^ h4[6]);

        uint64_t idx0 = h0[0] ^h0[4];
        uint64_t idx1 = h1[0] ^h1[4];
        uint64_t idx2 = h2[0] ^h2[4];
        uint64_t idx3 = h3[0] ^h3[4];
        uint64_t idx4 = h4[0] ^h4[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx0;
            __m128i cx1;
            __m128i cx2;
            __m128i cx3;
            __m128i cx4;

            if (SOFT_AES) {
                cx0 = soft_aesenc((uint32_t*) &l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*) &l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*) &l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*) &l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
                cx4 = soft_aesenc((uint32_t*) &l4[idx4 & MASK], _mm_set_epi64x(ah4, al4));
            } else {
                cx0 = _mm_load_si128((__m128i*) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i*) &l1[idx1 & MASK]);
                cx2 = _mm_load_si128((__m128i*) &l2[idx2 & MASK]);
                cx3 = _mm_load_si128((__m128i*) &l3[idx3 & MASK]);
                cx4 = _mm_load_si128((__m128i*) &l4[idx4 & MASK]);

                cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
                cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
                cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
                cx3 = _mm_aesenc_si128(cx3, _mm_set_epi64x(ah3, al3));
                cx4 = _mm_aesenc_si128(cx4, _mm_set_epi64x(ah4, al4));
            }

            _mm_store_si128((__m128i*) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
            _mm_store_si128((__m128i*) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
            _mm_store_si128((__m128i*) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));
            _mm_store_si128((__m128i*) &l3[idx3 & MASK], _mm_xor_si128(bx3, cx3));
            _mm_store_si128((__m128i*) &l4[idx4 & MASK], _mm_xor_si128(bx4, cx4));

            static const uint32_t table = 0x75310;
            uint8_t tmp = reinterpret_cast<const uint8_t*>(&l0[idx0 & MASK])[11];
            uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l4[idx4 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*) (&l4[idx4 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

            idx0 = EXTRACT64(cx0);
            idx1 = EXTRACT64(cx1);
            idx2 = EXTRACT64(cx2);
            idx3 = EXTRACT64(cx3);
            idx4 = EXTRACT64(cx4);

            bx0 = cx0;
            bx1 = cx1;
            bx2 = cx2;
            bx3 = cx3;
            bx4 = cx4;

            uint64_t hi, lo, cl, ch;
            cl = ((uint64_t*) &l0[idx0 & MASK])[0];
            ch = ((uint64_t*) &l0[idx0 & MASK])[1];
            lo = __umul128(idx0, cl, &hi);

            al0 += hi;
            ah0 += lo;

            ah0 ^= tweak1_2_0;
            ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
            ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;
            ah0 ^= tweak1_2_0;

            ((uint64_t*) &l0[idx0 & MASK])[1] ^= ((uint64_t*) &l0[idx0 & MASK])[0];

            ah0 ^= ch;
            al0 ^= cl;
            idx0 = al0;


            cl = ((uint64_t*) &l1[idx1 & MASK])[0];
            ch = ((uint64_t*) &l1[idx1 & MASK])[1];
            lo = __umul128(idx1, cl, &hi);

            al1 += hi;
            ah1 += lo;

            ah1 ^= tweak1_2_1;
            ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
            ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;
            ah1 ^= tweak1_2_1;

            ((uint64_t*) &l1[idx1 & MASK])[1] ^= ((uint64_t*) &l1[idx1 & MASK])[0];

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;


            cl = ((uint64_t*) &l2[idx2 & MASK])[0];
            ch = ((uint64_t*) &l2[idx2 & MASK])[1];
            lo = __umul128(idx2, cl, &hi);

            al2 += hi;
            ah2 += lo;

            ah2 ^= tweak1_2_2;
            ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
            ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;
            ah2 ^= tweak1_2_2;

            ((uint64_t*) &l2[idx2 & MASK])[1] ^= ((uint64_t*) &l2[idx2 & MASK])[0];

            ah2 ^= ch;
            al2 ^= cl;
            idx2 = al2;


            cl = ((uint64_t*) &l3[idx3 & MASK])[0];
            ch = ((uint64_t*) &l3[idx3 & MASK])[1];
            lo = __umul128(idx3, cl, &hi);

            al3 += hi;
            ah3 += lo;

            ah3 ^= tweak1_2_3;
            ((uint64_t*) &l3[idx3 & MASK])[0] = al3;
            ((uint64_t*) &l3[idx3 & MASK])[1] = ah3;
            ah3 ^= tweak1_2_3;

            ((uint64_t*) &l3[idx3 & MASK])[1] ^= ((uint64_t*) &l3[idx3 & MASK])[0];

            ah3 ^= ch;
            al3 ^= cl;
            idx3 = al3;


            cl = ((uint64_t*) &l4[idx4 & MASK])[0];
            ch = ((uint64_t*) &l4[idx4 & MASK])[1];
            lo = __umul128(idx4, cl, &hi);

            al4 += hi;
            ah4 += lo;

            ah4 ^= tweak1_2_4;
            ((uint64_t*) &l4[idx4 & MASK])[0] = al4;
            ((uint64_t*) &l4[idx4 & MASK])[1] = ah4;
            ah4 ^= tweak1_2_4;

            ((uint64_t*) &l4[idx4 & MASK])[1] ^= ((uint64_t*) &l4[idx4 & MASK])[0];

            ah4 ^= ch;
            al4 ^= cl;
            idx4 = al4;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l3, (__m128i*) h3);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l4, (__m128i*) h4);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);
        keccakf(h3, 24);
        keccakf(h4, 24);

        extra_hashes[scratchPad[0]->state[0] & 3](scratchPad[0]->state, 200, output);
        extra_hashes[scratchPad[1]->state[0] & 3](scratchPad[1]->state, 200, output + 32);
        extra_hashes[scratchPad[2]->state[0] & 3](scratchPad[2]->state, 200, output + 64);
        extra_hashes[scratchPad[3]->state[0] & 3](scratchPad[3]->state, 200, output + 96);
        extra_hashes[scratchPad[4]->state[0] & 3](scratchPad[4]->state, 200, output + 128);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 ScratchPad** __restrict__ scratchPad)
    {
        // not supported
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                      size_t size,
                                      uint8_t* __restrict__ output,
                                      ScratchPad** __restrict__ scratchPad)
    {
        // not supported
    }

    inline static void hashHeavyTube(const uint8_t* __restrict__ input,
                                      size_t size,
                                      uint8_t* __restrict__ output,
                                      ScratchPad** __restrict__ scratchPad)
    {
        // not supported
    }
};

#endif /* __CRYPTONIGHT_ARM_H__ */
