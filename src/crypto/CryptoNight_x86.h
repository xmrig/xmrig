/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
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

#ifndef __CRYPTONIGHT_X86_H__
#define __CRYPTONIGHT_X86_H__


#ifdef __GNUC__
#   include <x86intrin.h>
#else
#   include <intrin.h>
#   define __restrict__ __restrict
#endif


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

static inline void do_blake_hash(const uint8_t *input, size_t len, uint8_t *output) {
    blake256_hash(output, input, len);
}


static inline void do_groestl_hash(const uint8_t *input, size_t len, uint8_t *output) {
    groestl(input, len * 8, output);
}


static inline void do_jh_hash(const uint8_t *input, size_t len, uint8_t *output) {
    jh_hash(32 * 8, input, 8 * len, output);
}


static inline void do_skein_hash(const uint8_t *input, size_t len, uint8_t *output) {
    xmr_skein(input, output);
}


void (* const extra_hashes[4])(const uint8_t *, size_t, uint8_t *) = {do_blake_hash, do_groestl_hash, do_jh_hash, do_skein_hash};

#if defined(__x86_64__) || defined(_M_AMD64)
#   define EXTRACT64(X) _mm_cvtsi128_si64(X)

#   ifdef __GNUC__

static inline uint64_t __umul128(uint64_t a, uint64_t b, uint64_t* hi)
{
    unsigned __int128 r = (unsigned __int128) a * (unsigned __int128) b;
    *hi = r >> 64;
    return (uint64_t) r;
}

#   else
#define __umul128 _umul128
#   endif
#elif defined(__i386__) || defined(_M_IX86)
#   define HI32(X) \
    _mm_srli_si128((X), 4)


#   define EXTRACT64(X) \
    ((uint64_t)(uint32_t)_mm_cvtsi128_si32(X) | \
    ((uint64_t)(uint32_t)_mm_cvtsi128_si32(HI32(X)) << 32))

static inline uint64_t __umul128(uint64_t multiplier, uint64_t multiplicand, uint64_t *product_hi) {
    // multiplier   = ab = a * 2^32 + b
    // multiplicand = cd = c * 2^32 + d
    // ab * cd = a * c * 2^64 + (a * d + b * c) * 2^32 + b * d
    uint64_t a = multiplier >> 32;
    uint64_t b = multiplier & 0xFFFFFFFF;
    uint64_t c = multiplicand >> 32;
    uint64_t d = multiplicand & 0xFFFFFFFF;

    //uint64_t ac = a * c;
    uint64_t ad = a * d;
    //uint64_t bc = b * c;
    uint64_t bd = b * d;

    uint64_t adbc = ad + (b * c);
    uint64_t adbc_carry = adbc < ad ? 1 : 0;

    // multiplier * multiplicand = product_hi * 2^64 + product_lo
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
static inline void aes_genkey_sub(__m128i* xout0, __m128i* xout2)
{
    __m128i xout1 = _mm_aeskeygenassist_si128(*xout2, rcon);
    xout1 = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
    *xout0 = sl_xor(*xout0);
    *xout0 = _mm_xor_si128(*xout0, xout1);
    xout1 = _mm_aeskeygenassist_si128(*xout0, 0x00);
    xout1 = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
    *xout2 = sl_xor(*xout2);
    *xout2 = _mm_xor_si128(*xout2, xout1);
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

    SOFT_AES ? soft_aes_genkey_sub<0x01>(&xout0, &xout2) : aes_genkey_sub<0x01>(&xout0, &xout2);
    *k2 = xout0;
    *k3 = xout2;

    SOFT_AES ? soft_aes_genkey_sub<0x02>(&xout0, &xout2) : aes_genkey_sub<0x02>(&xout0, &xout2);
    *k4 = xout0;
    *k5 = xout2;

    SOFT_AES ? soft_aes_genkey_sub<0x04>(&xout0, &xout2) : aes_genkey_sub<0x04>(&xout0, &xout2);
    *k6 = xout0;
    *k7 = xout2;

    SOFT_AES ? soft_aes_genkey_sub<0x08>(&xout0, &xout2) : aes_genkey_sub<0x08>(&xout0, &xout2);
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
}

inline void mix_and_propagate(__m128i& x0, __m128i& x1, __m128i& x2, __m128i& x3, __m128i& x4, __m128i& x5, __m128i& x6, __m128i& x7)
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
        aes_round<SOFT_AES>(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);

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
        aes_round<SOFT_AES>(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);

        mix_and_propagate(xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
    }

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8) {
        aes_round<SOFT_AES>(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);

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

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
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

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);

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

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);

        mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
    }

    for (size_t i = 0; i < 16; i++) {
        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);

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
                            cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, ctx->state[hashBlock], 200);
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = ctx->memory + hashBlock * MEM;
            h[hashBlock] = reinterpret_cast<uint64_t*>(ctx->state[hashBlock]);

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
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK], _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
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
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[ctx->state[hashBlock][0] & 3](ctx->state[hashBlock], 200,
                                                       output + hashBlock * 32);
        }
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];
        uint64_t tweak1_2[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, ctx->state[hashBlock], 200);
            tweak1_2[hashBlock] = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + hashBlock * size) ^
                    *(reinterpret_cast<const uint64_t*>(ctx->state[hashBlock]) + 24));
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = ctx->memory + hashBlock * MEM;
            h[hashBlock] = reinterpret_cast<uint64_t*>(ctx->state[hashBlock]);

            cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h[hashBlock], (__m128i*) l[hashBlock]);

            al[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
            ah[hashBlock] = h[hashBlock][1] ^ h[hashBlock][5];
            bx[hashBlock] = _mm_set_epi64x(h[hashBlock][3] ^ h[hashBlock][7], h[hashBlock][2] ^ h[hashBlock][6]);
            idx[hashBlock] = h[hashBlock][0] ^ h[hashBlock][4];
        }

        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
                __m128i cx;

                if (SOFT_AES) {
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK], _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                } else {
                    cx = _mm_load_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                }

                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK], _mm_xor_si128(bx[hashBlock], cx));

                const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[hashBlock][idx[hashBlock] & MASK])[11];
                static const uint32_t table = 0x75310;
                const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
                ((uint8_t*)(&l[hashBlock][idx[hashBlock] & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[ctx->state[hashBlock][0] & 3](ctx->state[hashBlock], 200,
                                                       output + hashBlock * 32);
        }
    }

    inline static void hashLiteIpbc(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];
        uint64_t tweak1_2[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, ctx->state[hashBlock], 200);
            tweak1_2[hashBlock] = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + hashBlock * size) ^
                                   *(reinterpret_cast<const uint64_t*>(ctx->state[hashBlock]) + 24));
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = ctx->memory + hashBlock * MEM;
            h[hashBlock] = reinterpret_cast<uint64_t*>(ctx->state[hashBlock]);

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
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK], _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                } else {
                    cx = _mm_load_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK]);
                    cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
                }

                _mm_store_si128((__m128i*) &l[hashBlock][idx[hashBlock] & MASK],
                                _mm_xor_si128(bx[hashBlock], cx));

                const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[hashBlock][idx[hashBlock] & MASK])[11];
                static const uint32_t table = 0x75310;
                const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
                ((uint8_t*)(&l[hashBlock][idx[hashBlock] & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

                ((uint64_t*)&l[hashBlock][idx[hashBlock] & MASK])[1] ^= ((uint64_t*)&l[hashBlock][idx[hashBlock] & MASK])[0];

                ah[hashBlock] ^= ch;
                al[hashBlock] ^= cl;
                idx[hashBlock] = al[hashBlock];
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[ctx->state[hashBlock][0] & 3](ctx->state[hashBlock], 200,
                                                       output + hashBlock * 32);
        }
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, ctx->state[hashBlock], 200);
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = ctx->memory + hashBlock * MEM;
            h[hashBlock] = reinterpret_cast<uint64_t*>(ctx->state[hashBlock]);

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
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK], _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
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

                int64_t n  = ((int64_t*)&l[hashBlock][idx[hashBlock] & MASK])[0];
                int32_t d  = ((int32_t*)&l[hashBlock][idx[hashBlock] & MASK])[2];
                int64_t q = n / (d | 0x5);

                ((int64_t*)&l[hashBlock][idx[hashBlock] & MASK])[0] = n ^ q;
                idx[hashBlock] = d ^ q;
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[ctx->state[hashBlock][0] & 3](ctx->state[hashBlock], 200,
                                                       output + hashBlock * 32);
        }
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l[NUM_HASH_BLOCKS];
        uint64_t* h[NUM_HASH_BLOCKS];
        uint64_t al[NUM_HASH_BLOCKS];
        uint64_t ah[NUM_HASH_BLOCKS];
        __m128i bx[NUM_HASH_BLOCKS];
        uint64_t idx[NUM_HASH_BLOCKS];

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            keccak(static_cast<const uint8_t*>(input) + hashBlock * size, (int) size, ctx->state[hashBlock], 200);
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            l[hashBlock] = ctx->memory + hashBlock * MEM;
            h[hashBlock] = reinterpret_cast<uint64_t*>(ctx->state[hashBlock]);

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
                    cx = soft_aesenc((uint32_t*) &l[hashBlock][idx[hashBlock] & MASK], _mm_set_epi64x(ah[hashBlock], al[hashBlock]));
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

                int64_t n  = ((int64_t*)&l[hashBlock][idx[hashBlock] & MASK])[0];
                int32_t d  = ((int32_t*)&l[hashBlock][idx[hashBlock] & MASK])[2];
                int64_t q = n / (d | 0x5);

                ((int64_t*)&l[hashBlock][idx[hashBlock] & MASK])[0] = n ^ q;
                idx[hashBlock] = (~d) ^ q;
            }
        }

        for (size_t hashBlock = 0; hashBlock < NUM_HASH_BLOCKS; ++hashBlock) {
            cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l[hashBlock], (__m128i*) h[hashBlock]);
            keccakf(h[hashBlock], 24);
            extra_hashes[ctx->state[hashBlock][0] & 3](ctx->state[hashBlock], 200,
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
                            cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, ctx->state[0], 200);

        l = ctx->memory;
        h = reinterpret_cast<uint64_t*>(ctx->state[0]);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*)&l[idx & MASK], _mm_set_epi64x(ah, al));
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
        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, ctx->state[0], 200);

        uint64_t tweak1_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
        l = ctx->memory;
        h = reinterpret_cast<uint64_t*>(ctx->state[0]);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*)&l[idx & MASK], _mm_set_epi64x(ah, al));
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah, al));
            }

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[idx & MASK])[11];
            static const uint32_t table = 0x75310;
            const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l[idx & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
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
        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
    }

    inline static void hashLiteIpbc(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, ctx->state[0], 200);

        uint64_t tweak1_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
        l = ctx->memory;
        h = reinterpret_cast<uint64_t*>(ctx->state[0]);

        cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*)&l[idx & MASK], _mm_set_epi64x(ah, al));
            } else {
                cx = _mm_load_si128((__m128i*) &l[idx & MASK]);
                cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah, al));
            }

            _mm_store_si128((__m128i*) &l[idx & MASK], _mm_xor_si128(bx, cx));
            const uint8_t tmp = reinterpret_cast<const uint8_t*>(&l[idx & MASK])[11];
            static const uint32_t table = 0x75310;
            const uint8_t index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l[idx & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
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

            ((uint64_t*)&l[idx & MASK])[1] ^= ((uint64_t*)&l[idx & MASK])[0];

            ah ^= ch;
            al ^= cl;
            idx = al;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, ctx->state[0], 200);

        l = ctx->memory;
        h = reinterpret_cast<uint64_t*>(ctx->state[0]);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*)&l[idx & MASK], _mm_set_epi64x(ah, al));
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

            int64_t n  = ((int64_t*)&l[idx & MASK])[0];
            int32_t d  = ((int32_t*)&l[idx & MASK])[2];
            int64_t q = n / (d | 0x5);

            ((int64_t*)&l[idx & MASK])[0] = n ^ q;
            idx = d ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        const uint8_t* l;
        uint64_t* h;
        uint64_t al;
        uint64_t ah;
        __m128i bx;
        uint64_t idx;

        keccak(static_cast<const uint8_t*>(input), (int) size, ctx->state[0], 200);

        l = ctx->memory;
        h = reinterpret_cast<uint64_t*>(ctx->state[0]);

        cn_explode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) h, (__m128i*) l);

        al = h[0] ^ h[4];
        ah = h[1] ^ h[5];
        bx = _mm_set_epi64x(h[3] ^ h[7], h[2] ^ h[6]);
        idx = h[0] ^ h[4];

        for (size_t i = 0; i < ITERATIONS; i++) {
            __m128i cx;

            if (SOFT_AES) {
                cx = soft_aesenc((uint32_t*)&l[idx & MASK], _mm_set_epi64x(ah, al));
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

            int64_t n  = ((int64_t*)&l[idx & MASK])[0];
            int32_t d  = ((int32_t*)&l[idx & MASK])[2];
            int64_t q = n / (d | 0x5);

            ((int64_t*)&l[idx & MASK])[0] = n ^ q;
            idx = (~d) ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l, (__m128i*) h);
        keccakf(h, 24);
        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 2>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                          size_t size,
                          uint8_t* __restrict__ output,
                          cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
    }

    inline static void hashPowV2(const uint8_t* __restrict__ input,
                              size_t size,
                              uint8_t* __restrict__ output,
                              cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
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
            ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
    }

    inline static void hashLiteIpbc(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
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
            ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

            ((uint64_t*)&l0[idx0 & MASK])[1] ^= ((uint64_t*)&l0[idx0 & MASK])[0];

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

            ((uint64_t*)&l1[idx1 & MASK])[1] ^= ((uint64_t*)&l1[idx1 & MASK])[0];

            ah1 ^= ch;
            al1 ^= cl;
            idx1 = al1;
        }

        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
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

            int64_t n  = ((int64_t*)&l0[idx0 & MASK])[0];
            int32_t d  = ((int32_t*)&l0[idx0 & MASK])[2];
            int64_t q = n / (d | 0x5);

            ((int64_t*)&l0[idx0 & MASK])[0] = n ^ q;
            idx0 = d ^ q;


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

            n  = ((int64_t*)&l1[idx1 & MASK])[0];
            d  = ((int32_t*)&l1[idx1 & MASK])[2];
            q = n / (d | 0x5);

            ((int64_t*)&l1[idx1 & MASK])[0] = n ^ q;
            idx1 = d ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
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

            int64_t n  = ((int64_t*)&l0[idx0 & MASK])[0];
            int32_t d  = ((int32_t*)&l0[idx0 & MASK])[2];
            int64_t q = n / (d | 0x5);

            ((int64_t*)&l0[idx0 & MASK])[0] = n ^ q;
            idx0 = (~d) ^ q;


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

            n  = ((int64_t*)&l1[idx1 & MASK])[0];
            d  = ((int32_t*)&l1[idx1 & MASK])[2];
            q = n / (d | 0x5);

            ((int64_t*)&l1[idx1 & MASK])[0] = n ^ q;
            idx1 = (~d) ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

        keccakf(h0, 24);
        keccakf(h1, 24);

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 3>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
    }

  inline static void hashPowV2(const uint8_t* __restrict__ input,
                          size_t size,
                          uint8_t* __restrict__ output,
                          cryptonight_ctx* __restrict__ ctx)
  {
      keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
      keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
      keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);

      uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
      uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));
      uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 2 * size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[2]) + 24));

      const uint8_t* l0 = ctx->memory;
      const uint8_t* l1 = ctx->memory + MEM;
      const uint8_t* l2 = ctx->memory + 2 * MEM;
      uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
      uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
      uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);

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
              cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
              cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
              cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
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
          ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

      extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
      extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
      extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
  }

    inline static void hashLiteIpbc(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[2]) + 24));

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
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
            ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

            ((uint64_t*)&l0[idx0 & MASK])[1] ^= ((uint64_t*)&l0[idx0 & MASK])[0];

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

            ((uint64_t*)&l1[idx1 & MASK])[1] ^= ((uint64_t*)&l1[idx1 & MASK])[0];

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

            ((uint64_t*)&l2[idx2 & MASK])[1] ^= ((uint64_t*)&l2[idx2 & MASK])[0];

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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
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

            int64_t n  = ((int64_t*)&l0[idx0 & MASK])[0];
            int32_t d  = ((int32_t*)&l0[idx0 & MASK])[2];
            int64_t q = n / (d | 0x5);

            ((int64_t*)&l0[idx0 & MASK])[0] = n ^ q;
            idx0 = d ^ q;


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

            n  = ((int64_t*)&l1[idx1 & MASK])[0];
            d  = ((int32_t*)&l1[idx1 & MASK])[2];
            q = n / (d | 0x5);

            ((int64_t*)&l1[idx1 & MASK])[0] = n ^ q;
            idx1 = d ^ q;


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

            n  = ((int64_t*)&l2[idx2 & MASK])[0];
            d  = ((int32_t*)&l2[idx2 & MASK])[2];
            q = n / (d | 0x5);

            ((int64_t*)&l2[idx2 & MASK])[0] = n ^ q;
            idx2 = d ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
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

            int64_t n  = ((int64_t*)&l0[idx0 & MASK])[0];
            int32_t d  = ((int32_t*)&l0[idx0 & MASK])[2];
            int64_t q = n / (d | 0x5);

            ((int64_t*)&l0[idx0 & MASK])[0] = n ^ q;
            idx0 = (~d) ^ q;


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

            n  = ((int64_t*)&l1[idx1 & MASK])[0];
            d  = ((int32_t*)&l1[idx1 & MASK])[2];
            q = n / (d | 0x5);

            ((int64_t*)&l1[idx1 & MASK])[0] = n ^ q;
            idx1 = (~d) ^ q;


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

            n  = ((int64_t*)&l2[idx2 & MASK])[0];
            d  = ((int32_t*)&l2[idx2 & MASK])[2];
            q = n / (d | 0x5);

            ((int64_t*)&l2[idx2 & MASK])[0] = n ^ q;
            idx2 = (~d) ^ q;
        }

        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
        cn_implode_scratchpad_heavy<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

        keccakf(h0, 24);
        keccakf(h1, 24);
        keccakf(h2, 24);

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 4>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);
        keccak((const uint8_t*) input + 3 * size, (int) size, ctx->state[3], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        const uint8_t* l3 = ctx->memory + 3 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(ctx->state[3]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*)&l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
        extra_hashes[ctx->state[3][0] & 3](ctx->state[3], 200, output + 96);
    }

  inline static void hashPowV2(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
  {
      keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
      keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
      keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);
      keccak((const uint8_t*) input + 3 * size, (int) size, ctx->state[3], 200);

      uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
      uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));
      uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 2 * size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[2]) + 24));
      uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 3 * size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[3]) + 24));

      const uint8_t* l0 = ctx->memory;
      const uint8_t* l1 = ctx->memory + MEM;
      const uint8_t* l2 = ctx->memory + 2 * MEM;
      const uint8_t* l3 = ctx->memory + 3 * MEM;
      uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
      uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
      uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);
      uint64_t* h3 = reinterpret_cast<uint64_t*>(ctx->state[3]);

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
              cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
              cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
              cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
              cx3 = soft_aesenc((uint32_t*)&l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
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
          ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

      extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
      extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
      extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
      extra_hashes[ctx->state[3][0] & 3](ctx->state[3], 200, output + 96);
  }

    inline static void hashLiteIpbc(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);
        keccak((const uint8_t*) input + 3 * size, (int) size, ctx->state[3], 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[2]) + 24));
        uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 3 * size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[3]) + 24));

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        const uint8_t* l3 = ctx->memory + 3 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(ctx->state[3]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*)&l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
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
            ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

            ((uint64_t*)&l0[idx0 & MASK])[1] ^= ((uint64_t*)&l0[idx0 & MASK])[0];

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

            ((uint64_t*)&l1[idx1 & MASK])[1] ^= ((uint64_t*)&l1[idx1 & MASK])[0];

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

            ((uint64_t*)&l2[idx2 & MASK])[1] ^= ((uint64_t*)&l2[idx2 & MASK])[0];

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

            ((uint64_t*)&l3[idx3 & MASK])[1] ^= ((uint64_t*)&l3[idx3 & MASK])[0];

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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
        extra_hashes[ctx->state[3][0] & 3](ctx->state[3], 200, output + 96);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        // not supported
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        // not supported
    }
};

template<size_t ITERATIONS, size_t INDEX_SHIFT, size_t MEM, size_t MASK, bool SOFT_AES>
class CryptoNightMultiHash<ITERATIONS, INDEX_SHIFT, MEM, MASK, SOFT_AES, 5>
{
public:
    inline static void hash(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);
        keccak((const uint8_t*) input + 3 * size, (int) size, ctx->state[3], 200);
        keccak((const uint8_t*) input + 4 * size, (int) size, ctx->state[4], 200);

        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        const uint8_t* l3 = ctx->memory + 3 * MEM;
        const uint8_t* l4 = ctx->memory + 4 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(ctx->state[3]);
        uint64_t* h4 = reinterpret_cast<uint64_t*>(ctx->state[4]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*)&l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
                cx4 = soft_aesenc((uint32_t*)&l4[idx4 & MASK], _mm_set_epi64x(ah4, al4));
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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
        extra_hashes[ctx->state[3][0] & 3](ctx->state[3], 200, output + 96);
        extra_hashes[ctx->state[4][0] & 3](ctx->state[4], 200, output + 128);
    }

  inline static void hashPowV2(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
  {
      keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
      keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
      keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);
      keccak((const uint8_t*) input + 3 * size, (int) size, ctx->state[3], 200);
      keccak((const uint8_t*) input + 4 * size, (int) size, ctx->state[4], 200);

      uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
      uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));
      uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 2 * size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[2]) + 24));
      uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 3 * size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[3]) + 24));
      uint64_t tweak1_2_4 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 4 * size) ^
                             *(reinterpret_cast<const uint64_t*>(ctx->state[4]) + 24));


      const uint8_t* l0 = ctx->memory;
      const uint8_t* l1 = ctx->memory + MEM;
      const uint8_t* l2 = ctx->memory + 2 * MEM;
      const uint8_t* l3 = ctx->memory + 3 * MEM;
      const uint8_t* l4 = ctx->memory + 4 * MEM;
      uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
      uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
      uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);
      uint64_t* h3 = reinterpret_cast<uint64_t*>(ctx->state[3]);
      uint64_t* h4 = reinterpret_cast<uint64_t*>(ctx->state[4]);

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
              cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
              cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
              cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
              cx3 = soft_aesenc((uint32_t*)&l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
              cx4 = soft_aesenc((uint32_t*)&l4[idx4 & MASK], _mm_set_epi64x(ah4, al4));
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
          ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
          tmp = reinterpret_cast<const uint8_t*>(&l4[idx4 & MASK])[11];
          index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
          ((uint8_t*)(&l4[idx4 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

      extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
      extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
      extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
      extra_hashes[ctx->state[3][0] & 3](ctx->state[3], 200, output + 96);
      extra_hashes[ctx->state[4][0] & 3](ctx->state[4], 200, output + 128);
  }

    inline static void hashLiteIpbc(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        keccak((const uint8_t*) input, (int) size, ctx->state[0], 200);
        keccak((const uint8_t*) input + size, (int) size, ctx->state[1], 200);
        keccak((const uint8_t*) input + 2 * size, (int) size, ctx->state[2], 200);
        keccak((const uint8_t*) input + 3 * size, (int) size, ctx->state[3], 200);
        keccak((const uint8_t*) input + 4 * size, (int) size, ctx->state[4], 200);

        uint64_t tweak1_2_0 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[0]) + 24));
        uint64_t tweak1_2_1 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[1]) + 24));
        uint64_t tweak1_2_2 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 2 * size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[2]) + 24));
        uint64_t tweak1_2_3 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 3 * size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[3]) + 24));
        uint64_t tweak1_2_4 = (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + 4 * size) ^
                               *(reinterpret_cast<const uint64_t*>(ctx->state[4]) + 24));


        const uint8_t* l0 = ctx->memory;
        const uint8_t* l1 = ctx->memory + MEM;
        const uint8_t* l2 = ctx->memory + 2 * MEM;
        const uint8_t* l3 = ctx->memory + 3 * MEM;
        const uint8_t* l4 = ctx->memory + 4 * MEM;
        uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state[0]);
        uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state[1]);
        uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state[2]);
        uint64_t* h3 = reinterpret_cast<uint64_t*>(ctx->state[3]);
        uint64_t* h4 = reinterpret_cast<uint64_t*>(ctx->state[4]);

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
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], _mm_set_epi64x(ah0, al0));
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], _mm_set_epi64x(ah1, al1));
                cx2 = soft_aesenc((uint32_t*)&l2[idx2 & MASK], _mm_set_epi64x(ah2, al2));
                cx3 = soft_aesenc((uint32_t*)&l3[idx3 & MASK], _mm_set_epi64x(ah3, al3));
                cx4 = soft_aesenc((uint32_t*)&l4[idx4 & MASK], _mm_set_epi64x(ah4, al4));
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
            ((uint8_t*)(&l0[idx0 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l1[idx1 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l1[idx1 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l2[idx2 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l2[idx2 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l3[idx3 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l3[idx3 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);
            tmp = reinterpret_cast<const uint8_t*>(&l4[idx4 & MASK])[11];
            index = (((tmp >> INDEX_SHIFT) & 6) | (tmp & 1)) << 1;
            ((uint8_t*)(&l4[idx4 & MASK]))[11] = tmp ^ ((table >> index) & 0x30);

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

            ((uint64_t*)&l0[idx0 & MASK])[1] ^= ((uint64_t*)&l0[idx0 & MASK])[0];

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

            ((uint64_t*)&l1[idx1 & MASK])[1] ^= ((uint64_t*)&l1[idx1 & MASK])[0];

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

            ((uint64_t*)&l2[idx2 & MASK])[1] ^= ((uint64_t*)&l2[idx2 & MASK])[0];

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

            ((uint64_t*)&l3[idx3 & MASK])[1] ^= ((uint64_t*)&l3[idx3 & MASK])[0];

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

            ((uint64_t*)&l4[idx4 & MASK])[1] ^= ((uint64_t*)&l4[idx4 & MASK])[0];

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

        extra_hashes[ctx->state[0][0] & 3](ctx->state[0], 200, output);
        extra_hashes[ctx->state[1][0] & 3](ctx->state[1], 200, output + 32);
        extra_hashes[ctx->state[2][0] & 3](ctx->state[2], 200, output + 64);
        extra_hashes[ctx->state[3][0] & 3](ctx->state[3], 200, output + 96);
        extra_hashes[ctx->state[4][0] & 3](ctx->state[4], 200, output + 128);
    }

    inline static void hashHeavy(const uint8_t* __restrict__ input,
                            size_t size,
                            uint8_t* __restrict__ output,
                            cryptonight_ctx* __restrict__ ctx)
    {
        // not supported
    }

    inline static void hashHeavyHaven(const uint8_t* __restrict__ input,
                                 size_t size,
                                 uint8_t* __restrict__ output,
                                 cryptonight_ctx* __restrict__ ctx)
    {
        // not supported
    }
};
#endif /* __CRYPTONIGHT_X86_H__ */
