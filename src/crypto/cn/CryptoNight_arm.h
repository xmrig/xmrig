/* XMRig
 * Copyright 2010      Jeff Garzik  <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler       <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones  <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466     <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee    <jayddee246@gmail.com>
 * Copyright 2016      Imran Yusuff <https://github.com/imranyusuff>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CRYPTONIGHT_ARM_H
#define XMRIG_CRYPTONIGHT_ARM_H


#include "base/crypto/keccak.h"
#include "crypto/cn/CnAlgo.h"
#include "crypto/cn/CryptoNight_monero.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/cn/soft_aes.h"
#include "crypto/common/portable/mm_malloc.h"


extern "C"
{
#include "crypto/cn/c_groestl.h"
#include "crypto/cn/c_blake256.h"
#include "crypto/cn/c_jh.h"
#include "crypto/cn/c_skein.h"
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
    alignas(16) const __m128i zero = { 0 };
    return zero;
}
#endif


/* this one was not implemented yet so here it is */
static inline __attribute__((always_inline)) uint64_t _mm_cvtsi128_si64(__m128i a)
{
    return vgetq_lane_u64(a, 0);
}


#if defined (__arm64__) || defined (__aarch64__)
static inline uint64_t __umul128(uint64_t a, uint64_t b, uint64_t* hi)
{
    unsigned __int128 r = (unsigned __int128) a * (unsigned __int128) b;
    *hi = r >> 64;
    return (uint64_t) r;
}
#else
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
static inline void soft_aes_genkey_sub(__m128i* xout0, __m128i* xout2)
{
    __m128i xout1 = soft_aeskeygenassist<rcon>(*xout2);
    xout1  = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
    *xout0 = sl_xor(*xout0);
    *xout0 = _mm_xor_si128(*xout0, xout1);
    xout1  = soft_aeskeygenassist<0x00>(*xout0);
    xout1  = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
    *xout2 = sl_xor(*xout2);
    *xout2 = _mm_xor_si128(*xout2, xout1);
}


template<bool SOFT_AES>
static inline void aes_genkey(const __m128i* memory, __m128i* k0, __m128i* k1, __m128i* k2, __m128i* k3, __m128i* k4, __m128i* k5, __m128i* k6, __m128i* k7, __m128i* k8, __m128i* k9)
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
static inline void aes_round(__m128i key, __m128i* x0, __m128i* x1, __m128i* x2, __m128i* x3, __m128i* x4, __m128i* x5, __m128i* x6, __m128i* x7)
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


namespace xmrig {


template<Algorithm::Id ALGO, bool SOFT_AES>
static inline void cn_explode_scratchpad(const __m128i *input, __m128i *output)
{
    constexpr CnAlgo<ALGO> props;

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

    if (props.isHeavy()) {
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
    }

    for (size_t i = 0; i < props.memory() / sizeof(__m128i); i += 8) {
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


template<Algorithm::Id ALGO, bool SOFT_AES>
static inline void cn_implode_scratchpad(const __m128i *input, __m128i *output)
{
    constexpr CnAlgo<ALGO> props;

#   ifdef XMRIG_ALGO_CN_GPU
    constexpr bool IS_HEAVY = props.isHeavy() || ALGO == Algorithm::CN_GPU;
#   else
    constexpr bool IS_HEAVY = props.isHeavy();
#   endif

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

    for (size_t i = 0; i < props.memory() / sizeof(__m128i); i += 8) {
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

        if (IS_HEAVY) {
            mix_and_propagate(xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
        }
    }

    if (IS_HEAVY) {
        for (size_t i = 0; i < props.memory() / sizeof(__m128i); i += 8) {
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


} /* namespace xmrig */


static inline __m128i aes_round_tweak_div(const __m128i &in, const __m128i &key)
{
    alignas(16) uint32_t k[4];
    alignas(16) uint32_t x[4];

    _mm_store_si128((__m128i*) k, key);
    _mm_store_si128((__m128i*) x, _mm_xor_si128(in, _mm_set_epi64x(0xffffffffffffffff, 0xffffffffffffffff)));

    #define BYTE(p, i) ((unsigned char*)&x[p])[i]
    k[0] ^= saes_table[0][BYTE(0, 0)] ^ saes_table[1][BYTE(1, 1)] ^ saes_table[2][BYTE(2, 2)] ^ saes_table[3][BYTE(3, 3)];
    x[0] ^= k[0];
    k[1] ^= saes_table[0][BYTE(1, 0)] ^ saes_table[1][BYTE(2, 1)] ^ saes_table[2][BYTE(3, 2)] ^ saes_table[3][BYTE(0, 3)];
    x[1] ^= k[1];
    k[2] ^= saes_table[0][BYTE(2, 0)] ^ saes_table[1][BYTE(3, 1)] ^ saes_table[2][BYTE(0, 2)] ^ saes_table[3][BYTE(1, 3)];
    x[2] ^= k[2];
    k[3] ^= saes_table[0][BYTE(3, 0)] ^ saes_table[1][BYTE(0, 1)] ^ saes_table[2][BYTE(1, 2)] ^ saes_table[3][BYTE(2, 3)];
    #undef BYTE

    return _mm_load_si128((__m128i*)k);
}


namespace xmrig {


template<Algorithm::Id ALGO>
static inline void cryptonight_monero_tweak(const uint8_t* l, uint64_t idx, __m128i ax0, __m128i bx0, __m128i bx1, __m128i& cx)
{
    constexpr CnAlgo<ALGO> props;

    uint64_t* mem_out = (uint64_t*)&l[idx];

    if (props.base() == Algorithm::CN_2) {
        VARIANT2_SHUFFLE(l, idx, ax0, bx0, bx1, cx, (ALGO == Algorithm::CN_RWZ ? 1 : 0));
        _mm_store_si128((__m128i *)mem_out, _mm_xor_si128(bx0, cx));
    } else {
        __m128i tmp = _mm_xor_si128(bx0, cx);
        mem_out[0] = _mm_cvtsi128_si64(tmp);

        uint64_t vh = vgetq_lane_u64(tmp, 1);

        uint8_t x = vh >> 24;
        static const uint16_t table = 0x7531;
        const uint8_t index = (((x >> (3)) & 6) | (x & 1)) << 1;
        vh ^= ((table >> index) & 0x3) << 28;

        mem_out[1] = vh;
    }
}


static inline void cryptonight_conceal_tweak(__m128i& cx, __m128& conc_var)
{
    __m128 r = _mm_add_ps(_mm_cvtepi32_ps(cx), conc_var);
    r = _mm_mul_ps(r, _mm_mul_ps(r, r));
    r = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x807FFFFF)), r);
    r = _mm_or_ps(_mm_castsi128_ps(_mm_set1_epi32(0x40000000)), r);

    __m128 c_old = conc_var;
    conc_var = _mm_add_ps(conc_var, r);

    c_old = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x807FFFFF)), c_old);
    c_old = _mm_or_ps(_mm_castsi128_ps(_mm_set1_epi32(0x40000000)), c_old);

    __m128 nc = _mm_mul_ps(c_old, _mm_set1_ps(536870880.0f));
    cx = _mm_xor_si128(cx, _mm_cvttps_epi32(nc));
}


template<Algorithm::Id ALGO, bool SOFT_AES>
inline void cryptonight_single_hash(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, cryptonight_ctx **__restrict__ ctx, uint64_t height)
{
    constexpr CnAlgo<ALGO> props;
    constexpr size_t MASK        = props.mask();
    constexpr Algorithm::Id BASE = props.base();

#   ifdef XMRIG_ALGO_CN_HEAVY
    constexpr bool IS_CN_HEAVY_TUBE = ALGO == Algorithm::CN_HEAVY_TUBE;
#   else
    constexpr bool IS_CN_HEAVY_TUBE = false;
#   endif

    if (BASE == Algorithm::CN_1 && size < 43) {
        memset(output, 0, 32);
        return;
    }

    keccak(input, size, ctx[0]->state);
    cn_explode_scratchpad<ALGO, SOFT_AES>(reinterpret_cast<const __m128i *>(ctx[0]->state), reinterpret_cast<__m128i *>(ctx[0]->memory));

    uint8_t* l0 = ctx[0]->memory;
    uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx[0]->state);

    VARIANT1_INIT(0);
    VARIANT2_INIT(0);
    VARIANT4_RANDOM_MATH_INIT(0);

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    __m128i bx0  = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx1  = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);

    __m128 conc_var;
    if (ALGO == Algorithm::CN_CCX) {
        conc_var = _mm_setzero_ps();
    }

    uint64_t idx0 = al0;

    for (size_t i = 0; i < props.iterations(); i++) {
        __m128i cx;
        if (IS_CN_HEAVY_TUBE || !SOFT_AES) {
            cx = _mm_load_si128(reinterpret_cast<const __m128i *>(&l0[idx0 & MASK]));
            if (ALGO == Algorithm::CN_CCX) {
                cryptonight_conceal_tweak(cx, conc_var);
            }
        }

        const __m128i ax0 = _mm_set_epi64x(ah0, al0);
        if (IS_CN_HEAVY_TUBE) {
            cx = aes_round_tweak_div(cx, ax0);
        }
        else if (SOFT_AES) {
            if (ALGO == Algorithm::CN_CCX) {
                cx = _mm_load_si128(reinterpret_cast<const __m128i*>(&l0[idx0 & MASK]));
                cryptonight_conceal_tweak(cx, conc_var);
                cx = soft_aesenc((uint32_t*)&cx, ax0);
            }
            else {
                cx = soft_aesenc((uint32_t*)&l0[idx0 & MASK], ax0);
            }
        }
        else {
            cx = _mm_aesenc_si128(cx, ax0);
        }

        if (BASE == Algorithm::CN_1 || BASE == Algorithm::CN_2) {
            cryptonight_monero_tweak<ALGO>(l0, idx0 & MASK, ax0, bx0, bx1, cx);
        } else {
            _mm_store_si128((__m128i *)&l0[idx0 & MASK], _mm_xor_si128(bx0, cx));
        }

        idx0 = _mm_cvtsi128_si64(cx);

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & MASK])[0];
        ch = ((uint64_t*) &l0[idx0 & MASK])[1];

        if (BASE == Algorithm::CN_2) {
            if (props.isR()) {
                VARIANT4_RANDOM_MATH(0, al0, ah0, cl, bx0, bx1);
                if (ALGO == Algorithm::CN_R) {
                    al0 ^= r0[2] | ((uint64_t)(r0[3]) << 32);
                    ah0 ^= r0[0] | ((uint64_t)(r0[1]) << 32);
                }
            } else {
                VARIANT2_INTEGER_MATH(0, cl, cx);
            }
        }

        lo = __umul128(idx0, cl, &hi);

        if (BASE == Algorithm::CN_2) {
            if (ALGO == Algorithm::CN_R) {
                VARIANT2_SHUFFLE(l0, idx0 & MASK, ax0, bx0, bx1, cx, 0);
            } else {
                VARIANT2_SHUFFLE2(l0, idx0 & MASK, ax0, bx0, bx1, hi, lo, (ALGO == Algorithm::CN_RWZ ? 1 : 0));
            }
        }

        al0 += hi;
        ah0 += lo;

        ((uint64_t*)&l0[idx0 & MASK])[0] = al0;

        if (IS_CN_HEAVY_TUBE || ALGO == Algorithm::CN_RTO) {
            ((uint64_t*)&l0[idx0 & MASK])[1] = ah0 ^ tweak1_2_0 ^ al0;
        } else if (BASE == Algorithm::CN_1) {
            ((uint64_t*)&l0[idx0 & MASK])[1] = ah0 ^ tweak1_2_0;
        } else {
            ((uint64_t*)&l0[idx0 & MASK])[1] = ah0;
        }

        al0 ^= cl;
        ah0 ^= ch;
        idx0 = al0;

#       ifdef XMRIG_ALGO_CN_HEAVY
        if (props.isHeavy()) {
            const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t *>(&l0[idx0 & MASK]));
            const int64_t n   = vgetq_lane_s64(x, 0);
            const int32_t d   = vgetq_lane_s32(x, 2);
            const int64_t q   = n / (d | 0x5);

            ((int64_t*)&l0[idx0 & MASK])[0] = n ^ q;

            if (ALGO == Algorithm::CN_HEAVY_XHV) {
                idx0 = (~d) ^ q;
            }
            else {
                idx0 = d ^ q;
            }
        }
#       endif

        if (BASE == Algorithm::CN_2) {
            bx1 = bx0;
        }

        bx0 = cx;
    }

    cn_implode_scratchpad<ALGO, SOFT_AES>(reinterpret_cast<const __m128i *>(ctx[0]->memory), reinterpret_cast<__m128i *>(ctx[0]->state));
    keccakf(h0, 24);
    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
}


} /* namespace xmrig */


#ifdef XMRIG_ALGO_CN_GPU
template<size_t ITER, uint32_t MASK>
void cn_gpu_inner_arm(const uint8_t *spad, uint8_t *lpad);


namespace xmrig {


template<size_t MEM>
void cn_explode_scratchpad_gpu(const uint8_t *input, uint8_t *output)
{
    constexpr size_t hash_size = 200; // 25x8 bytes
    alignas(16) uint64_t hash[25];

    for (uint64_t i = 0; i < MEM / 512; i++) {
        memcpy(hash, input, hash_size);
        hash[0] ^= i;

        xmrig::keccakf(hash, 24);
        memcpy(output, hash, 160);
        output += 160;

        xmrig::keccakf(hash, 24);
        memcpy(output, hash, 176);
        output += 176;

        xmrig::keccakf(hash, 24);
        memcpy(output, hash, 176);
        output += 176;
    }
}


template<xmrig::Algorithm::Id ALGO, bool SOFT_AES>
inline void cryptonight_single_hash_gpu(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, cryptonight_ctx **__restrict__ ctx, uint64_t height)
{
    constexpr CnAlgo<ALGO> props;

    keccak(input, size, ctx[0]->state);
    cn_explode_scratchpad_gpu<props.memory()>(ctx[0]->state, ctx[0]->memory);

    fesetround(FE_TONEAREST);

    cn_gpu_inner_arm<props.iterations(), props.mask()>(ctx[0]->state, ctx[0]->memory);

    cn_implode_scratchpad<ALGO, SOFT_AES>(reinterpret_cast<const __m128i *>(ctx[0]->memory), reinterpret_cast<__m128i *>(ctx[0]->state));
    keccakf(reinterpret_cast<uint64_t*>(ctx[0]->state), 24);
    memcpy(output, ctx[0]->state, 32);
}

} /* namespace xmrig */
#endif


namespace xmrig {


template<Algorithm::Id ALGO, bool SOFT_AES>
inline void cryptonight_double_hash(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, struct cryptonight_ctx **__restrict__ ctx, uint64_t height)
{
    constexpr CnAlgo<ALGO> props;
    constexpr size_t MASK        = props.mask();
    constexpr Algorithm::Id BASE = props.base();

#   ifdef XMRIG_ALGO_CN_HEAVY
    constexpr bool IS_CN_HEAVY_TUBE = ALGO == Algorithm::CN_HEAVY_TUBE;
#   else
    constexpr bool IS_CN_HEAVY_TUBE = false;
#   endif

    if (BASE == Algorithm::CN_1 && size < 43) {
        memset(output, 0, 64);
        return;
    }

    keccak(input,        size, ctx[0]->state);
    keccak(input + size, size, ctx[1]->state);

    uint8_t *l0  = ctx[0]->memory;
    uint8_t *l1  = ctx[1]->memory;
    uint64_t *h0 = reinterpret_cast<uint64_t*>(ctx[0]->state);
    uint64_t *h1 = reinterpret_cast<uint64_t*>(ctx[1]->state);

    VARIANT1_INIT(0);
    VARIANT1_INIT(1);
    VARIANT2_INIT(0);
    VARIANT2_INIT(1);
    VARIANT4_RANDOM_MATH_INIT(0);
    VARIANT4_RANDOM_MATH_INIT(1);

    cn_explode_scratchpad<ALGO, SOFT_AES>(reinterpret_cast<const __m128i *>(h0), reinterpret_cast<__m128i *>(l0));
    cn_explode_scratchpad<ALGO, SOFT_AES>(reinterpret_cast<const __m128i *>(h1), reinterpret_cast<__m128i *>(l1));

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t al1 = h1[0] ^ h1[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    uint64_t ah1 = h1[1] ^ h1[5];

    __m128i bx00 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx01 = _mm_set_epi64x(h0[9] ^ h0[11], h0[8] ^ h0[10]);
    __m128i bx10 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
    __m128i bx11 = _mm_set_epi64x(h1[9] ^ h1[11], h1[8] ^ h1[10]);

    __m128 conc_var0, conc_var1;
    if (ALGO == Algorithm::CN_CCX) {
        conc_var0 = _mm_setzero_ps();
        conc_var1 = _mm_setzero_ps();
    }

    uint64_t idx0 = al0;
    uint64_t idx1 = al1;

    for (size_t i = 0; i < props.iterations(); i++) {
        __m128i cx0, cx1;
        if (IS_CN_HEAVY_TUBE || !SOFT_AES) {
            cx0 = _mm_load_si128((__m128i *) &l0[idx0 & MASK]);
            cx1 = _mm_load_si128((__m128i *) &l1[idx1 & MASK]);
            if (ALGO == Algorithm::CN_CCX) {
                cryptonight_conceal_tweak(cx0, conc_var0);
                cryptonight_conceal_tweak(cx1, conc_var1);
            }
        }

        const __m128i ax0 = _mm_set_epi64x(ah0, al0);
        const __m128i ax1 = _mm_set_epi64x(ah1, al1);
        if (IS_CN_HEAVY_TUBE) {
            cx0 = aes_round_tweak_div(cx0, ax0);
            cx1 = aes_round_tweak_div(cx1, ax1);
        }
        else if (SOFT_AES) {
            if (ALGO == Algorithm::CN_CCX) {
                cx0 = _mm_load_si128((__m128i *) &l0[idx0 & MASK]);
                cx1 = _mm_load_si128((__m128i *) &l1[idx1 & MASK]);
                cryptonight_conceal_tweak(cx0, conc_var0);
                cryptonight_conceal_tweak(cx1, conc_var1);
                cx0 = soft_aesenc((uint32_t*)&cx0, ax0);
                cx1 = soft_aesenc((uint32_t*)&cx1, ax1);
            }
            else {
                cx0 = soft_aesenc((uint32_t*)&l0[idx0 & MASK], ax0);
                cx1 = soft_aesenc((uint32_t*)&l1[idx1 & MASK], ax1);
            }
        }
        else {
            cx0 = _mm_aesenc_si128(cx0, ax0);
            cx1 = _mm_aesenc_si128(cx1, ax1);
        }

        if (BASE == Algorithm::CN_1 || BASE == Algorithm::CN_2) {
            cryptonight_monero_tweak<ALGO>(l0, idx0 & MASK, ax0, bx00, bx01, cx0);
            cryptonight_monero_tweak<ALGO>(l1, idx1 & MASK, ax1, bx10, bx11, cx1);
        } else {
            _mm_store_si128((__m128i *) &l0[idx0 & MASK], _mm_xor_si128(bx00, cx0));
            _mm_store_si128((__m128i *) &l1[idx1 & MASK], _mm_xor_si128(bx10, cx1));
        }

        idx0 = _mm_cvtsi128_si64(cx0);
        idx1 = _mm_cvtsi128_si64(cx1);

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & MASK])[0];
        ch = ((uint64_t*) &l0[idx0 & MASK])[1];

        if (BASE == Algorithm::CN_2) {
            if (props.isR()) {
                VARIANT4_RANDOM_MATH(0, al0, ah0, cl, bx00, bx01);
                if (ALGO == Algorithm::CN_R) {
                    al0 ^= r0[2] | ((uint64_t)(r0[3]) << 32);
                    ah0 ^= r0[0] | ((uint64_t)(r0[1]) << 32);
                }
            } else {
                VARIANT2_INTEGER_MATH(0, cl, cx0);
            }
        }

        lo = __umul128(idx0, cl, &hi);

        if (BASE == Algorithm::CN_2) {
            if (ALGO == Algorithm::CN_R) {
                VARIANT2_SHUFFLE(l0, idx0 & MASK, ax0, bx00, bx01, cx0, 0);
            } else {
                VARIANT2_SHUFFLE2(l0, idx0 & MASK, ax0, bx00, bx01, hi, lo, (ALGO == Algorithm::CN_RWZ ? 1 : 0));
            }
        }

        al0 += hi;
        ah0 += lo;

        ((uint64_t*)&l0[idx0 & MASK])[0] = al0;

        if (IS_CN_HEAVY_TUBE || ALGO == Algorithm::CN_RTO) {
            ((uint64_t*)&l0[idx0 & MASK])[1] = ah0 ^ tweak1_2_0 ^ al0;
        } else if (BASE == Algorithm::CN_1) {
            ((uint64_t*)&l0[idx0 & MASK])[1] = ah0 ^ tweak1_2_0;
        } else {
            ((uint64_t*)&l0[idx0 & MASK])[1] = ah0;
        }

        al0 ^= cl;
        ah0 ^= ch;
        idx0 = al0;

#       ifdef XMRIG_ALGO_CN_HEAVY
        if (props.isHeavy()) {
            const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t *>(&l0[idx0 & MASK]));
            const int64_t n   = vgetq_lane_s64(x, 0);
            const int32_t d   = vgetq_lane_s32(x, 2);
            const int64_t q   = n / (d | 0x5);

            ((int64_t*)&l0[idx0 & MASK])[0] = n ^ q;

            if (ALGO == Algorithm::CN_HEAVY_XHV) {
                idx0 = (~d) ^ q;
            }
            else {
                idx0 = d ^ q;
            }
        }
#       endif

        cl = ((uint64_t*) &l1[idx1 & MASK])[0];
        ch = ((uint64_t*) &l1[idx1 & MASK])[1];

        if (BASE == Algorithm::CN_2) {
            if (props.isR()) {
                VARIANT4_RANDOM_MATH(1, al1, ah1, cl, bx10, bx11);
                if (ALGO == Algorithm::CN_R) {
                    al1 ^= r1[2] | ((uint64_t)(r1[3]) << 32);
                    ah1 ^= r1[0] | ((uint64_t)(r1[1]) << 32);
                }
            } else {
                VARIANT2_INTEGER_MATH(1, cl, cx1);
            }
        }

        lo = __umul128(idx1, cl, &hi);

        if (BASE == Algorithm::CN_2) {
            if (ALGO == Algorithm::CN_R) {
                VARIANT2_SHUFFLE(l1, idx1 & MASK, ax1, bx10, bx11, cx1, 0);
            } else {
                VARIANT2_SHUFFLE2(l1, idx1 & MASK, ax1, bx10, bx11, hi, lo, (ALGO == Algorithm::CN_RWZ ? 1 : 0));
            }
        }

        al1 += hi;
        ah1 += lo;

        ((uint64_t*)&l1[idx1 & MASK])[0] = al1;

        if (IS_CN_HEAVY_TUBE || ALGO == Algorithm::CN_RTO) {
            ((uint64_t*)&l1[idx1 & MASK])[1] = ah1 ^ tweak1_2_1 ^ al1;
        } else if (BASE == Algorithm::CN_1) {
            ((uint64_t*)&l1[idx1 & MASK])[1] = ah1 ^ tweak1_2_1;
        } else {
            ((uint64_t*)&l1[idx1 & MASK])[1] = ah1;
        }

        al1 ^= cl;
        ah1 ^= ch;
        idx1 = al1;

#       ifdef XMRIG_ALGO_CN_HEAVY
        if (props.isHeavy()) {
            const int64x2_t x = vld1q_s64(reinterpret_cast<const int64_t *>(&l1[idx1 & MASK]));
            const int64_t n   = vgetq_lane_s64(x, 0);
            const int32_t d   = vgetq_lane_s32(x, 2);
            const int64_t q   = n / (d | 0x5);

            ((int64_t*)&l1[idx1 & MASK])[0] = n ^ q;

            if (ALGO == Algorithm::CN_HEAVY_XHV) {
                idx1 = (~d) ^ q;
            }
            else {
                idx1 = d ^ q;
            }
        }
#       endif

        if (BASE == Algorithm::CN_2) {
            bx01 = bx00;
            bx11 = bx10;
        }

        bx00 = cx0;
        bx10 = cx1;
    }

    cn_implode_scratchpad<ALGO, SOFT_AES>(reinterpret_cast<const __m128i *>(l0), reinterpret_cast<__m128i *>(h0));
    cn_implode_scratchpad<ALGO, SOFT_AES>(reinterpret_cast<const __m128i *>(l1), reinterpret_cast<__m128i *>(h1));

    keccakf(h0, 24);
    keccakf(h1, 24);

    extra_hashes[ctx[0]->state[0] & 3](ctx[0]->state, 200, output);
    extra_hashes[ctx[1]->state[0] & 3](ctx[1]->state, 200, output + 32);
}


template<Algorithm::Id ALGO, bool SOFT_AES>
inline void cryptonight_triple_hash(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, struct cryptonight_ctx **__restrict__ ctx, uint64_t height)
{
}


template<Algorithm::Id ALGO, bool SOFT_AES>
inline void cryptonight_quad_hash(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, struct cryptonight_ctx **__restrict__ ctx, uint64_t height)
{
}


template<Algorithm::Id ALGO, bool SOFT_AES>
inline void cryptonight_penta_hash(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, struct cryptonight_ctx **__restrict__ ctx, uint64_t height)
{
}


} /* namespace xmrig */


#endif /* XMRIG_CRYPTONIGHT_ARM_H */
