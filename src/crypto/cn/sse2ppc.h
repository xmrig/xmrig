/* XMRig
 * Copyright (c) 2018-2025 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2025 XMRig       <support@xmrig.com>
 * Copyright 2026 PalindromicBreadLoaf <https://github.com/palindromicbreadloaf>
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

// Scalar SSE2/SSSE3/NEON compatibility for PowerPC.
//
// Backs the CryptoNight family (CryptoNight_arm.h + CryptoNight_monero.h +
// soft_aes.h) on PowerPC.
//
// Lane semantics follow x86 SSE2

#ifndef XMRIG_SSE2PPC_H
#define XMRIG_SSE2PPC_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#   define XMRIG_PPC_LE 1
#else
#   define XMRIG_PPC_LE 0
#endif


static inline uint64_t xm_ppc_bswap64(uint64_t x) {
#if XMRIG_PPC_LE
    return x;
#else
    return __builtin_bswap64(x);
#endif
}


static inline uint32_t xm_ppc_bswap32(uint32_t x) {
#if XMRIG_PPC_LE
    return x;
#else
    return __builtin_bswap32(x);
#endif
}


// u64[0] is SSE2 lane 0
typedef union alignas(16) __m128i {
    uint64_t u64[2];
    int64_t  i64[2];
    uint32_t u32[4];
    int32_t  i32[4];
    uint16_t u16[8];
    uint8_t  u8[16];
} __m128i;


typedef __m128i uint64x2_t;
typedef __m128i int64x2_t;


// f32[0] is the low SSE lane
typedef union alignas(16) __m128 {
    float    f32[4];
    uint32_t u32[4];
} __m128;


// 32-bit rotate-right
#ifndef _rotr
static inline uint32_t _rotr(uint32_t x, int n)
{
    return (x >> (n & 31)) | (x << ((32 - n) & 31));
}
#endif

static inline __m128i _mm_load_si128(const __m128i *p)
{
    __m128i r;
    uint64_t a, b;
    memcpy(&a, (const uint8_t *)p,     8);
    memcpy(&b, (const uint8_t *)p + 8, 8);
    r.u64[0] = xm_ppc_bswap64(a);
    r.u64[1] = xm_ppc_bswap64(b);
    return r;
}


static inline void _mm_store_si128(__m128i *p, __m128i v)
{
    const uint64_t a = xm_ppc_bswap64(v.u64[0]);
    const uint64_t b = xm_ppc_bswap64(v.u64[1]);
    memcpy((uint8_t *)p,     &a, 8);
    memcpy((uint8_t *)p + 8, &b, 8);
}

static inline __m128i _mm_set_epi64x(int64_t hi, int64_t lo)
{
    __m128i r;
    r.u64[0] = (uint64_t)lo;
    r.u64[1] = (uint64_t)hi;
    return r;
}


static inline __m128i _mm_set_epi32(int32_t d, int32_t c, int32_t b, int32_t a)
{
    __m128i r;
    r.u64[0] = ((uint64_t)(uint32_t)a)       | (((uint64_t)(uint32_t)b) << 32);
    r.u64[1] = ((uint64_t)(uint32_t)c)       | (((uint64_t)(uint32_t)d) << 32);
    return r;
}


static inline __m128i _mm_set1_epi32(int32_t a)
{
    return _mm_set_epi32(a, a, a, a);
}


static inline __m128i _mm_cvtsi64_si128(int64_t a)
{
    __m128i r;
    r.u64[0] = (uint64_t)a;
    r.u64[1] = 0;
    return r;
}

static inline int64_t _mm_cvtsi128_si64(__m128i a)
{
    return (int64_t)a.u64[0];
}


static inline int32_t _mm_cvtsi128_si32(__m128i a)
{
    return (int32_t)(uint32_t)a.u64[0];
}

static inline __m128i _mm_xor_si128(__m128i a, __m128i b)
{
    __m128i r;
    r.u64[0] = a.u64[0] ^ b.u64[0];
    r.u64[1] = a.u64[1] ^ b.u64[1];
    return r;
}


static inline __m128i _mm_add_epi64(__m128i a, __m128i b)
{
    __m128i r;
    r.u64[0] = a.u64[0] + b.u64[0];
    r.u64[1] = a.u64[1] + b.u64[1];
    return r;
}

// _mm_slli_si128(a, imm): shift left by `imm` bytes, fill with zero
static inline __m128i _mm_slli_si128_impl(__m128i a, const int imm)
{
    if (imm >= 16) {
        a.u64[0] = 0;
        a.u64[1] = 0;
        return a;
    }
    if (imm <= 0) {
        return a;
    }
    if (imm >= 8) {
        a.u64[1] = a.u64[0] << ((imm - 8) * 8);
        a.u64[0] = 0;
        return a;
    }
    const unsigned s = imm * 8;
    a.u64[1] = (a.u64[1] << s) | (a.u64[0] >> (64 - s));
    a.u64[0] = a.u64[0] << s;
    return a;
}
#define _mm_slli_si128(a, imm) _mm_slli_si128_impl((a), (imm))


// _mm_srli_si128(a, imm): shift right by `imm` bytes, fill with zero
static inline __m128i _mm_srli_si128_impl(__m128i a, const int imm)
{
    if (imm >= 16) {
        a.u64[0] = 0;
        a.u64[1] = 0;
        return a;
    }
    if (imm <= 0) {
        return a;
    }
    if (imm >= 8) {
        a.u64[0] = a.u64[1] >> ((imm - 8) * 8);
        a.u64[1] = 0;
        return a;
    }
    const unsigned s = imm * 8;
    a.u64[0] = (a.u64[0] >> s) | (a.u64[1] << (64 - s));
    a.u64[1] = a.u64[1] >> s;
    return a;
}
#define _mm_srli_si128(a, imm) _mm_srli_si128_impl((a), (imm))


// _mm_shuffle_epi32(a, imm8): permute 32-bit lanes
// Result lane k = a.lane[(imm8 >> (2*k)) & 3]
static inline uint32_t xm_ppc_lane32(__m128i a, int k)
{
    if (k < 2) {
        return (uint32_t)(a.u64[0] >> (32 * k));
    }
    return (uint32_t)(a.u64[1] >> (32 * (k - 2)));
}

static inline __m128i _mm_shuffle_epi32_impl(__m128i a, int imm)
{
    const uint32_t l0 = xm_ppc_lane32(a, (imm >> 0) & 3);
    const uint32_t l1 = xm_ppc_lane32(a, (imm >> 2) & 3);
    const uint32_t l2 = xm_ppc_lane32(a, (imm >> 4) & 3);
    const uint32_t l3 = xm_ppc_lane32(a, (imm >> 6) & 3);
    __m128i r;
    r.u64[0] = ((uint64_t)l0) | (((uint64_t)l1) << 32);
    r.u64[1] = ((uint64_t)l2) | (((uint64_t)l3) << 32);
    return r;
}
#define _mm_shuffle_epi32(a, imm) _mm_shuffle_epi32_impl((a), (imm))

#ifndef _MM_SHUFFLE
#   define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#endif

static inline __m128 _mm_setzero_ps(void)
{
    __m128 r = {{ 0.0f, 0.0f, 0.0f, 0.0f }};
    return r;
}


static inline __m128 _mm_set1_ps(float a)
{
    __m128 r = {{ a, a, a, a }};
    return r;
}


static inline __m128 _mm_castsi128_ps(__m128i a)
{
    __m128 r;
    memcpy(&r, &a, 16);
    return r;
}


static inline __m128 _mm_cvtepi32_ps(__m128i a)
{
    __m128 r;
    r.f32[0] = (float)(int32_t)(uint32_t)(a.u64[0]);
    r.f32[1] = (float)(int32_t)(uint32_t)(a.u64[0] >> 32);
    r.f32[2] = (float)(int32_t)(uint32_t)(a.u64[1]);
    r.f32[3] = (float)(int32_t)(uint32_t)(a.u64[1] >> 32);
    return r;
}


static inline __m128i _mm_cvttps_epi32(__m128 a)
{
    __m128i r;
    const uint32_t l0 = (uint32_t)(int32_t)a.f32[0];
    const uint32_t l1 = (uint32_t)(int32_t)a.f32[1];
    const uint32_t l2 = (uint32_t)(int32_t)a.f32[2];
    const uint32_t l3 = (uint32_t)(int32_t)a.f32[3];
    r.u64[0] = ((uint64_t)l0) | (((uint64_t)l1) << 32);
    r.u64[1] = ((uint64_t)l2) | (((uint64_t)l3) << 32);
    return r;
}


static inline __m128 _mm_add_ps(__m128 a, __m128 b)
{
    __m128 r;
    r.f32[0] = a.f32[0] + b.f32[0];
    r.f32[1] = a.f32[1] + b.f32[1];
    r.f32[2] = a.f32[2] + b.f32[2];
    r.f32[3] = a.f32[3] + b.f32[3];
    return r;
}


static inline __m128 _mm_mul_ps(__m128 a, __m128 b)
{
    __m128 r;
    r.f32[0] = a.f32[0] * b.f32[0];
    r.f32[1] = a.f32[1] * b.f32[1];
    r.f32[2] = a.f32[2] * b.f32[2];
    r.f32[3] = a.f32[3] * b.f32[3];
    return r;
}


static inline __m128 _mm_and_ps(__m128 a, __m128 b)
{
    __m128 r;
    r.u32[0] = a.u32[0] & b.u32[0];
    r.u32[1] = a.u32[1] & b.u32[1];
    r.u32[2] = a.u32[2] & b.u32[2];
    r.u32[3] = a.u32[3] & b.u32[3];
    return r;
}


static inline __m128 _mm_or_ps(__m128 a, __m128 b)
{
    __m128 r;
    r.u32[0] = a.u32[0] | b.u32[0];
    r.u32[1] = a.u32[1] | b.u32[1];
    r.u32[2] = a.u32[2] | b.u32[2];
    r.u32[3] = a.u32[3] | b.u32[3];
    return r;
}


// AES falls back to soft_aesenc from soft_aes.h
#define _mm_aesenc_si128(v, key) soft_aesenc((v), (key))


// NEON intrinsics

static inline uint64x2_t vld1q_u64(const uint64_t *p)
{
    return _mm_load_si128((const __m128i *)p);
}


static inline uint64x2_t vld1q_s64(const int64_t *p)
{
    return _mm_load_si128((const __m128i *)p);
}


static inline void vst1q_u64(uint64_t *p, uint64x2_t v)
{
    _mm_store_si128((__m128i *)p, v);
}


static inline uint64x2_t vaddq_u64(uint64x2_t a, uint64x2_t b)
{
    return _mm_add_epi64(a, b);
}


static inline uint64x2_t veorq_u64(uint64x2_t a, uint64x2_t b)
{
    return _mm_xor_si128(a, b);
}


static inline uint64_t vcreate_u64(uint64_t a) { return a; }


static inline uint64x2_t vcombine_u64(uint64_t lo, uint64_t hi)
{
    uint64x2_t r;
    r.u64[0] = lo;
    r.u64[1] = hi;
    return r;
}

static inline uint64x2_t vreinterpretq_u64_u8(__m128i a) { return a; }


#define vgetq_lane_u64(v, i) ((v).u64[(i)])
#define vgetq_lane_s64(v, i) ((int64_t)(v).u64[(i)])

static inline int32_t vgetq_lane_s32(uint64x2_t v, int i)
{
    const int lane64 = i >> 1;
    const int half   = i & 1;
    return (int32_t)(uint32_t)(v.u64[lane64] >> (half * 32));
}


#endif // XMRIG_SSE2PPC_H
