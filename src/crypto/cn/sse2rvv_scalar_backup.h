/* XMRig
 * Copyright (c) 2025 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

/*
 * SSE to RISC-V compatibility header
 * Provides scalar implementations of SSE intrinsics for RISC-V architecture
 */

#ifndef XMRIG_SSE2RVV_H
#define XMRIG_SSE2RVV_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>

/* 128-bit vector type */
typedef union {
    uint8_t  u8[16];
    uint16_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
    int8_t   i8[16];
    int16_t  i16[8];
    int32_t  i32[4];
    int64_t  i64[2];
} __m128i_union;

typedef __m128i_union __m128i;

/* Set operations */
static inline __m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
{
    __m128i result;
    result.i32[0] = e0;
    result.i32[1] = e1;
    result.i32[2] = e2;
    result.i32[3] = e3;
    return result;
}

static inline __m128i _mm_set_epi64x(int64_t e1, int64_t e0)
{
    __m128i result;
    result.i64[0] = e0;
    result.i64[1] = e1;
    return result;
}

static inline __m128i _mm_setzero_si128(void)
{
    __m128i result;
    memset(&result, 0, sizeof(result));
    return result;
}

/* Extract/insert operations */
static inline int _mm_cvtsi128_si32(__m128i a)
{
    return a.i32[0];
}

static inline int64_t _mm_cvtsi128_si64(__m128i a)
{
    return a.i64[0];
}

static inline __m128i _mm_cvtsi32_si128(int a)
{
    __m128i result = _mm_setzero_si128();
    result.i32[0] = a;
    return result;
}

static inline __m128i _mm_cvtsi64_si128(int64_t a)
{
    __m128i result = _mm_setzero_si128();
    result.i64[0] = a;
    return result;
}

/* Shuffle operations */
static inline __m128i _mm_shuffle_epi32(__m128i a, int imm8)
{
    __m128i result;
    result.u32[0] = a.u32[(imm8 >> 0) & 0x3];
    result.u32[1] = a.u32[(imm8 >> 2) & 0x3];
    result.u32[2] = a.u32[(imm8 >> 4) & 0x3];
    result.u32[3] = a.u32[(imm8 >> 6) & 0x3];
    return result;
}

/* Logical operations */
static inline __m128i _mm_xor_si128(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = a.u64[0] ^ b.u64[0];
    result.u64[1] = a.u64[1] ^ b.u64[1];
    return result;
}

static inline __m128i _mm_or_si128(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = a.u64[0] | b.u64[0];
    result.u64[1] = a.u64[1] | b.u64[1];
    return result;
}

static inline __m128i _mm_and_si128(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = a.u64[0] & b.u64[0];
    result.u64[1] = a.u64[1] & b.u64[1];
    return result;
}

static inline __m128i _mm_andnot_si128(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = (~a.u64[0]) & b.u64[0];
    result.u64[1] = (~a.u64[1]) & b.u64[1];
    return result;
}

/* Shift operations */
static inline __m128i _mm_slli_si128(__m128i a, int imm8)
{
    __m128i result = _mm_setzero_si128();
    int count = imm8 & 0xFF;
    if (count > 15) return result;
    
    for (int i = 0; i < 16 - count; i++) {
        result.u8[i + count] = a.u8[i];
    }
    return result;
}

static inline __m128i _mm_srli_si128(__m128i a, int imm8)
{
    __m128i result = _mm_setzero_si128();
    int count = imm8 & 0xFF;
    if (count > 15) return result;
    
    for (int i = count; i < 16; i++) {
        result.u8[i - count] = a.u8[i];
    }
    return result;
}

static inline __m128i _mm_slli_epi64(__m128i a, int imm8)
{
    __m128i result;
    if (imm8 > 63) {
        result.u64[0] = 0;
        result.u64[1] = 0;
    } else {
        result.u64[0] = a.u64[0] << imm8;
        result.u64[1] = a.u64[1] << imm8;
    }
    return result;
}

static inline __m128i _mm_srli_epi64(__m128i a, int imm8)
{
    __m128i result;
    if (imm8 > 63) {
        result.u64[0] = 0;
        result.u64[1] = 0;
    } else {
        result.u64[0] = a.u64[0] >> imm8;
        result.u64[1] = a.u64[1] >> imm8;
    }
    return result;
}

/* Load/store operations */
static inline __m128i _mm_load_si128(const __m128i* p)
{
    __m128i result;
    memcpy(&result, p, sizeof(__m128i));
    return result;
}

static inline __m128i _mm_loadu_si128(const __m128i* p)
{
    __m128i result;
    memcpy(&result, p, sizeof(__m128i));
    return result;
}

static inline void _mm_store_si128(__m128i* p, __m128i a)
{
    memcpy(p, &a, sizeof(__m128i));
}

static inline void _mm_storeu_si128(__m128i* p, __m128i a)
{
    memcpy(p, &a, sizeof(__m128i));
}

/* Arithmetic operations */
static inline __m128i _mm_add_epi64(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = a.u64[0] + b.u64[0];
    result.u64[1] = a.u64[1] + b.u64[1];
    return result;
}

static inline __m128i _mm_add_epi32(__m128i a, __m128i b)
{
    __m128i result;
    for (int i = 0; i < 4; i++) {
        result.i32[i] = a.i32[i] + b.i32[i];
    }
    return result;
}

static inline __m128i _mm_sub_epi64(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = a.u64[0] - b.u64[0];
    result.u64[1] = a.u64[1] - b.u64[1];
    return result;
}

static inline __m128i _mm_mul_epu32(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = (uint64_t)a.u32[0] * (uint64_t)b.u32[0];
    result.u64[1] = (uint64_t)a.u32[2] * (uint64_t)b.u32[2];
    return result;
}

/* Unpack operations */
static inline __m128i _mm_unpacklo_epi64(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = a.u64[0];
    result.u64[1] = b.u64[0];
    return result;
}

static inline __m128i _mm_unpackhi_epi64(__m128i a, __m128i b)
{
    __m128i result;
    result.u64[0] = a.u64[1];
    result.u64[1] = b.u64[1];
    return result;
}

/* Pause instruction for spin-wait loops */
static inline void _mm_pause(void)
{
    /* RISC-V doesn't have a direct equivalent to x86 PAUSE
     * Use a simple NOP or yield hint */
    __asm__ __volatile__("nop");
}

/* Memory fence */
static inline void _mm_mfence(void)
{
    __asm__ __volatile__("fence" ::: "memory");
}

static inline void _mm_lfence(void)
{
    __asm__ __volatile__("fence r,r" ::: "memory");
}

static inline void _mm_sfence(void)
{
    __asm__ __volatile__("fence w,w" ::: "memory");
}

/* Comparison operations */
static inline __m128i _mm_cmpeq_epi32(__m128i a, __m128i b)
{
    __m128i result;
    for (int i = 0; i < 4; i++) {
        result.u32[i] = (a.u32[i] == b.u32[i]) ? 0xFFFFFFFF : 0;
    }
    return result;
}

static inline __m128i _mm_cmpeq_epi64(__m128i a, __m128i b)
{
    __m128i result;
    for (int i = 0; i < 2; i++) {
        result.u64[i] = (a.u64[i] == b.u64[i]) ? 0xFFFFFFFFFFFFFFFFULL : 0;
    }
    return result;
}

/* Additional shift operations */
static inline __m128i _mm_slli_epi32(__m128i a, int imm8)
{
    __m128i result;
    if (imm8 > 31) {
        for (int i = 0; i < 4; i++) result.u32[i] = 0;
    } else {
        for (int i = 0; i < 4; i++) {
            result.u32[i] = a.u32[i] << imm8;
        }
    }
    return result;
}

static inline __m128i _mm_srli_epi32(__m128i a, int imm8)
{
    __m128i result;
    if (imm8 > 31) {
        for (int i = 0; i < 4; i++) result.u32[i] = 0;
    } else {
        for (int i = 0; i < 4; i++) {
            result.u32[i] = a.u32[i] >> imm8;
        }
    }
    return result;
}

/* 64-bit integer operations */
static inline __m128i _mm_set1_epi64x(int64_t a)
{
    __m128i result;
    result.i64[0] = a;
    result.i64[1] = a;
    return result;
}

/* Float type for compatibility - we'll treat it as int for simplicity */
typedef __m128i __m128;

/* Float operations - simplified scalar implementations */
static inline __m128 _mm_set1_ps(float a)
{
    __m128 result;
    uint32_t val;
    memcpy(&val, &a, sizeof(float));
    for (int i = 0; i < 4; i++) {
        result.u32[i] = val;
    }
    return result;
}

static inline __m128 _mm_setzero_ps(void)
{
    __m128 result;
    memset(&result, 0, sizeof(result));
    return result;
}

static inline __m128 _mm_add_ps(__m128 a, __m128 b)
{
    __m128 result;
    float fa[4], fb[4], fr[4];
    memcpy(fa, &a, sizeof(__m128));
    memcpy(fb, &b, sizeof(__m128));
    for (int i = 0; i < 4; i++) {
        fr[i] = fa[i] + fb[i];
    }
    memcpy(&result, fr, sizeof(__m128));
    return result;
}

static inline __m128 _mm_mul_ps(__m128 a, __m128 b)
{
    __m128 result;
    float fa[4], fb[4], fr[4];
    memcpy(fa, &a, sizeof(__m128));
    memcpy(fb, &b, sizeof(__m128));
    for (int i = 0; i < 4; i++) {
        fr[i] = fa[i] * fb[i];
    }
    memcpy(&result, fr, sizeof(__m128));
    return result;
}

static inline __m128 _mm_and_ps(__m128 a, __m128 b)
{
    __m128 result;
    result.u64[0] = a.u64[0] & b.u64[0];
    result.u64[1] = a.u64[1] & b.u64[1];
    return result;
}

static inline __m128 _mm_or_ps(__m128 a, __m128 b)
{
    __m128 result;
    result.u64[0] = a.u64[0] | b.u64[0];
    result.u64[1] = a.u64[1] | b.u64[1];
    return result;
}

static inline __m128 _mm_cvtepi32_ps(__m128i a)
{
    __m128 result;
    float fr[4];
    for (int i = 0; i < 4; i++) {
        fr[i] = (float)a.i32[i];
    }
    memcpy(&result, fr, sizeof(__m128));
    return result;
}

static inline __m128i _mm_cvttps_epi32(__m128 a)
{
    __m128i result;
    float fa[4];
    memcpy(fa, &a, sizeof(__m128));
    for (int i = 0; i < 4; i++) {
        result.i32[i] = (int32_t)fa[i];
    }
    return result;
}

/* Casting operations */
static inline __m128 _mm_castsi128_ps(__m128i a)
{
    __m128 result;
    memcpy(&result, &a, sizeof(__m128));
    return result;
}

static inline __m128i _mm_castps_si128(__m128 a)
{
    __m128i result;
    memcpy(&result, &a, sizeof(__m128));
    return result;
}

/* Additional set operations */
static inline __m128i _mm_set1_epi32(int a)
{
    __m128i result;
    for (int i = 0; i < 4; i++) {
        result.i32[i] = a;
    }
    return result;
}

/* AES instructions - these are placeholders, actual AES is done via soft_aes.h */
/* On RISC-V without crypto extensions, these should never be called directly */
/* They are only here for compilation compatibility */
static inline __m128i _mm_aesenc_si128(__m128i a, __m128i roundkey)
{
    /* This is a placeholder - actual implementation should use soft_aes */
    /* If this function is called, it means SOFT_AES template parameter wasn't used */
    /* We return a XOR as a minimal fallback, but proper code should use soft_aesenc */
    return _mm_xor_si128(a, roundkey);
}

static inline __m128i _mm_aeskeygenassist_si128(__m128i a, const int rcon)
{
    /* Placeholder for AES key generation - should use soft_aeskeygenassist */
    return a;
}

/* Rotate right operation for soft_aes.h */
static inline uint32_t _rotr(uint32_t value, unsigned int count)
{
    const unsigned int mask = 31;
    count &= mask;
    return (value >> count) | (value << ((-count) & mask));
}

/* ARM NEON compatibility types and intrinsics for RISC-V */
typedef __m128i_union uint64x2_t;
typedef __m128i_union uint8x16_t;
typedef __m128i_union int64x2_t;
typedef __m128i_union int32x4_t;

static inline uint64x2_t vld1q_u64(const uint64_t *ptr)
{
    uint64x2_t result;
    result.u64[0] = ptr[0];
    result.u64[1] = ptr[1];
    return result;
}

static inline int64x2_t vld1q_s64(const int64_t *ptr)
{
    int64x2_t result;
    result.i64[0] = ptr[0];
    result.i64[1] = ptr[1];
    return result;
}

static inline void vst1q_u64(uint64_t *ptr, uint64x2_t val)
{
    ptr[0] = val.u64[0];
    ptr[1] = val.u64[1];
}

static inline uint64x2_t veorq_u64(uint64x2_t a, uint64x2_t b)
{
    uint64x2_t result;
    result.u64[0] = a.u64[0] ^ b.u64[0];
    result.u64[1] = a.u64[1] ^ b.u64[1];
    return result;
}

static inline uint64x2_t vaddq_u64(uint64x2_t a, uint64x2_t b)
{
    uint64x2_t result;
    result.u64[0] = a.u64[0] + b.u64[0];
    result.u64[1] = a.u64[1] + b.u64[1];
    return result;
}

static inline uint64x2_t vreinterpretq_u64_u8(uint8x16_t a)
{
    uint64x2_t result;
    memcpy(&result, &a, sizeof(uint64x2_t));
    return result;
}

static inline uint64_t vgetq_lane_u64(uint64x2_t v, int lane)
{
    return v.u64[lane];
}

static inline int64_t vgetq_lane_s64(int64x2_t v, int lane)
{
    return v.i64[lane];
}

static inline int32_t vgetq_lane_s32(int32x4_t v, int lane)
{
    return v.i32[lane];
}

typedef struct { uint64_t val[1]; } uint64x1_t;

static inline uint64x1_t vcreate_u64(uint64_t a)
{
    uint64x1_t result;
    result.val[0] = a;
    return result;
}

static inline uint64x2_t vcombine_u64(uint64x1_t low, uint64x1_t high)
{
    uint64x2_t result;
    result.u64[0] = low.val[0];
    result.u64[1] = high.val[0];
    return result;
}

#ifdef __cplusplus
}
#endif

#endif /* XMRIG_SSE2RVV_H */
