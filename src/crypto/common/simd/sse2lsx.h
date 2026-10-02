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
 *   along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * SSE2/SSSE3/SSE4.1 compatibility layer for LoongArch LSX (128 bit SIMD).
 *
 * The x86 build feeds the CryptoNight family (CryptoNight_x86.h,
 * CryptoNight_monero.h, CnCtx.h and soft_aes.h) and RandomX's BLAKE2b
 * compressor (blake2b-round.h) through the x86 intrinsic headers.  On
 * LoongArch <x86intrin.h>, <immintrin.h> and <smmintrin.h> do not exist.
 *
 * <lsxintrin.h> on the other hand already publishes __m128, __m128d and
 * __m128i with the very same layout, because GCC models the LSX types with
 * the SSE type names.  So only the _mm_* names have to be supplied here,
 * and every one of them is a thin wrapper over the matching LSX builtin.
 *
 * This layer needs -mlsx, see XMRIG_FEATURE_LSX in cmake/cpu.cmake.  It is
 * kept separate from the LASX layer of RandomX's BLAKE2b so that a CPU which
 * implements LSX only never executes a single LASX instruction.
 *
 * Notes on the two byte oriented intrinsics:
 *   - vbbit/vbsll/vbsrl shift the whole register and therefore map onto
 *     the whole-register shifts of the SSE layer.
 *   - vshuf.b indexes within a vector, which equals the PSHUFB
 *     behaviour for the byte masks below 16 used in blake2b-round.h.
 * Both were tested against a reference implementation of the Intel
 * documented behaviour before use.
 */

#ifndef XMRIG_SSE2LSX_H
#define XMRIG_SSE2LSX_H


#if !defined(__loongarch_sx)
#   error "sse2lsx.h requires LSX, build with -mlsx"
#endif

#include <stdint.h>
#include <string.h>

#include <lsxintrin.h>


#define XMRIG_LSX_INLINE static inline __attribute__((__always_inline__, __unused__))


/* The x86 headers define both, the LoongArch headers define neither. */
#ifndef _MM_SHUFFLE
#   define _MM_SHUFFLE(d, c, b, a) (((d) << 6) | ((c) << 4) | ((b) << 2) | (a))
#endif

#ifndef _MM_HINT_NTA
#   define _MM_HINT_NTA 0
#   define _MM_HINT_T0  3
#   define _MM_HINT_T1  2
#   define _MM_HINT_T2  1
#endif


/* --------------------------------------------------------- load/store */

XMRIG_LSX_INLINE __m128i _mm_load_si128(const void *p)  { return __lsx_vld(p, 0); }
XMRIG_LSX_INLINE __m128i _mm_loadu_si128(const void *p) { return __lsx_vld(p, 0); }
XMRIG_LSX_INLINE void    _mm_store_si128(void *p, __m128i v)
{
    __lsx_vst(v, p, 0);
}

XMRIG_LSX_INLINE void _mm_storeu_si128(void *p, __m128i v)
{
    __lsx_vst(v, p, 0);
}


/* --------------------------------------------------------------- sets */

XMRIG_LSX_INLINE __m128i _mm_setzero_si128(void)
{
    return __lsx_vreplgr2vr_d(0);
}


XMRIG_LSX_INLINE __m128i _mm_set_epi64x(long long e1, long long e0)
{
    __m128i v = __lsx_vreplgr2vr_d(e0);

    return __lsx_vinsgr2vr_d(v, e1, 1);
}


XMRIG_LSX_INLINE __m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
{
    __m128i v = __lsx_vreplgr2vr_w(e0);

    v = __lsx_vinsgr2vr_w(v, e1, 1);
    v = __lsx_vinsgr2vr_w(v, e2, 2);

    return __lsx_vinsgr2vr_w(v, e3, 3);
}


/*
 * PSHUFB style byte masks are written down as a list of 16 bytes, which is
 * what the x86 header offers too.  The element type of __m128i is a 64 bit one
 * on LoongArch, so the bytes are placed through a v16i8 first.
 */
XMRIG_LSX_INLINE __m128i _mm_setr_epi8(int8_t b0, int8_t b1, int8_t b2,  int8_t b3,
                                      int8_t b4, int8_t b5, int8_t b6,  int8_t b7,
                                      int8_t b8, int8_t b9, int8_t b10, int8_t b11,
                                      int8_t b12, int8_t b13, int8_t b14, int8_t b15)
{
    const v16i8 v = { b0, b1, b2,  b3,  b4,  b5,  b6,  b7,
                      b8, b9, b10, b11, b12, b13, b14, b15 };

    return (__m128i)v;
}


XMRIG_LSX_INLINE __m128i _mm_set1_epi32(int a)      { return __lsx_vreplgr2vr_w(a); }
XMRIG_LSX_INLINE __m128i _mm_set1_epi64x(long long a) { return __lsx_vreplgr2vr_d(a); }


/* MOVD/MOVQ: the low element, that is what the first bytes hold. */
XMRIG_LSX_INLINE int32_t _mm_cvtsi128_si32(__m128i a)
{
    int32_t r;

    memcpy(&r, &a, sizeof(r));

    return r;
}

XMRIG_LSX_INLINE int64_t _mm_cvtsi128_si64(__m128i a)
{
    int64_t r;

    memcpy(&r, &a, sizeof(r));

    return r;
}

XMRIG_LSX_INLINE __m128i _mm_cvtsi64_si128(long long a)
{
    return __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(0), a, 0);
}

XMRIG_LSX_INLINE __m128i _mm_cvtsi32_si128(int a)
{
    return __lsx_vinsgr2vr_w(__lsx_vreplgr2vr_w(0), a, 0);
}

XMRIG_LSX_INLINE __m128i _mm_add_epi8(__m128i a, __m128i b)  { return __lsx_vadd_b(a, b); }
XMRIG_LSX_INLINE __m128i _mm_add_epi16(__m128i a, __m128i b) { return __lsx_vadd_h(a, b); }
XMRIG_LSX_INLINE __m128i _mm_sub_epi8(__m128i a, __m128i b)  { return __lsx_vsub_b(a, b); }
XMRIG_LSX_INLINE __m128i _mm_sub_epi16(__m128i a, __m128i b) { return __lsx_vsub_h(a, b); }
XMRIG_LSX_INLINE __m128i _mm_sub_epi32(__m128i a, __m128i b) { return __lsx_vsub_w(a, b); }

XMRIG_LSX_INLINE __m128 _mm_set1_ps(float a)
{
    __m128 v;

    v[0] = a;
    v[1] = a;
    v[2] = a;
    v[3] = a;

    return v;
}

XMRIG_LSX_INLINE __m128  _mm_setzero_ps(void) { __m128 v;  v[0] = 0; v[1] = 0; v[2] = 0; v[3] = 0; return v; }
XMRIG_LSX_INLINE __m128d _mm_setzero_pd(void) { __m128d v; v[0] = 0; v[1] = 0; return v; }


/* ----------------------------------------------------- cast operations */

XMRIG_LSX_INLINE __m128  _mm_castsi128_ps(__m128i a) { return (__m128)a; }
XMRIG_LSX_INLINE __m128d _mm_castsi128_pd(__m128i a) { return (__m128d)a; }
XMRIG_LSX_INLINE __m128i _mm_castps_si128(__m128 a)  { return (__m128i)a; }
XMRIG_LSX_INLINE __m128d _mm_castps_pd(__m128 a)     { return (__m128d)a; }
XMRIG_LSX_INLINE __m128i _mm_castpd_si128(__m128d a) { return (__m128i)a; }
XMRIG_LSX_INLINE __m128  _mm_castpd_ps(__m128d a)    { return (__m128)a; }


/* --------------------------------------------------------------- logic */

XMRIG_LSX_INLINE __m128i _mm_and_si128(__m128i a, __m128i b) { return __lsx_vand_v(a, b); }
XMRIG_LSX_INLINE __m128i _mm_or_si128(__m128i a, __m128i b)  { return __lsx_vor_v(a, b); }
XMRIG_LSX_INLINE __m128i _mm_xor_si128(__m128i a, __m128i b) { return __lsx_vxor_v(a, b); }

/* PANDN is ~a & b, while vandn.v computes vj & ~vk. */
XMRIG_LSX_INLINE __m128i _mm_andnot_si128(__m128i a, __m128i b) { return __lsx_vandn_v(b, a); }

XMRIG_LSX_INLINE __m128 _mm_and_ps(__m128 a, __m128 b) { return (__m128)__lsx_vand_v((__m128i)a, (__m128i)b); }
XMRIG_LSX_INLINE __m128 _mm_or_ps(__m128 a, __m128 b)  { return (__m128)__lsx_vor_v((__m128i)a, (__m128i)b); }


/* ---------------------------------------------------------- arithmetic */

XMRIG_LSX_INLINE __m128i _mm_add_epi64(__m128i a, __m128i b) { return __lsx_vadd_d(a, b); }
XMRIG_LSX_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b) { return __lsx_vadd_w(a, b); }
XMRIG_LSX_INLINE __m128i _mm_sub_epi64(__m128i a, __m128i b) { return __lsx_vsub_d(a, b); }

XMRIG_LSX_INLINE __m128i _mm_cmpeq_epi32(__m128i a, __m128i b) { return __lsx_vseq_w(a, b); }
XMRIG_LSX_INLINE __m128i _mm_cmpeq_epi64(__m128i a, __m128i b) { return __lsx_vseq_d(a, b); }


/* -------------------------------------------------------------- floats */

XMRIG_LSX_INLINE __m128 _mm_add_ps(__m128 a, __m128 b)
{
    return (__m128)((v4f32)a + (v4f32)b);
}

XMRIG_LSX_INLINE __m128 _mm_mul_ps(__m128 a, __m128 b)
{
    return (__m128)((v4f32)a * (v4f32)b);
}

XMRIG_LSX_INLINE __m128 _mm_cvtepi32_ps(__m128i a)
{
    return (__m128)((v4f32)((v4i32)a));
}

XMRIG_LSX_INLINE __m128i _mm_cvttps_epi32(__m128 a)
{
    return (__m128i)((v4i32)((v4f32)a));
}

/* MOVHLPS: the upper half of the second operand, the upper half of the
 * first duplicated into the upper half of the result. */
XMRIG_LSX_INLINE __m128 _mm_movehl_ps(__m128 a, __m128 b)
{
    __m128 r;

    r[0] = b[2];
    r[1] = b[3];
    r[2] = a[2];
    r[3] = a[3];

    return r;
}

/* SQRTSD only touches the low element. */
XMRIG_LSX_INLINE __m128d _mm_sqrt_sd(__m128d a, __m128d b)
{
    __m128d r;

    r[0] = __builtin_sqrt(b[0]);
    r[1] = a[1];

    return r;
}


/* ---------------------------------------------- shuffle / align / interleave */

XMRIG_LSX_INLINE __m128i _mm_shuffle_epi32(__m128i a, int imm) { return __lsx_vshuf4i_w(a, imm); }

/* PSHUFB: vshuf.b with both sources pointing at the same register.
 * Masks with bit 7 set ( PSHUFB zeroes those bytes ) are not used by
 * the CryptoNight and BLAKE2b code paths this header serves. */
XMRIG_LSX_INLINE __m128i _mm_shuffle_epi8(__m128i a, __m128i mask) { return __lsx_vshuf_b(a, a, mask); }

XMRIG_LSX_INLINE __m128i _mm_unpacklo_epi64(__m128i a, __m128i b) { return __lsx_vpickev_d(b, a); }
XMRIG_LSX_INLINE __m128i _mm_unpackhi_epi64(__m128i a, __m128i b) { return __lsx_vpickod_d(b, a); }

/*
 * PALIGNR shifts the concatenation of both operands, which is the same as
 * shifting one operand left by ( 16 - n ) and the other one right by n and
 * merging both halves.  __lsx_vbsll_v/__lsx_vbsrl_v need a literal
 * immediate, so this is a macro and not a function.
 */
#define _mm_alignr_epi8(a, b, imm)                                              \
    ((__m128i)(__lsx_vbsrl_v((__m128i)(b), (imm)) |                               \
               __lsx_vbsll_v((__m128i)(a), 16 - (imm))))

/* PSLLDQ / PSRLDQ, the same argument applies. */
#define _mm_slli_si128(a, imm) __lsx_vbsll_v((__m128i)(a), (imm))
#define _mm_srli_si128(a, imm) __lsx_vbsrl_v((__m128i)(a), (imm))

/* The immediate shifts need a literal immediate as well. */
#define _mm_slli_epi64(a, imm) __lsx_vslli_d((__m128i)(a), (imm))
#define _mm_srli_epi64(a, imm) __lsx_vsrli_d((__m128i)(a), (imm))
#define _mm_slli_epi32(a, imm) __lsx_vslli_w((__m128i)(a), (imm))
#define _mm_srli_epi32(a, imm) __lsx_vsrli_w((__m128i)(a), (imm))


/* -------------------------------------------------------- prefetch/pause */

XMRIG_LSX_INLINE void _mm_prefetch(const char *p, int hint)
{
    __builtin_prefetch(p, 0, hint == _MM_HINT_T0 ? 3 : (hint == _MM_HINT_T1 ? 2 : 1));
}

XMRIG_LSX_INLINE void _mm_pause(void)   {}
XMRIG_LSX_INLINE void _mm_mfence(void)  { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
XMRIG_LSX_INLINE void _mm_lfence(void)  { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
XMRIG_LSX_INLINE void _mm_sfence(void)  { __atomic_thread_fence(__ATOMIC_SEQ_CST); }


/* --------------------------------------------------------------- AES128
 *
 * AESENC/AESKEYGENASSIST placeholders, exactly as in the RISC-V layer of
 * this repository ( src/crypto/cn/sse2rvv.h ).
 *
 * LoongArch has no AES instructions, and BasicCpuInfo_loongarch.cpp reports
 * FLAG_AES as false, so the CryptoNight code always takes the SOFT_AES
 * branch built on soft_aes.h.  These two names only exist so that the
 * hardware AES templates still compile; a LoongArch build never calls them.
 */

XMRIG_LSX_INLINE __m128i _mm_aesenc_si128(__m128i a, __m128i roundkey)
{
    return _mm_xor_si128(a, roundkey);
}

XMRIG_LSX_INLINE __m128i _mm_aeskeygenassist_si128(__m128i a, const int rcon)
{
    (void)rcon;

    return a;
}


#endif /* XMRIG_SSE2LSX_H */
