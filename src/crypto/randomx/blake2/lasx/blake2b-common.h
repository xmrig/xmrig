/*
 * AVX2 compatibility layer for LoongArch LASX (256 bit SIMD).
 *
 * The LASX flavour of RandomX's BLAKE2b compressor lives in
 * blake2b_lasx.c and its message loading table in blake2b-load-lasx.h.
 * Both files are kept byte for byte identical to their AVX2 twins
 * ( blake2b_avx2.c and blake2b-load-avx2.h ) and every _mm256_* name
 * they use is supplied here on top of <lasxintrin.h>.
 *
 * This is deliberately a separate layer from src/crypto/common/simd/sse2lsx.h
 * ( which covers the LSX / SSE2 pairing ).  A CPU which implements LSX but
 * no LASX never sees -mlasx and never links this header, so the two
 * extensions are adapted and enabled independently.
 *
 * Facts verified against the LoongArch backend of GCC 15.3 before use:
 *   - vpickev.d( vj, vk ) is { vk[0], vj[0] } per 128 bit lane, i.e. the
 *     two operands are in the opposite order of _mm256_unpacklo_epi64.
 *   - vshuf.b( vj, vk, va ) uses va as the control vector, indexes
 *     within one 128 bit lane and picks vk for control bytes with bit 4
 *     set, which is exactly PSHUFB while bit 4 stays clear.
 *   - vbitsel.v( vj, vk, va ) is ( va & vk ) | ( ~va & vj ), which is
 *     _mm256_blend_epi32 once the immediate is turned into a lane mask.
 *   - vbsll.v / vbsrl.v and vshuf4i.w keep the encodings of PSLLDQ,
 *     PSRLDQ and PSHUFD.
 *   - vpermi.d uses the same immediate layout as VPERMQ.
 */

#ifndef XMRIG_LASX_BLAKE2B_COMMON_H
#define XMRIG_LASX_BLAKE2B_COMMON_H


#if !defined(__loongarch_asx)
#   error "blake2b-common.h (LASX) requires LASX, build with -mlasx"
#endif

#include <stdint.h>
#include <string.h>

#include <lsxintrin.h>
#include <lasxintrin.h>

#include "blake2.h"


/* --------------------------------------------------------- load/store
 *
 * LoongArch vector loads and stores accept both, aligned and unaligned
 * addresses, so the AVX2 aligned and unaligned helpers coincide.
 */

#define LOAD128(p)    ((__m128i)__lsx_vld((const void *)(p), 0))
#define STORE128(p,r) __lsx_vst((__m128i)(r), (void *)(p), 0)

#define LOADU128(p)    LOAD128(p)
#define STOREU128(p,r) STORE128(p, r)

#define LOAD(p)    ((__m256i)__lasx_xvld((const void *)(p), 0))
#define STORE(p,r) __lasx_xvst((__m256i)(r), (void *)(p), 0)

#define LOADU(p)    LOAD(p)
#define STOREU(p,r) STORE(p, r)


static INLINE uint64_t LOADU64(void const * p) {
  uint64_t v;
  memcpy(&v, p, sizeof v);
  return v;
}


/* ------------------------------------------------------- x86 stand-ins
 *
 * _MM_SHUFFLE and the byte level intrinsics are the only pieces the
 * LoongArch headers do not offer.
 */

#ifndef _MM_SHUFFLE
#   define _MM_SHUFFLE(d, c, b, a) (((d) << 6) | ((c) << 4) | ((b) << 2) | (a))
#endif

#define _mm256_setr_epi8(b0, b1, b2,  b3,  b4,  b5,  b6,  b7,   \
                         b8, b9, b10, b11, b12, b13, b14, b15, \
                         b16, b17, b18, b19, b20, b21, b22, b23, \
                         b24, b25, b26, b27, b28, b29, b30, b31) \
  ({ uint8_t xmrig_lasx_mask_[32] = { b0, b1, b2,  b3,  b4,  b5,  b6,  b7,   \
                                      b8, b9, b10, b11, b12, b13, b14, b15, \
                                      b16, b17, b18, b19, b20, b21, b22, b23, \
                                      b24, b25, b26, b27, b28, b29, b30, b31 }; \
     (__m256i)__lasx_xvld(xmrig_lasx_mask_, 0); })

#define _mm256_set_epi64x(e3, e2, e1, e0) ((__m256i)(v4i64){ (e0), (e1), (e2), (e3) })

/* PSHUFB, and PSHUFB on both halves at once. */
#define _mm256_shuffle_epi8(a, b) __lasx_xvshuf_b((__m256i)(a), (__m256i)(a), (__m256i)(b))

#define _mm256_unpacklo_epi64(a, b) __lasx_xvpickev_d((__m256i)(b), (__m256i)(a))
#define _mm256_unpackhi_epi64(a, b) __lasx_xvpickod_d((__m256i)(b), (__m256i)(a))

/* _mm256_blend_epi32: the immediate selects the 32 bit word, bit i
 * belongs to word i.  vbitsel.v takes the word from its second operand
 * where the mask is set, which is the Intel convention. */
#define XMRIG_LASX_BLEND32_MASK(imm) ((__m256i)(v8i32){          \
    -(int)(((imm) >> 0) & 1), -(int)(((imm) >> 1) & 1),             \
    -(int)(((imm) >> 2) & 1), -(int)(((imm) >> 3) & 1),             \
    -(int)(((imm) >> 4) & 1), -(int)(((imm) >> 5) & 1),             \
    -(int)(((imm) >> 6) & 1), -(int)(((imm) >> 7) & 1) })

#define _mm256_blend_epi32(a, b, imm) \
    __lasx_xvbitsel_v((__m256i)(a), (__m256i)(b), XMRIG_LASX_BLEND32_MASK(imm))

#define _mm256_shuffle_epi32(a, imm) __lasx_xvshuf4i_w((__m256i)(a), (imm))
#define _mm256_permute4x64_epi64(a, imm) __lasx_xvpermi_d((__m256i)(a), (imm))

/* _mm256_broadcastsi128_si256 duplicates the low 128 bit lane.  The two
 * vector widths cannot be cast into each other, so the halves are copied one
 * by one. */
static INLINE __m256i _mm256_broadcastsi128_si256(__m128i a)
{
  __m256i r;

  r[0] = a[0];
  r[1] = a[1];
  r[2] = a[0];
  r[3] = a[1];

  return r;
}

/* _mm256_alignr_epi8 == _mm256_alignr_epi8 on both 128 bit lanes.
 * vshuf.b is the only general 256 bit byte shuffle LASX offers, so the
 * PALIGNR pairing of both operands is expressed with it directly. */
#define _mm256_alignr_epi8(a, b, imm) __lasx_xvshuf_b(                       \
    (__m256i)(a), (__m256i)(b), (__m256i)(v32i8){                               \
        (imm) + 0, (imm) + 1, (imm) + 2,  (imm) + 3,  (imm) + 4,                  \
        (imm) + 5, (imm) + 6, (imm) + 7,  (imm) + 8,  (imm) + 9,                  \
        (imm) + 10, (imm) + 11, (imm) + 12, (imm) + 13, (imm) + 14, (imm) + 15,     \
        (imm) + 0, (imm) + 1, (imm) + 2,  (imm) + 3,  (imm) + 4,                  \
        (imm) + 5, (imm) + 6, (imm) + 7,  (imm) + 8,  (imm) + 9,                  \
        (imm) + 10, (imm) + 11, (imm) + 12, (imm) + 13, (imm) + 14, (imm) + 15 })

#define _mm256_zeroupper() ((void)0)


/* --------------------------------------------------------------- math */

#define ROTATE16 _mm256_setr_epi8( 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, \
                                   2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9 )

#define ROTATE24 _mm256_setr_epi8( 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, \
                                   3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10 )

#define ADD(a, b) ((__m256i)((v4i64)(a) + (v4i64)(b)))
#define SUB(a, b) ((__m256i)((v4i64)(a) - (v4i64)(b)))
#define XOR(a, b) __lasx_xvxor_v((__m256i)(a), (__m256i)(b))
#define AND(a, b) __lasx_xvand_v((__m256i)(a), (__m256i)(b))
#define  OR(a, b) __lasx_xvor_v((__m256i)(a), (__m256i)(b))

#define ROT32(x) __lasx_xvshuf4i_w((__m256i)(x), _MM_SHUFFLE(2, 3, 0, 1))
#define ROT24(x) _mm256_shuffle_epi8((x), ROTATE24)
#define ROT16(x) _mm256_shuffle_epi8((x), ROTATE16)
#define ROT63(x) OR(__lasx_xvsrli_d((__m256i)(x), 63), ADD((x), (x)))

#endif
