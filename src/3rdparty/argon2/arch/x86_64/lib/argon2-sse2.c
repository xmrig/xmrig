#include "argon2-sse2.h"

#ifdef HAVE_SSE2
#ifdef __GNUC__
#   include <x86intrin.h>
#else
#   include <intrin.h>
#endif

#define ror64_16(x) \
    _mm_shufflehi_epi16( \
        _mm_shufflelo_epi16((x), _MM_SHUFFLE(0, 3, 2, 1)), \
        _MM_SHUFFLE(0, 3, 2, 1))
#define ror64_24(x) \
    _mm_xor_si128(_mm_srli_epi64((x), 24), _mm_slli_epi64((x), 40))
#define ror64_32(x) _mm_shuffle_epi32((x), _MM_SHUFFLE(2, 3, 0, 1))
#define ror64_63(x) \
    _mm_xor_si128(_mm_srli_epi64((x), 63), _mm_add_epi64((x), (x)))

static __m128i f(__m128i x, __m128i y)
{
    __m128i z = _mm_mul_epu32(x, y);
    return _mm_add_epi64(_mm_add_epi64(x, y), _mm_add_epi64(z, z));
}

#define G1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = f(A0, B0); \
        A1 = f(A1, B1); \
\
        D0 = _mm_xor_si128(D0, A0); \
        D1 = _mm_xor_si128(D1, A1); \
\
        D0 = ror64_32(D0); \
        D1 = ror64_32(D1); \
\
        C0 = f(C0, D0); \
        C1 = f(C1, D1); \
\
        B0 = _mm_xor_si128(B0, C0); \
        B1 = _mm_xor_si128(B1, C1); \
\
        B0 = ror64_24(B0); \
        B1 = ror64_24(B1); \
    } while ((void)0, 0)

#define G2(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = f(A0, B0); \
        A1 = f(A1, B1); \
\
        D0 = _mm_xor_si128(D0, A0); \
        D1 = _mm_xor_si128(D1, A1); \
\
        D0 = ror64_16(D0); \
        D1 = ror64_16(D1); \
\
        C0 = f(C0, D0); \
        C1 = f(C1, D1); \
\
        B0 = _mm_xor_si128(B0, C0); \
        B1 = _mm_xor_si128(B1, C1); \
\
        B0 = ror64_63(B0); \
        B1 = ror64_63(B1); \
    } while ((void)0, 0)

#define DIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        __m128i t0 = D0; \
        __m128i t1 = B0; \
        D0 = _mm_unpackhi_epi64(D1, _mm_unpacklo_epi64(t0, t0)); \
        D1 = _mm_unpackhi_epi64(t0, _mm_unpacklo_epi64(D1, D1)); \
        B0 = _mm_unpackhi_epi64(B0, _mm_unpacklo_epi64(B1, B1)); \
        B1 = _mm_unpackhi_epi64(B1, _mm_unpacklo_epi64(t1, t1)); \
    } while ((void)0, 0)

#define UNDIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        __m128i t0 = B0; \
        __m128i t1 = D0; \
        B0 = _mm_unpackhi_epi64(B1, _mm_unpacklo_epi64(B0, B0)); \
        B1 = _mm_unpackhi_epi64(t0, _mm_unpacklo_epi64(B1, B1)); \
        D0 = _mm_unpackhi_epi64(D0, _mm_unpacklo_epi64(D1, D1)); \
        D1 = _mm_unpackhi_epi64(D1, _mm_unpacklo_epi64(t1, t1)); \
    } while ((void)0, 0)

#define BLAKE2_ROUND(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        G1(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        DIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        G1(A0, B0, C1, D0, A1, B1, C0, D1); \
        G2(A0, B0, C1, D0, A1, B1, C0, D1); \
\
        UNDIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1); \
    } while ((void)0, 0)

#include "argon2-template-128.h"

void xmrig_ar2_fill_segment_sse2(const argon2_instance_t *instance, argon2_position_t position)
{
    fill_segment_128(instance, position);
}

extern int cpu_flags_has_sse2(void);
int xmrig_ar2_check_sse2(void) { return cpu_flags_has_sse2(); }

#else

void xmrig_ar2_fill_segment_sse2(const argon2_instance_t *instance, argon2_position_t position) {}
int xmrig_ar2_check_sse2(void) { return 0; }

#endif
