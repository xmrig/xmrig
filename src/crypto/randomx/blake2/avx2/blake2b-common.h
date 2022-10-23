#ifndef BLAKE2_AVX2_BLAKE2B_COMMON_H
#define BLAKE2_AVX2_BLAKE2B_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "blake2.h"

#define LOAD128(p)  _mm_load_si128( (__m128i *)(p) )
#define STORE128(p,r) _mm_store_si128((__m128i *)(p), r)

#define LOADU128(p)  _mm_loadu_si128( (__m128i *)(p) )
#define STOREU128(p,r) _mm_storeu_si128((__m128i *)(p), r)

#define LOAD(p)  _mm256_load_si256( (__m256i *)(p) )
#define STORE(p,r) _mm256_store_si256((__m256i *)(p), r)

#define LOADU(p)  _mm256_loadu_si256( (__m256i *)(p) )
#define STOREU(p,r) _mm256_storeu_si256((__m256i *)(p), r)

static INLINE uint64_t LOADU64(void const * p) {
  uint64_t v;
  memcpy(&v, p, sizeof v);
  return v;
}

#define ROTATE16 _mm256_setr_epi8( 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, \
                                   2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9 )

#define ROTATE24 _mm256_setr_epi8( 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, \
                                   3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10 )

#define ADD(a, b) _mm256_add_epi64(a, b)
#define SUB(a, b) _mm256_sub_epi64(a, b)

#define XOR(a, b) _mm256_xor_si256(a, b)
#define AND(a, b) _mm256_and_si256(a, b)
#define  OR(a, b) _mm256_or_si256(a, b)

#define ROT32(x) _mm256_shuffle_epi32((x), _MM_SHUFFLE(2, 3, 0, 1))
#define ROT24(x) _mm256_shuffle_epi8((x), ROTATE24)
#define ROT16(x) _mm256_shuffle_epi8((x), ROTATE16)
#define ROT63(x) _mm256_or_si256(_mm256_srli_epi64((x), 63), ADD((x), (x)))

#endif
