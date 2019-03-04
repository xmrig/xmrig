/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CRYPTONIGHT_MONERO_H
#define XMRIG_CRYPTONIGHT_MONERO_H


#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <x86intrin.h>


static inline __m128i int_sqrt_v2(const uint64_t n0)
{
    __m128d x = _mm_castsi128_pd(_mm_add_epi64(_mm_cvtsi64_si128(n0 >> 12), _mm_set_epi64x(0, 1023ULL << 52)));
    x = _mm_sqrt_sd(_mm_setzero_pd(), x);
    uint64_t r = (uint64_t)(_mm_cvtsi128_si64(_mm_castpd_si128(x)));

    const uint64_t s = r >> 20;
    r >>= 19;

    uint64_t x2 = (s - (1022ULL << 32)) * (r - s - (1022ULL << 32) + 1);
#   if (defined(_MSC_VER) || __GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ > 1)) && (defined(__x86_64__) || defined(_M_AMD64))
    _addcarry_u64(_subborrow_u64(0, x2, n0, (unsigned long long int*)&x2), r, 0, (unsigned long long int*)&r);
#   else
    if (x2 < n0) ++r;
#   endif

    return _mm_cvtsi64_si128(r);
}


#   define VARIANT1_INIT(part) \
    uint64_t tweak1_2_##part = (*(const uint64_t*)(input + 35 + part * size) ^ \
                               *((const uint64_t*)(ctx[part]->state) + 24)); \

#   define VARIANT2_INIT(part) \
    __m128i division_result_xmm_##part = _mm_cvtsi64_si128(h##part[12]); \
    __m128i sqrt_result_xmm_##part = _mm_cvtsi64_si128(h##part[13]);

#ifdef _MSC_VER
#   define VARIANT2_SET_ROUNDING_MODE() { _control87(RC_DOWN, MCW_RC); }
#else
#   define VARIANT2_SET_ROUNDING_MODE() { fesetround(FE_DOWNWARD); }
#endif

#   define VARIANT2_INTEGER_MATH(part, cl, cx) \
    { \
        const uint64_t sqrt_result = (uint64_t)(_mm_cvtsi128_si64(sqrt_result_xmm_##part)); \
        const uint64_t cx_0 = _mm_cvtsi128_si64(cx); \
        cl ^= (uint64_t)(_mm_cvtsi128_si64(division_result_xmm_##part)) ^ (sqrt_result << 32); \
        const uint32_t d = (uint32_t)(cx_0 + (sqrt_result << 1)) | 0x80000001UL; \
        const uint64_t cx_1 = _mm_cvtsi128_si64(_mm_srli_si128(cx, 8)); \
        const uint64_t division_result = (uint32_t)(cx_1 / d) + ((cx_1 % d) << 32); \
        division_result_xmm_##part = _mm_cvtsi64_si128((int64_t)(division_result)); \
        sqrt_result_xmm_##part = int_sqrt_v2(cx_0 + division_result); \
    }

#   define VARIANT2_SHUFFLE(base_ptr, offset, _a, _b, _b1) \
    { \
        const __m128i chunk1 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10))); \
        const __m128i chunk2 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20))); \
        const __m128i chunk3 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30))); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, _b1)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30)), _mm_add_epi64(chunk2, _a)); \
    }

#   define VARIANT4_SHUFFLE(base_ptr, offset, _a, _b, _b1, _c) \
    { \
        const __m128i chunk1 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10))); \
        const __m128i chunk2 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20))); \
        const __m128i chunk3 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30))); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, _b1)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30)), _mm_add_epi64(chunk2, _a)); \
        _c = _mm_xor_si128(_mm_xor_si128(_c, chunk3), _mm_xor_si128(chunk1, chunk2)); \
    }

#   define VARIANT2_SHUFFLE2(base_ptr, offset, _a, _b, _b1, hi, lo) \
    { \
        const __m128i chunk1 = _mm_xor_si128(_mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10))), _mm_set_epi64x(lo, hi)); \
        const __m128i chunk2 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20))); \
        hi ^= ((uint64_t*)((base_ptr) + ((offset) ^ 0x20)))[0]; \
        lo ^= ((uint64_t*)((base_ptr) + ((offset) ^ 0x20)))[1]; \
        const __m128i chunk3 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30))); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, _b1)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30)), _mm_add_epi64(chunk2, _a)); \
    }


#ifndef NOINLINE
#ifdef __GNUC__
#define NOINLINE __attribute__ ((noinline))
#elif _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif
#endif

#include "variant4_random_math.h"

#define VARIANT4_RANDOM_MATH_INIT(part) \
  uint32_t r##part[9]; \
  struct V4_Instruction code##part[256]; \
  { \
    r##part[0] = (uint32_t)(h##part[12]); \
    r##part[1] = (uint32_t)(h##part[12] >> 32); \
    r##part[2] = (uint32_t)(h##part[13]); \
    r##part[3] = (uint32_t)(h##part[13] >> 32); \
  } \
  v4_random_math_init(code##part, ctx[part]->height);

#define VARIANT4_RANDOM_MATH(part, al, ah, cl, bx0, bx1) \
  { \
    cl ^= (r##part[0] + r##part[1]) | ((uint64_t)(r##part[2] + r##part[3]) << 32); \
    r##part[4] = (uint32_t)(al); \
    r##part[5] = (uint32_t)(ah); \
    r##part[6] = (uint32_t)(_mm_cvtsi128_si32(bx0)); \
    r##part[7] = (uint32_t)(_mm_cvtsi128_si32(bx1)); \
    r##part[8] = (uint32_t)(_mm_cvtsi128_si32(_mm_srli_si128(bx1, 8))); \
    v4_random_math(code##part, r##part); \
  }

#endif /* XMRIG_CRYPTONIGHT_MONERO_H */
