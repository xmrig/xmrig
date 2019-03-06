/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018      SChernykh   <https://github.com/SChernykh>
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

// VARIANT ALTERATIONS
#ifndef XMRIG_ARM
#   define VARIANT1_INIT(part) \
    uint64_t tweak1_2_##part = 0; \
    if (BASE == xmrig::VARIANT_1) { \
        tweak1_2_##part = (*reinterpret_cast<const uint64_t*>(input + 35 + part * size) ^ \
                          *(reinterpret_cast<const uint64_t*>(ctx[part]->state) + 24)); \
    }
#else
#   define VARIANT1_INIT(part) \
    uint64_t tweak1_2_##part = 0; \
    if (BASE == xmrig::VARIANT_1) { \
        memcpy(&tweak1_2_##part, input + 35 + part * size, sizeof tweak1_2_##part); \
        tweak1_2_##part ^= *(reinterpret_cast<const uint64_t*>(ctx[part]->state) + 24); \
    }
#endif

#define VARIANT1_1(p) \
    if (BASE == xmrig::VARIANT_1) { \
        const uint8_t tmp = reinterpret_cast<const uint8_t*>(p)[11]; \
        static const uint32_t table = 0x75310; \
        const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1; \
        ((uint8_t*)(p))[11] = tmp ^ ((table >> index) & 0x30); \
    }

#define VARIANT1_2(p, part) \
    if (BASE == xmrig::VARIANT_1) { \
        (p) ^= tweak1_2_##part; \
    }


#ifndef XMRIG_ARM
#   define VARIANT2_INIT(part) \
    __m128i division_result_xmm_##part = _mm_cvtsi64_si128(h##part[12]); \
    __m128i sqrt_result_xmm_##part = _mm_cvtsi64_si128(h##part[13]);

#ifdef _MSC_VER
#   define VARIANT2_SET_ROUNDING_MODE() if (BASE == xmrig::VARIANT_2) { _control87(RC_DOWN, MCW_RC); }
#else
#   define VARIANT2_SET_ROUNDING_MODE() if (BASE == xmrig::VARIANT_2) { fesetround(FE_DOWNWARD); }
#endif

#   define VARIANT2_INTEGER_MATH(part, cl, cx) \
    do { \
        const uint64_t sqrt_result = static_cast<uint64_t>(_mm_cvtsi128_si64(sqrt_result_xmm_##part)); \
        const uint64_t cx_0 = _mm_cvtsi128_si64(cx); \
        cl ^= static_cast<uint64_t>(_mm_cvtsi128_si64(division_result_xmm_##part)) ^ (sqrt_result << 32); \
        const uint32_t d = static_cast<uint32_t>(cx_0 + (sqrt_result << 1)) | 0x80000001UL; \
        const uint64_t cx_1 = _mm_cvtsi128_si64(_mm_srli_si128(cx, 8)); \
        const uint64_t division_result = static_cast<uint32_t>(cx_1 / d) + ((cx_1 % d) << 32); \
        division_result_xmm_##part = _mm_cvtsi64_si128(static_cast<int64_t>(division_result)); \
        sqrt_result_xmm_##part = int_sqrt_v2(cx_0 + division_result); \
    } while (0)

#   define VARIANT2_SHUFFLE(base_ptr, offset, _a, _b, _b1, _c, reverse) \
    do { \
        const __m128i chunk1 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ (reverse ? 0x30 : 0x10)))); \
        const __m128i chunk2 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20))); \
        const __m128i chunk3 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ (reverse ? 0x10 : 0x30)))); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, _b1)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30)), _mm_add_epi64(chunk2, _a)); \
        if (VARIANT == xmrig::VARIANT_4) { \
            _c = _mm_xor_si128(_mm_xor_si128(_c, chunk3), _mm_xor_si128(chunk1, chunk2)); \
        } \
    } while (0)

#   define VARIANT2_SHUFFLE2(base_ptr, offset, _a, _b, _b1, hi, lo, reverse) \
    do { \
        const __m128i chunk1 = _mm_xor_si128(_mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10))), _mm_set_epi64x(lo, hi)); \
        const __m128i chunk2 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20))); \
        hi ^= ((uint64_t*)((base_ptr) + ((offset) ^ 0x20)))[0]; \
        lo ^= ((uint64_t*)((base_ptr) + ((offset) ^ 0x20)))[1]; \
        const __m128i chunk3 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30))); \
        if (reverse) { \
            _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk1, _b1)); \
            _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk3, _b)); \
        } else { \
            _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, _b1)); \
            _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b)); \
        } \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30)), _mm_add_epi64(chunk2, _a)); \
    } while (0)

#else
#   define VARIANT2_INIT(part) \
    uint64_t division_result_##part = h##part[12]; \
    uint64_t sqrt_result_##part = h##part[13];

#   define VARIANT2_INTEGER_MATH(part, cl, cx) \
    do { \
        const uint64_t cx_0 = _mm_cvtsi128_si64(cx); \
        cl ^= division_result_##part ^ (sqrt_result_##part << 32); \
        const uint32_t d = static_cast<uint32_t>(cx_0 + (sqrt_result_##part << 1)) | 0x80000001UL; \
        const uint64_t cx_1 = _mm_cvtsi128_si64(_mm_srli_si128(cx, 8)); \
        division_result_##part = static_cast<uint32_t>(cx_1 / d) + ((cx_1 % d) << 32); \
        const uint64_t sqrt_input = cx_0 + division_result_##part; \
        sqrt_result_##part = sqrt(sqrt_input + 18446744073709551616.0) * 2.0 - 8589934592.0; \
        const uint64_t s = sqrt_result_##part >> 1; \
        const uint64_t b = sqrt_result_##part & 1; \
        const uint64_t r2 = (uint64_t)(s) * (s + b) + (sqrt_result_##part << 32); \
        sqrt_result_##part += ((r2 + b > sqrt_input) ? -1 : 0) + ((r2 + (1ULL << 32) < sqrt_input - s) ? 1 : 0); \
    } while (0)

#   define VARIANT2_SHUFFLE(base_ptr, offset, _a, _b, _b1, _c, reverse) \
    do { \
        const uint64x2_t chunk1 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ (reverse ? 0x30 : 0x10)))); \
        const uint64x2_t chunk2 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x20))); \
        const uint64x2_t chunk3 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ (reverse ? 0x10 : 0x30)))); \
        vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x10)), vaddq_u64(chunk3, vreinterpretq_u64_u8(_b1))); \
        vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x20)), vaddq_u64(chunk1, vreinterpretq_u64_u8(_b))); \
        vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x30)), vaddq_u64(chunk2, vreinterpretq_u64_u8(_a))); \
        if (VARIANT == xmrig::VARIANT_4) { \
            _c = veorq_u64(veorq_u64(_c, chunk3), veorq_u64(chunk1, chunk2)); \
        } \
    } while (0)

#   define VARIANT2_SHUFFLE2(base_ptr, offset, _a, _b, _b1, hi, lo, reverse) \
    do { \
        const uint64x2_t chunk1 = veorq_u64(vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x10))), vcombine_u64(vcreate_u64(hi), vcreate_u64(lo))); \
        const uint64x2_t chunk2 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x20))); \
        hi ^= ((uint64_t*)((base_ptr) + ((offset) ^ 0x20)))[0]; \
        lo ^= ((uint64_t*)((base_ptr) + ((offset) ^ 0x20)))[1]; \
        const uint64x2_t chunk3 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x30))); \
        if (reverse) { \
            vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x10)), vaddq_u64(chunk1, vreinterpretq_u64_u8(_b1))); \
            vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x20)), vaddq_u64(chunk3, vreinterpretq_u64_u8(_b))); \
        } else { \
            vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x10)), vaddq_u64(chunk3, vreinterpretq_u64_u8(_b1))); \
            vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x20)), vaddq_u64(chunk1, vreinterpretq_u64_u8(_b))); \
        } \
        vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x30)), vaddq_u64(chunk2, vreinterpretq_u64_u8(_a))); \
    } while (0)
#endif

#define SWAP32LE(x) x
#define SWAP64LE(x) x
#define hash_extra_blake(data, length, hash) blake256_hash((uint8_t*)(hash), (uint8_t*)(data), (length))

#ifndef NOINLINE
#ifdef __GNUC__
#define NOINLINE __attribute__ ((noinline))
#elif _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif
#endif

#include "common/xmrig.h"
#include "variant4_random_math.h"

#define VARIANT4_RANDOM_MATH_INIT(part) \
  uint32_t r##part[9]; \
  struct V4_Instruction code##part[256]; \
  if ((VARIANT == xmrig::VARIANT_WOW) || (VARIANT == xmrig::VARIANT_4)) { \
    r##part[0] = (uint32_t)(h##part[12]); \
    r##part[1] = (uint32_t)(h##part[12] >> 32); \
    r##part[2] = (uint32_t)(h##part[13]); \
    r##part[3] = (uint32_t)(h##part[13] >> 32); \
  } \
  v4_random_math_init<VARIANT>(code##part, height);

#define VARIANT4_RANDOM_MATH(part, al, ah, cl, bx0, bx1) \
  if ((VARIANT == xmrig::VARIANT_WOW) || (VARIANT == xmrig::VARIANT_4)) { \
    cl ^= (r##part[0] + r##part[1]) | ((uint64_t)(r##part[2] + r##part[3]) << 32); \
    r##part[4] = static_cast<uint32_t>(al); \
    r##part[5] = static_cast<uint32_t>(ah); \
    r##part[6] = static_cast<uint32_t>(_mm_cvtsi128_si32(bx0)); \
    r##part[7] = static_cast<uint32_t>(_mm_cvtsi128_si32(bx1)); \
    r##part[8] = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(bx1, 8))); \
    v4_random_math(code##part, r##part); \
  }

#endif /* XMRIG_CRYPTONIGHT_MONERO_H */
