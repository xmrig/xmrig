/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018      SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
    if (IS_V1) { \
        tweak1_2_##part = (*reinterpret_cast<const uint64_t*>(input + 35 + part * size) ^ \
                          *(reinterpret_cast<const uint64_t*>(ctx[part]->state) + 24)); \
    }
#else
#   define VARIANT1_INIT(part) \
    uint64_t tweak1_2_##part = 0; \
    if (IS_V1) { \
        memcpy(&tweak1_2_##part, input + 35 + part * size, sizeof tweak1_2_##part); \
        tweak1_2_##part ^= *(reinterpret_cast<const uint64_t*>(ctx[part]->state) + 24); \
    }
#endif

#define VARIANT1_1(p) \
    if (IS_V1) { \
        const uint8_t tmp = reinterpret_cast<const uint8_t*>(p)[11]; \
        static const uint32_t table = 0x75310; \
        const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1; \
        ((uint8_t*)(p))[11] = tmp ^ ((table >> index) & 0x30); \
    }

#define VARIANT1_2(p, part) \
    if (IS_V1) { \
        (p) ^= tweak1_2_##part; \
    }


#ifndef XMRIG_ARM
#   define VARIANT2_INIT(part) \
    __m128i division_result_xmm_##part = _mm_cvtsi64_si128(h##part[12]); \
    __m128i sqrt_result_xmm_##part = _mm_cvtsi64_si128(h##part[13]);

#ifdef _MSC_VER
#   define VARIANT2_SET_ROUNDING_MODE() if (VARIANT == xmrig::VARIANT_2) { _control87(RC_DOWN, MCW_RC); }
#else
#   define VARIANT2_SET_ROUNDING_MODE() if (VARIANT == xmrig::VARIANT_2) { std::fesetround(FE_DOWNWARD); }
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

#   define VARIANT2_SHUFFLE(base_ptr, offset, _a, _b, _b1) \
    do { \
        const __m128i chunk1 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10))); \
        const __m128i chunk2 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20))); \
        const __m128i chunk3 = _mm_load_si128((__m128i *)((base_ptr) + ((offset) ^ 0x30))); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, _b1)); \
        _mm_store_si128((__m128i *)((base_ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b)); \
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

#   define VARIANT2_SHUFFLE(base_ptr, offset, _a, _b, _b1) \
    do { \
        const uint64x2_t chunk1 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x10))); \
        const uint64x2_t chunk2 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x20))); \
        const uint64x2_t chunk3 = vld1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x30))); \
        vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x10)), vaddq_u64(chunk3, vreinterpretq_u64_u8(_b1))); \
        vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x20)), vaddq_u64(chunk1, vreinterpretq_u64_u8(_b))); \
        vst1q_u64((uint64_t*)((base_ptr) + ((offset) ^ 0x30)), vaddq_u64(chunk2, vreinterpretq_u64_u8(_a))); \
    } while (0)
#endif
#endif /* XMRIG_CRYPTONIGHT_MONERO_H */
