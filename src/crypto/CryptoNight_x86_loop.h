/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018      aegroto
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

#ifndef __CRYPTONIGHT_X86_LOOP_H__
#define __CRYPTONIGHT_X86_LOOP_H__

#define SINGLEHASH_LOOP_COMMON \
    _mm_store_si128((__m128i *) memoryPointer, _mm_xor_si128(bx0, cx)); \
    VARIANT1_1(memoryPointer); \
    idx0 = EXTRACT64(cx); \
    memoryPointer = ((uint8_t*) l0) + ((idx0) & MASK); \
    bx0 = cx; \
    uint64_t hi, lo, cl, ch; \
    cl = ((uint64_t*) memoryPointer)[0]; \
    ch = ((uint64_t*) memoryPointer)[1]; \
    lo = __umul128(idx0, cl, &hi); \
    al0 += hi; \
    ah0 += lo; \
    VARIANT1_2(ah0, 0); \
    ((uint64_t*) memoryPointer)[0] = al0; \
    ((uint64_t*) memoryPointer)[1] = ah0; \
    VARIANT1_2(ah0, 0); \
    ah0 ^= ch; \
    al0 ^= cl; \
    memoryPointer = ((uint8_t*) l0) + ((al0) & MASK); 

#define SINGLEHASH_LOOP_CNHEAVY \
    int64_t n  = ((int64_t*)memoryPointer)[0]; \
    int32_t d  = ((int32_t*)memoryPointer)[2]; \
    int64_t q = n / (d | 0x5); \
    ((int64_t*) memoryPointer)[0] = n ^ q; 

#define SINGLEHASH_LOOP_SOFTAES \
    cx = soft_aesenc((uint32_t*) memoryPointer, _mm_set_epi64x(ah0, al0)); 

#define SINGLEHASH_LOOP_HARDAES \
    cx = _mm_load_si128((__m128i *) memoryPointer); \
    cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah0, al0));

#endif /* __CRYPTONIGHT_X86_LOOP_H__ */