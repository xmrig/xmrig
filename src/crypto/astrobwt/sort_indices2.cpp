/* XMRig
 * Copyright (c) 2018      Lee Clagett              <https://github.com/vtnerd>
 * Copyright (c) 2018-2019 tevador                  <tevador@gmail.com>
 * Copyright (c) 2000      Transmeta Corporation    <https://github.com/intel/msr-tools>
 * Copyright (c) 2004-2008 H. Peter Anvin           <https://github.com/intel/msr-tools>
 * Copyright (c) 2018-2021 SChernykh                <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig                    <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/astrobwt/sort_indices2.h"
#include "base/tools/bswap_64.h"
#include <cstring>


#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#define RESTRICT __restrict__
#elif _MSC_VER
#define NOINLINE __declspec(noinline)
#define RESTRICT __restrict
#else
#define NOINLINE
#define RESTRICT
#endif


#if __has_cpp_attribute(unlikely)
#define UNLIKELY(X) (X) [[unlikely]]
#elif defined __GNUC__
#define UNLIKELY(X) (__builtin_expect((X), 0))
#else
#define UNLIKELY(X) (X)
#endif


static NOINLINE void fix(const uint8_t* RESTRICT v, uint32_t* RESTRICT indices, int32_t i)
{
    uint32_t prev_t = indices[i - 1];
    uint32_t t = indices[i];

    const uint32_t data_a = bswap_32(*(const uint32_t*)(v + (t & 0xFFFF) + 2));
    if (data_a < bswap_32(*(const uint32_t*)(v + (prev_t & 0xFFFF) + 2)))
    {
        const uint32_t t2 = prev_t;
        int32_t j = i - 1;
        do
        {
            indices[j + 1] = prev_t;
            --j;

            if (j < 0) {
                break;
            }

            prev_t = indices[j];
        } while (((t ^ prev_t) <= 0xFFFF) && (data_a < bswap_32(*(const uint32_t*)(v + (prev_t & 0xFFFF) + 2))));
        indices[j + 1] = t;
        t = t2;
    }
}


static NOINLINE void sort_indices(uint32_t N, const uint8_t* RESTRICT v, uint32_t* RESTRICT indices, uint32_t* RESTRICT tmp_indices)
{
    uint8_t byte_counters[2][256] = {};
    uint32_t counters[2][256];

    {
#define ITER(X) ++byte_counters[1][v[i + X]];

        enum { unroll = 12 };

        uint32_t i = 0;
        const uint32_t n = N - (unroll - 1);
        for (; i < n; i += unroll) {
            ITER(0); ITER(1); ITER(2); ITER(3); ITER(4); ITER(5); ITER(6); ITER(7); ITER(8); ITER(9); ITER(10); ITER(11);
        }
        for (; i < N; ++i) {
            ITER(0);
        }
        memcpy(&byte_counters[0], &byte_counters[1], 256);
        --byte_counters[0][v[0]];

#undef ITER
    }

    {
        uint32_t c0 = byte_counters[0][0];
        uint32_t c1 = byte_counters[1][0] - 1;
        counters[0][0] = c0;
        counters[1][0] = c1;
        uint8_t* src = &byte_counters[0][0] + 1;
        uint32_t* dst = &counters[0][0] + 1;
        const uint8_t* const e = &byte_counters[0][0] + 256;
        do {
            c0 += src[0];
            c1 += src[256];
            dst[0] = c0;
            dst[256] = c1;
            ++src;
            ++dst;
        } while (src < e);
    }

    {
#define ITER(X) \
        do { \
            const uint32_t byte0 = v[i - X + 0]; \
            const uint32_t byte1 = v[i - X + 1]; \
            tmp_indices[counters[0][byte1]--] = (byte0 << 24) | (byte1 << 16) | (i - X); \
        } while (0)

        enum { unroll = 8 };

        uint32_t i = N;
        for (; i >= unroll; i -= unroll) {
            ITER(1); ITER(2); ITER(3); ITER(4); ITER(5); ITER(6); ITER(7); ITER(8);
        }
        for (; i > 0; --i) {
            ITER(1);
        }

#undef ITER
    }

    {
#define ITER(X) \
        do { \
            const uint32_t data = tmp_indices[i - X]; \
            indices[counters[1][data >> 24]--] = data; \
        } while (0)

        enum { unroll = 8 };

        uint32_t i = N;
        for (; i >= unroll; i -= unroll) {
            ITER(1); ITER(2); ITER(3); ITER(4); ITER(5); ITER(6); ITER(7); ITER(8);
        }
        for (; i > 0; --i) {
            ITER(1);
        }

#undef ITER
    }

    {
#define ITER(X) do { if UNLIKELY(a[X * 2] == a[(X + 1) * 2]) fix(v, indices, i + X); } while (0)

        enum { unroll = 16 };

        uint32_t i = 1;
        const uint32_t n = N - (unroll - 1);
        const uint16_t* a = ((const uint16_t*)indices) + 1;

        for (; i < n; i += unroll, a += unroll * 2) {
            ITER(0); ITER(1); ITER(2); ITER(3); ITER(4); ITER(5); ITER(6); ITER(7);
            ITER(8); ITER(9); ITER(10); ITER(11); ITER(12); ITER(13); ITER(14); ITER(15);
        }
        for (; i < N; ++i, a += 2) {
            ITER(0);
        }

#undef ITER
    }

    {
#define ITER(X) a[X] = b[X * 2];

        enum { unroll = 32 };

        uint16_t* a = (uint16_t*)indices;
        uint16_t* b = (uint16_t*)indices;
        uint16_t* e = ((uint16_t*)indices) + (N - (unroll - 1));

        for (; a < e; a += unroll, b += unroll * 2) {
            ITER(0); ITER(1); ITER(2); ITER(3); ITER(4); ITER(5); ITER(6); ITER(7);
            ITER(8); ITER(9); ITER(10); ITER(11); ITER(12); ITER(13); ITER(14); ITER(15);
            ITER(16); ITER(17); ITER(18); ITER(19); ITER(20); ITER(21); ITER(22); ITER(23);
            ITER(24); ITER(25); ITER(26); ITER(27); ITER(28); ITER(29); ITER(30); ITER(31);
        }

        e = ((uint16_t*)indices) + N;
        for (; a < e; ++a, b += 2) {
            ITER(0);
        }

#undef ITER
    }
}


void sort_indices_astrobwt_v2(uint32_t N, const uint8_t* v, uint32_t* indices, uint32_t* tmp_indices)
{
    sort_indices(N, v, indices, tmp_indices);
}
