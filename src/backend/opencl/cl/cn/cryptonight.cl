/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

/* For Mesa clover support */
#ifdef cl_clang_storage_class_specifiers
#   pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif


#include "algorithm.cl"
#include "wolf-aes.cl"
#include "wolf-skein.cl"
#include "jh.cl"
#include "blake256.cl"
#include "groestl256.cl"
#include "fast_int_math_v2.cl"
#include "fast_div_heavy.cl"
#include "keccak.cl"


#if defined(__NV_CL_C_VERSION) && STRIDED_INDEX != 0
#   undef STRIDED_INDEX
#   define STRIDED_INDEX 0
#endif


#define MEM_CHUNK (1 << MEM_CHUNK_EXPONENT)


#if (STRIDED_INDEX == 0)
#   define IDX(x) (x)
#elif (STRIDED_INDEX == 1)
#   if (ALGO_FAMILY == FAMILY_CN_HEAVY)
#       define IDX(x) ((x) * WORKSIZE)
#   else
#       define IDX(x) mul24((x), Threads)
#   endif
#elif (STRIDED_INDEX == 2)
#   define IDX(x) (((x) % MEM_CHUNK) + ((x) / MEM_CHUNK) * WORKSIZE * MEM_CHUNK)
#endif


inline ulong getIdx()
{
    return get_global_id(0) - get_global_offset(0);
}


#define mix_and_propagate(xin) (xin)[(get_local_id(1)) % 8][get_local_id(0)] ^ (xin)[(get_local_id(1) + 1) % 8][get_local_id(0)]


__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void cn0(__global ulong *input, int inlen, __global uint4 *Scratchpad, __global ulong *states, uint Threads)
{
    uint ExpandedKey1[40];
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];
    uint4 text;

    const uint gIdx = getIdx();

    for (int i = get_local_id(1) * 8 + get_local_id(0); i < 256; i += 8 * 8) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local ulong State_buf[8 * 25];

    {
        states += 25 * gIdx;

#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO_FAMILY == FAMILY_CN_HEAVY)
            Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + (gIdx % WORKSIZE);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * (gIdx % WORKSIZE);
#       endif

        if (get_local_id(1) == 0) {
            __local ulong* State = State_buf + get_local_id(0) * 25;

            #pragma unroll
            for (int i = 0; i < 25; ++i) {
                State[i] = 0;
            }

            // Input length must be a multiple of 136 and padded on the host side
            for (int i = 0; inlen > 0; i += 17, inlen -= 136) {
                #pragma unroll
                for (int j = 0; j < 17; ++j) {
                    State[j] ^= input[i + j];
                }
                if (i == 0) {
                    ((__local uint *)State)[9]  &= 0x00FFFFFFU;
                    ((__local uint *)State)[9]  |= (((uint)get_global_id(0)) & 0xFF) << 24;
                    ((__local uint *)State)[10] &= 0xFF000000U;
                    ((__local uint *)State)[10] |= (((uint)get_global_id(0) >> 8));
                }
                keccakf1600_2(State);
            }

            #pragma unroll 1
            for (int i = 0; i < 25; ++i) {
                states[i] = State[i];
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    {
        text = vload4(get_local_id(1) + 4, (__global uint *)(states));

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ((ulong *)ExpandedKey1)[i] = states[i];
        }

        AESExpandKey256(ExpandedKey1);
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (ALGO_FAMILY == FAMILY_CN_HEAVY)
    __local uint4 xin[8][8];

    /* Also left over threads perform this loop.
     * The left over thread results will be ignored
     */
    #pragma unroll 16
    for (size_t i = 0; i < 16; i++) {
        #pragma unroll 10
        for (int j = 0; j < 10; ++j) {
            uint4 t = ((uint4 *)ExpandedKey1)[j];
            t.s0 ^= AES0[BYTE(text.s0, 0)] ^ AES1[BYTE(text.s1, 1)] ^ AES2[BYTE(text.s2, 2)] ^ AES3[BYTE(text.s3, 3)];
            t.s1 ^= AES0[BYTE(text.s1, 0)] ^ AES1[BYTE(text.s2, 1)] ^ AES2[BYTE(text.s3, 2)] ^ AES3[BYTE(text.s0, 3)];
            t.s2 ^= AES0[BYTE(text.s2, 0)] ^ AES1[BYTE(text.s3, 1)] ^ AES2[BYTE(text.s0, 2)] ^ AES3[BYTE(text.s1, 3)];
            t.s3 ^= AES0[BYTE(text.s3, 0)] ^ AES1[BYTE(text.s0, 1)] ^ AES2[BYTE(text.s1, 2)] ^ AES3[BYTE(text.s2, 3)];
            text = t;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        xin[get_local_id(1)][get_local_id(0)] = text;
        barrier(CLK_LOCAL_MEM_FENCE);
        text = mix_and_propagate(xin);
    }
#   endif

    {
        const uint local_id1 = get_local_id(1);
        #pragma unroll 2
        for(uint i = 0; i < (MEMORY >> 4); i += 8) {
            #pragma unroll 10
            for (uint j = 0; j < 10; ++j) {
                uint4 t = ((uint4 *)ExpandedKey1)[j];
                t.s0 ^= AES0[BYTE(text.s0, 0)] ^ AES1[BYTE(text.s1, 1)] ^ AES2[BYTE(text.s2, 2)] ^ AES3[BYTE(text.s3, 3)];
                t.s1 ^= AES0[BYTE(text.s1, 0)] ^ AES1[BYTE(text.s2, 1)] ^ AES2[BYTE(text.s3, 2)] ^ AES3[BYTE(text.s0, 3)];
                t.s2 ^= AES0[BYTE(text.s2, 0)] ^ AES1[BYTE(text.s3, 1)] ^ AES2[BYTE(text.s0, 2)] ^ AES3[BYTE(text.s1, 3)];
                t.s3 ^= AES0[BYTE(text.s3, 0)] ^ AES1[BYTE(text.s0, 1)] ^ AES2[BYTE(text.s1, 2)] ^ AES3[BYTE(text.s2, 3)];
                text = t;
            }

            Scratchpad[IDX(i + local_id1)] = text;
        }
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);
}


#if (ALGO_BASE == ALGO_CN_0)
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1(__global ulong *input, __global uint4 *Scratchpad, __global ulong *states, uint Threads)
{
    ulong a[2], b[2];
    __local uint AES0[256], AES1[256];

    const ulong gIdx = getIdx();

    for (int i = get_local_id(0); i < 256; i += WORKSIZE) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint4 b_x;

    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO_FAMILY == FAMILY_CN_HEAVY)
            Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + get_local_id(0);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#       endif

        a[0] = states[0] ^ states[4];
        b[0] = states[2] ^ states[6];
        a[1] = states[1] ^ states[5];
        b[1] = states[3] ^ states[7];

        b_x = ((uint4 *)b)[0];
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    {
        uint idx0 = a[0];

#       if (ALGO == ALGO_CN_CCX)
        float4 conc_var = (float4)(0.0f);
        const uint4 conc_t = (uint4)(0x807FFFFFU);
        const uint4 conc_u = (uint4)(0x40000000U);
        const uint4 conc_v = (uint4)(0x4DFFFFFFU);
#       endif

        #pragma unroll CN_UNROLL
        for (int i = 0; i < ITERATIONS; ++i) {
            ulong c[2];

            ((uint4 *)c)[0] = Scratchpad[IDX((idx0 & MASK) >> 4)];

#           if (ALGO == ALGO_CN_CCX)
            {
                float4 r = convert_float4_rte(((int4 *)c)[0]) + conc_var;
                r = r * r * r;
                r = as_float4((as_uint4(r) & conc_t) | conc_u);

                float4 c_old = conc_var;
                conc_var += r;

                c_old = as_float4((as_uint4(c_old) & conc_t) | conc_u);

                ((int4 *)c)[0] ^= convert_int4_rtz(c_old * as_float4(conc_v));
            }
#           endif

            ((uint4 *)c)[0] = AES_Round_Two_Tables(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);

            Scratchpad[IDX((idx0 & MASK) >> 4)] = b_x ^ ((uint4 *)c)[0];

            uint4 tmp;
            tmp = Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)];

            a[1] += c[0] * as_ulong2(tmp).s0;
            a[0] += mul_hi(c[0], as_ulong2(tmp).s0);

            Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)] = ((uint4 *)a)[0];

            ((uint4 *)a)[0] ^= tmp;
            idx0 = a[0];

            b_x = ((uint4 *)c)[0];

#           if (ALGO_FAMILY == FAMILY_CN_HEAVY)
            {
                const long2 n = *((__global long2*)(Scratchpad + (IDX((idx0 & MASK) >> 4))));
                long q = fast_div_heavy(n.s0, as_int4(n).s2 | 0x5);
                *((__global long*)(Scratchpad + (IDX((idx0 & MASK) >> 4)))) = n.s0 ^ q;

#               if (ALGO == ALGO_CN_HEAVY_XHV)
                idx0 = (~as_int4(n).s2) ^ q;
#               else
                idx0 = as_int4(n).s2 ^ q;
#               endif
            }
#           endif
        }
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);
}
#elif (ALGO_BASE == ALGO_CN_1)
#define VARIANT1_1(p) \
        uint table = 0x75310U; \
        uint index = (((p).s2 >> 26) & 12) | (((p).s2 >> 23) & 2); \
        (p).s2 ^= ((table >> index) & 0x30U) << 24


#define VARIANT1_2(p) ((uint2 *)&(p))[0] ^= tweak1_2_0


#define VARIANT1_INIT() \
        tweak1_2 = as_uint2(input[4]); \
        tweak1_2.s0 >>= 24; \
        tweak1_2.s0 |= tweak1_2.s1 << 8; \
        tweak1_2.s1 = (uint) get_global_id(0); \
        tweak1_2 ^= as_uint2(states[24])


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1(__global ulong *input, __global uint4 *Scratchpad, __global ulong *states, uint Threads)
{
    ulong a[2], b[2];
    __local uint AES0[256], AES1[256];

    const ulong gIdx = getIdx();

    for (int i = get_local_id(0); i < 256; i += WORKSIZE) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint2 tweak1_2;
    uint4 b_x;

    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO_FAMILY == FAMILY_CN_HEAVY)
            Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + get_local_id(0);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#       endif

        a[0] = states[0] ^ states[4];
        b[0] = states[2] ^ states[6];
        a[1] = states[1] ^ states[5];
        b[1] = states[3] ^ states[7];

        b_x = ((uint4 *)b)[0];
        VARIANT1_INIT();
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    {
#       if (ALGO == ALGO_CN_HEAVY_TUBE)
        uint idx0 = a[0];
#           define IDX_0 idx0
#       else
#           define IDX_0 as_uint2(a[0]).s0
#       endif

        #pragma unroll CN_UNROLL
        for (int i = 0; i < ITERATIONS; ++i) {
            ulong c[2];

            ((uint4 *)c)[0] = Scratchpad[IDX((IDX_0 & MASK) >> 4)];

#           if (ALGO == ALGO_CN_HEAVY_TUBE)
            ((uint4 *)c)[0] = AES_Round_bittube2(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);
#           else
            ((uint4 *)c)[0] = AES_Round_Two_Tables(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);
#           endif

            b_x ^= ((uint4 *)c)[0];
            VARIANT1_1(b_x);

            Scratchpad[IDX((IDX_0 & MASK) >> 4)] = b_x;

            uint4 tmp;
            tmp = Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)];

            a[1] += c[0] * as_ulong2(tmp).s0;
            a[0] += mul_hi(c[0], as_ulong2(tmp).s0);

            uint2 tweak1_2_0 = tweak1_2;
#           if (ALGO == ALGO_CN_RTO || ALGO == ALGO_CN_HEAVY_TUBE)
            tweak1_2_0 ^= ((uint2 *)&(a[0]))[0];
#           endif

            VARIANT1_2(a[1]);
            Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)] = ((uint4 *)a)[0];
            VARIANT1_2(a[1]);

            ((uint4 *)a)[0] ^= tmp;

#           if (ALGO == ALGO_CN_HEAVY_TUBE)
            idx0 = a[0];
#           endif

            b_x = ((uint4 *)c)[0];

#           if (ALGO == ALGO_CN_HEAVY_TUBE)
            {
                const long2 n = *((__global long2*)(Scratchpad + (IDX((idx0 & MASK) >> 4))));
                long q = fast_div_heavy(n.s0, as_int4(n).s2 | 0x5);
                *((__global long*)(Scratchpad + (IDX((idx0 & MASK) >> 4)))) = n.s0 ^ q;
                idx0 = as_int4(n).s2 ^ q;
            }
#           endif
        }
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

#undef IDX_0
#elif (ALGO_BASE == ALGO_CN_2)
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1(__global ulong *input, __global uint4 *Scratchpad, __global ulong *states, uint Threads)
{
    ulong a[2], b[4];
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];

    const ulong gIdx = getIdx();

    for(int i = get_local_id(0); i < 256; i += WORKSIZE)
    {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        states += 25 * gIdx;

#       if defined(__NV_CL_C_VERSION)
            Scratchpad += gIdx * (ITERATIONS >> 2);
#       else
#           if (STRIDED_INDEX == 0)
                Scratchpad += gIdx * (MEMORY >> 4);
#           elif (STRIDED_INDEX == 1)
                Scratchpad += gIdx;
#           elif (STRIDED_INDEX == 2)
                Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#           endif
#       endif

        a[0] = states[0] ^ states[4];
        a[1] = states[1] ^ states[5];

        b[0] = states[2] ^ states[6];
        b[1] = states[3] ^ states[7];
        b[2] = states[8] ^ states[10];
        b[3] = states[9] ^ states[11];
    }

    ulong2 bx0 = ((ulong2 *)b)[0];
    ulong2 bx1 = ((ulong2 *)b)[1];

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   ifdef __NV_CL_C_VERSION
        __local uint16 scratchpad_line_buf[WORKSIZE];
        __local uint16* scratchpad_line = scratchpad_line_buf + get_local_id(0);
#       define SCRATCHPAD_CHUNK(N) (*(__local uint4*)((__local uchar*)(scratchpad_line) + (idx1 ^ (N << 4))))
#   else
#       if (STRIDED_INDEX == 0)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (idx ^ (N << 4))))
#       elif (STRIDED_INDEX == 1)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + mul24(as_uint(idx ^ (N << 4)), Threads)))
#       elif (STRIDED_INDEX == 2)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (((idx ^ (N << 4)) % (MEM_CHUNK << 4)) + ((idx ^ (N << 4)) / (MEM_CHUNK << 4)) * WORKSIZE * (MEM_CHUNK << 4))))
#       endif
#   endif

    {
    uint2 division_result = as_uint2(states[12]);
    uint sqrt_result = as_uint2(states[13]).s0;

    #pragma unroll CN_UNROLL
    for (int i = 0; i < ITERATIONS; ++i) {
#       ifdef __NV_CL_C_VERSION
            uint idx  = a[0] & 0x1FFFC0;
            uint idx1 = a[0] & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            uint idx = a[0] & MASK;
#       endif

        uint4 c = SCRATCHPAD_CHUNK(0);
        c = AES_Round(AES0, AES1, AES2, AES3, c, ((uint4 *)a)[0]);

        {
#           if (ALGO == ALGO_CN_RWZ)
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(3));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(1));
#           else
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));
#           endif

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
        }

        SCRATCHPAD_CHUNK(0) = as_uint4(bx0) ^ c;

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;

            idx = as_ulong2(c).s0 & 0x1FFFC0;
            idx1 = as_ulong2(c).s0 & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            idx = as_ulong2(c).s0 & MASK;
#       endif

        uint4 tmp = SCRATCHPAD_CHUNK(0);

        {
            tmp.s0 ^= division_result.s0;
            tmp.s1 ^= division_result.s1 ^ sqrt_result;

            division_result = fast_div_v2(as_ulong2(c).s1, (c.s0 + (sqrt_result << 1)) | 0x80000001UL);
            sqrt_result = fast_sqrt_v2(as_ulong2(c).s0 + as_ulong(division_result));
        }

        ulong2 t;
        t.s0 = mul_hi(as_ulong2(c).s0, as_ulong2(tmp).s0);
        t.s1 = as_ulong2(c).s0 * as_ulong2(tmp).s0;
        {
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1)) ^ t;
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            t ^= chunk2;
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

#           if (ALGO == ALGO_CN_RWZ)
            SCRATCHPAD_CHUNK(1) = as_uint4(chunk1 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk3 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
#           else
            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
#           endif
        }

        a[1] += t.s1;
        a[0] += t.s0;

        SCRATCHPAD_CHUNK(0) = ((uint4 *)a)[0];

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;
#       endif

        ((uint4 *)a)[0] ^= tmp;
        bx1 = bx0;
        bx0 = as_ulong2(c);
    }

#   undef SCRATCHPAD_CHUNK
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);
}
#endif


__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void cn2(__global uint4 *Scratchpad, __global ulong *states, __global uint *Branch0, __global uint *Branch1, __global uint *Branch2, __global uint *Branch3, uint Threads)
{
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];
    uint ExpandedKey2[40];
    uint4 text;

    const ulong gIdx = getIdx();

    for (int i = get_local_id(1) * 8 + get_local_id(0); i < 256; i += 8 * 8) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO_FAMILY == FAMILY_CN_HEAVY)
            Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + (gIdx % WORKSIZE);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * (gIdx % WORKSIZE);
#       endif

        #if defined(__Tahiti__) || defined(__Pitcairn__)

        for(int i = 0; i < 4; ++i) ((ulong *)ExpandedKey2)[i] = states[i + 4];
        text = vload4(get_local_id(1) + 4, (__global uint *)states);

        #else

        text = vload4(get_local_id(1) + 4, (__global uint *)states);
        ((uint8 *)ExpandedKey2)[0] = vload8(1, (__global uint *)states);

        #endif

        AESExpandKey256(ExpandedKey2);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#   if (ALGO_FAMILY == FAMILY_CN_HEAVY)
    __local uint4 xin1[8][8];
    __local uint4 xin2[8][8];
    __local uint4* xin1_store = &xin1[get_local_id(1)][get_local_id(0)];
    __local uint4* xin1_load = &xin1[(get_local_id(1) + 1) % 8][get_local_id(0)];
    __local uint4* xin2_store = &xin2[get_local_id(1)][get_local_id(0)];
    __local uint4* xin2_load = &xin2[(get_local_id(1) + 1) % 8][get_local_id(0)];
    *xin2_store = (uint4)(0, 0, 0, 0);
#   endif

    {
#       if (ALGO_FAMILY == FAMILY_CN_HEAVY)
        #pragma unroll 2
        for(int i = 0, i1 = get_local_id(1); i < (MEMORY >> 7); ++i, i1 = (i1 + 16) % (MEMORY >> 4))
        {
            text ^= Scratchpad[IDX(i1)];
            barrier(CLK_LOCAL_MEM_FENCE);
            text ^= *xin2_load;

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

            *xin1_store = text;

            text ^= Scratchpad[IDX(i1 + 8)];
            barrier(CLK_LOCAL_MEM_FENCE);
            text ^= *xin1_load;

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

            *xin2_store = text;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        text ^= *xin2_load;

#       else
        const uint local_id1 = get_local_id(1);
        #pragma unroll 2
        for (uint i = 0; i < (MEMORY >> 7); ++i) {
            text ^= Scratchpad[IDX((i << 3) + local_id1)];

            #pragma unroll 10
            for(uint j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);
        }
#       endif
    }

#   if (ALGO_FAMILY == FAMILY_CN_HEAVY)
    /* Also left over threads performe this loop.
     * The left over thread results will be ignored
     */
    #pragma unroll 16
    for(size_t i = 0; i < 16; i++)
    {
        #pragma unroll 10
        for (int j = 0; j < 10; ++j) {
            text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        *xin1_store = text;
        barrier(CLK_LOCAL_MEM_FENCE);
        text ^= *xin1_load;
    }
#   endif

    {
        vstore2(as_ulong2(text), get_local_id(1) + 4, states);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    __local ulong State_buf[8 * 25];

    {
        if(!get_local_id(1))
        {
            __local ulong* State = State_buf + get_local_id(0) * 25;

            for(int i = 0; i < 25; ++i) State[i] = states[i];

            keccakf1600_2(State);

            for(int i = 0; i < 25; ++i) states[i] = State[i];

            uint StateSwitch = State[0] & 3;
            __global uint *destinationBranch1 = StateSwitch == 0 ? Branch0 : Branch1;
            __global uint *destinationBranch2 = StateSwitch == 2 ? Branch2 : Branch3;
            __global uint *destinationBranch = StateSwitch < 2 ? destinationBranch1 : destinationBranch2;
            destinationBranch[atomic_inc(destinationBranch + Threads)] = gIdx;
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}


#define VSWAP8(x)   (((x) >> 56) | (((x) >> 40) & 0x000000000000FF00UL) | (((x) >> 24) & 0x0000000000FF0000UL) \
          | (((x) >>  8) & 0x00000000FF000000UL) | (((x) <<  8) & 0x000000FF00000000UL) \
          | (((x) << 24) & 0x0000FF0000000000UL) | (((x) << 40) & 0x00FF000000000000UL) | (((x) << 56) & 0xFF00000000000000UL))


#define VSWAP4(x)   ((((x) >> 24) & 0xFFU) | (((x) >> 8) & 0xFF00U) | (((x) << 8) & 0xFF0000U) | (((x) << 24) & 0xFF000000U))


__kernel void Skein(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if(idx < BranchBuf[Threads]) {
        states += 25 * BranchBuf[idx];

        // skein
        ulong8 h = vload8(0, SKEIN512_256_IV);

        // Type field begins with final bit, first bit, then six bits of type; the last 96
        // bits are input processed (including in the block to be processed with that tweak)
        // The output transform is only one run of UBI, since we need only 256 bits of output
        // The tweak for the output transform is Type = Output with the Final bit set
        // T[0] for the output is 8, and I don't know why - should be message size...
        ulong t[3] = { 0x00UL, 0x7000000000000000UL, 0x00UL };
        ulong8 p, m;

        #pragma unroll 1
        for (uint i = 0; i < 4; ++i)
        {
            t[0] += i < 3 ? 0x40UL : 0x08UL;

            t[2] = t[0] ^ t[1];

            m = (i < 3) ? vload8(i, states) : (ulong8)(states[24], 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL);
            const ulong h8 = h.s0 ^ h.s1 ^ h.s2 ^ h.s3 ^ h.s4 ^ h.s5 ^ h.s6 ^ h.s7 ^ SKEIN_KS_PARITY;
            p = Skein512Block(m, h, h8, t);

            h = m ^ p;

            t[1] = i < 2 ? 0x3000000000000000UL : 0xB000000000000000UL;
        }

        t[0] = 0x08UL;
        t[1] = 0xFF00000000000000UL;
        t[2] = t[0] ^ t[1];

        p = (ulong8)(0);
        const ulong h8 = h.s0 ^ h.s1 ^ h.s2 ^ h.s3 ^ h.s4 ^ h.s5 ^ h.s6 ^ h.s7 ^ SKEIN_KS_PARITY;

        p = Skein512Block(p, h, h8, t);

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        if (p.s3 <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);
}


#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)


#define JHXOR \
    h0h ^= input[0]; \
    h0l ^= input[1]; \
    h1h ^= input[2]; \
    h1l ^= input[3]; \
    h2h ^= input[4]; \
    h2l ^= input[5]; \
    h3h ^= input[6]; \
    h3l ^= input[7]; \
\
    E8; \
\
    h4h ^= input[0]; \
    h4l ^= input[1]; \
    h5h ^= input[2]; \
    h5l ^= input[3]; \
    h6h ^= input[4]; \
    h6l ^= input[5]; \
    h7h ^= input[6]; \
    h7l ^= input[7]


__kernel void JH(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if (idx < BranchBuf[Threads]) {
        states += 25 * BranchBuf[idx];

        sph_u64 h0h = 0xEBD3202C41A398EBUL, h0l = 0xC145B29C7BBECD92UL, h1h = 0xFAC7D4609151931CUL, h1l = 0x038A507ED6820026UL, h2h = 0x45B92677269E23A4UL, h2l = 0x77941AD4481AFBE0UL, h3h = 0x7A176B0226ABB5CDUL, h3l = 0xA82FFF0F4224F056UL;
        sph_u64 h4h = 0x754D2E7F8996A371UL, h4l = 0x62E27DF70849141DUL, h5h = 0x948F2476F7957627UL, h5l = 0x6C29804757B6D587UL, h6h = 0x6C0D8EAC2D275E5CUL, h6l = 0x0F7A0557C6508451UL, h7h = 0xEA12247067D3E47BUL, h7l = 0x69D71CD313ABE389UL;
        sph_u64 tmp;

        for (uint i = 0; i < 3; ++i) {
            ulong input[8];

            const int shifted = i << 3;
            for (uint x = 0; x < 8; ++x) {
                input[x] = (states[shifted + x]);
            }

            JHXOR;
        }

        {
            ulong input[8] = { (states[24]), 0x80UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL };
            JHXOR;
        }

        {
            ulong input[8] = { 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x4006000000000000UL };
            JHXOR;
        }

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        if (h7l <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }
}


#define SWAP4(x)    as_uint(as_uchar4(x).s3210)


__kernel void Blake(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if (idx < BranchBuf[Threads]) {
        states += 25 * BranchBuf[idx];

        unsigned int m[16];
        unsigned int v[16];
        uint h[8];
        uint bitlen = 0;

        ((uint8 *)h)[0] = vload8(0U, c_IV256);

        for (uint i = 0; i < 3; ++i) {
            ((uint16 *)m)[0] = vload16(i, (__global uint *)states);
            for (uint x = 0; x < 16; ++x) {
                m[x] = SWAP4(m[x]);
            }

            bitlen += 512;

            ((uint16 *)v)[0].lo = ((uint8 *)h)[0];
            ((uint16 *)v)[0].hi = vload8(0U, c_u256);

            v[12] ^= bitlen;
            v[13] ^= bitlen;

            for (uint r = 0; r < 14; r++) {
                GS(0, 4, 0x8, 0xC, 0x0);
                GS(1, 5, 0x9, 0xD, 0x2);
                GS(2, 6, 0xA, 0xE, 0x4);
                GS(3, 7, 0xB, 0xF, 0x6);
                GS(0, 5, 0xA, 0xF, 0x8);
                GS(1, 6, 0xB, 0xC, 0xA);
                GS(2, 7, 0x8, 0xD, 0xC);
                GS(3, 4, 0x9, 0xE, 0xE);
            }

            ((uint8 *)h)[0] ^= ((uint8 *)v)[0] ^ ((uint8 *)v)[1];
        }

        m[0]  = SWAP4(((__global uint *)states)[48]);
        m[1]  = SWAP4(((__global uint *)states)[49]);
        m[2]  = 0x80000000U;
        m[3]  = 0x00U;
        m[4]  = 0x00U;
        m[5]  = 0x00U;
        m[6]  = 0x00U;
        m[7]  = 0x00U;
        m[8]  = 0x00U;
        m[9]  = 0x00U;
        m[10] = 0x00U;
        m[11] = 0x00U;
        m[12] = 0x00U;
        m[13] = 1U;
        m[14] = 0U;
        m[15] = 0x640;

        bitlen += 64;

        ((uint16 *)v)[0].lo = ((uint8 *)h)[0];
        ((uint16 *)v)[0].hi = vload8(0U, c_u256);

        v[12] ^= bitlen;
        v[13] ^= bitlen;

        for (uint r = 0; r < 14; r++) {
            GS(0, 4, 0x8, 0xC, 0x0);
            GS(1, 5, 0x9, 0xD, 0x2);
            GS(2, 6, 0xA, 0xE, 0x4);
            GS(3, 7, 0xB, 0xF, 0x6);
            GS(0, 5, 0xA, 0xF, 0x8);
            GS(1, 6, 0xB, 0xC, 0xA);
            GS(2, 7, 0x8, 0xD, 0xC);
            GS(3, 4, 0x9, 0xE, 0xE);
        }

        ((uint8 *)h)[0] ^= ((uint8 *)v)[0] ^ ((uint8 *)v)[1];

        for (uint i = 0; i < 8; ++i) {
            h[i] = SWAP4(h[i]);
        }

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        uint2 t = (uint2)(h[6],h[7]);
        if (as_ulong(t) <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }
}


#undef SWAP4


__kernel void Groestl(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if (idx < BranchBuf[Threads]) {
        states += 25 * BranchBuf[idx];

        ulong State[8] = { 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0x0001000000000000UL };
        ulong H[8], M[8];

        // BUG: AMD driver 19.7.X crashs if this is written as loop
        // Thx AMD for so bad software
        {
            ((ulong8 *)M)[0] = vload8(0, states);

            for (uint x = 0; x < 8; ++x) {
                H[x] = M[x] ^ State[x];
            }

            PERM_SMALL_P(H);
            PERM_SMALL_Q(M);

            for (uint x = 0; x < 8; ++x) {
                State[x] ^= H[x] ^ M[x];
            }
        }

        {
            ((ulong8 *)M)[0] = vload8(1, states);

            for (uint x = 0; x < 8; ++x) {
                H[x] = M[x] ^ State[x];
            }

            PERM_SMALL_P(H);
            PERM_SMALL_Q(M);

            for (uint x = 0; x < 8; ++x) {
                State[x] ^= H[x] ^ M[x];
            }
        }

        {
            ((ulong8 *)M)[0] = vload8(2, states);

            for (uint x = 0; x < 8; ++x) {
                H[x] = M[x] ^ State[x];
            }

            PERM_SMALL_P(H);
            PERM_SMALL_Q(M);

            for (uint x = 0; x < 8; ++x) {
                State[x] ^= H[x] ^ M[x];
            }
        }

        M[0] = states[24];
        M[1] = 0x80UL;
        M[2] = 0UL;
        M[3] = 0UL;
        M[4] = 0UL;
        M[5] = 0UL;
        M[6] = 0UL;
        M[7] = 0x0400000000000000UL;

        for (uint x = 0; x < 8; ++x) {
            H[x] = M[x] ^ State[x];
        }

        PERM_SMALL_P(H);
        PERM_SMALL_Q(M);

        ulong tmp[8];
        for (uint i = 0; i < 8; ++i) {
            tmp[i] = State[i] ^= H[i] ^ M[i];
        }

        PERM_SMALL_P(State);

        for (uint i = 0; i < 8; ++i) {
            State[i] ^= tmp[i];
        }

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        if (State[7] <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }
}
