R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_v2_rwz(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT)
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

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;

#       if defined(__NV_CL_C_VERSION)
            Scratchpad += gIdx * (0x40000 >> 2);
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

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
    uint2 division_result = as_uint2(states[12]);
    uint sqrt_result = as_uint2(states[13]).s0;

    #pragma unroll UNROLL_FACTOR
    for(int i = 0; i < 0x60000; ++i)
    {
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
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(3));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(1));

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

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk1 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk3 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
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
#   endif
}

)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_v2_zls(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT)
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

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;

#       if defined(__NV_CL_C_VERSION)
            Scratchpad += gIdx * (0x60000 >> 2);
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

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
    uint2 division_result = as_uint2(states[12]);
    uint sqrt_result = as_uint2(states[13]).s0;

    #pragma unroll UNROLL_FACTOR
    for(int i = 0; i < 0x60000; ++i)
    {
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
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

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

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
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
#   endif
}

)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_v2_double(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT)
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

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;

#       if defined(__NV_CL_C_VERSION)
            Scratchpad += gIdx * (0x100000 >> 2);
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

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
    uint2 division_result = as_uint2(states[12]);
    uint sqrt_result = as_uint2(states[13]).s0;

    #pragma unroll UNROLL_FACTOR
    for(int i = 0; i < 0x100000; ++i)
    {
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
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

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

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
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
#   endif
}

)==="
