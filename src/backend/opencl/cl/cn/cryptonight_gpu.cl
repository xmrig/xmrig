#include "wolf-aes.cl"
#include "keccak.cl"


inline uint getIdx()
{
    return get_global_id(0) - get_global_offset(0);
}


#define IDX(x) (x)


inline float4 _mm_add_ps(float4 a, float4 b)
{
    return a + b;
}


inline float4 _mm_sub_ps(float4 a, float4 b)
{
    return a - b;
}


inline float4 _mm_mul_ps(float4 a, float4 b)
{
    return a * b;
}


inline float4 _mm_div_ps(float4 a, float4 b)
{
    return a / b;
}


inline float4 _mm_and_ps(float4 a, int b)
{
    return as_float4(as_int4(a) & (int4)(b));
}


inline float4 _mm_or_ps(float4 a, int b)
{
    return as_float4(as_int4(a) | (int4)(b));
}


inline float4 _mm_fmod_ps(float4 v, float dc)
{
    float4 d = (float4)(dc);
    float4 c = _mm_div_ps(v, d);
    c = trunc(c);
    c = _mm_mul_ps(c, d);
    return _mm_sub_ps(v, c);
}


inline int4 _mm_xor_si128(int4 a, int4 b)
{
    return a ^ b;
}


inline float4 _mm_xor_ps(float4 a, int b)
{
    return as_float4(as_int4(a) ^ (int4)(b));
}


inline int4 _mm_alignr_epi8(int4 a, const uint rot)
{
    const uint right = 8 * rot;
    const uint left = (32 - 8 * rot);
    return (int4)(
        ((uint)a.x >> right) | ( a.y << left ),
        ((uint)a.y >> right) | ( a.z << left ),
        ((uint)a.z >> right) | ( a.w << left ),
        ((uint)a.w >> right) | ( a.x << left )
    );
}


inline global int4* scratchpad_ptr(uint idx, uint n, __global int *lpad) { return (__global int4*)((__global char*)lpad + (idx & MASK) + n * 16); }


inline float4 fma_break(float4 x)
{
    // Break the dependency chain by setitng the exp to ?????01
    x = _mm_and_ps(x, 0xFEFFFFFF);
    return _mm_or_ps(x, 0x00800000);
}


inline void sub_round(float4 n0, float4 n1, float4 n2, float4 n3, float4 rnd_c, float4* n, float4* d, float4* c)
{
    n1 = _mm_add_ps(n1, *c);
    float4 nn = _mm_mul_ps(n0, *c);
    nn = _mm_mul_ps(n1, _mm_mul_ps(nn,nn));
    nn = fma_break(nn);
    *n = _mm_add_ps(*n, nn);

    n3 = _mm_sub_ps(n3, *c);
    float4 dd = _mm_mul_ps(n2, *c);
    dd = _mm_mul_ps(n3, _mm_mul_ps(dd,dd));
    dd = fma_break(dd);
    *d = _mm_add_ps(*d, dd);

    //Constant feedback
    *c = _mm_add_ps(*c, rnd_c);
    *c = _mm_add_ps(*c, (float4)(0.734375f));
    float4 r = _mm_add_ps(nn, dd);
    r = _mm_and_ps(r, 0x807FFFFF);
    r = _mm_or_ps(r, 0x40000000);
    *c = _mm_add_ps(*c, r);

}


// 9*8 + 2 = 74
inline void round_compute(float4 n0, float4 n1, float4 n2, float4 n3, float4 rnd_c, float4* c, float4* r)
{
    float4 n = (float4)(0.0f);
    float4 d = (float4)(0.0f);

    sub_round(n0, n1, n2, n3, rnd_c, &n, &d, c);
    sub_round(n1, n2, n3, n0, rnd_c, &n, &d, c);
    sub_round(n2, n3, n0, n1, rnd_c, &n, &d, c);
    sub_round(n3, n0, n1, n2, rnd_c, &n, &d, c);
    sub_round(n3, n2, n1, n0, rnd_c, &n, &d, c);
    sub_round(n2, n1, n0, n3, rnd_c, &n, &d, c);
    sub_round(n1, n0, n3, n2, rnd_c, &n, &d, c);
    sub_round(n0, n3, n2, n1, rnd_c, &n, &d, c);

    // Make sure abs(d) > 2.0 - this prevents division by zero and accidental overflows by division by < 1.0
    d = _mm_and_ps(d, 0xFF7FFFFF);
    d = _mm_or_ps(d, 0x40000000);
    *r =_mm_add_ps(*r, _mm_div_ps(n,d));
}


inline int4 single_comupte(float4 n0, float4 n1, float4 n2, float4 n3, float cnt, float4 rnd_c, __local float4* sum)
{
    float4 c= (float4)(cnt);
    // 35 maths calls follow (140 FLOPS)
    float4 r = (float4)(0.0f);

    for (int i = 0; i < 4; ++i) {
        round_compute(n0, n1, n2, n3, rnd_c, &c, &r);
    }

    // do a quick fmod by setting exp to 2
    r = _mm_and_ps(r, 0x807FFFFF);
    r = _mm_or_ps(r, 0x40000000);
    *sum = r; // 34
    float4 x = (float4)(536870880.0f);
    r = _mm_mul_ps(r, x); // 35
    return convert_int4_rte(r);
}


inline void single_comupte_wrap(const uint rot, int4 v0, int4 v1, int4 v2, int4 v3, float cnt, float4 rnd_c, __local float4* sum, __local int4* out)
{
    float4 n0 = convert_float4_rte(v0);
    float4 n1 = convert_float4_rte(v1);
    float4 n2 = convert_float4_rte(v2);
    float4 n3 = convert_float4_rte(v3);

    int4 r = single_comupte(n0, n1, n2, n3, cnt, rnd_c, sum);
    *out = rot == 0 ? r : _mm_alignr_epi8(r, rot);
}


static const __constant uint look[16][4] = {
    {0, 1, 2, 3},
    {0, 2, 3, 1},
    {0, 3, 1, 2},
    {0, 3, 2, 1},

    {1, 0, 2, 3},
    {1, 2, 3, 0},
    {1, 3, 0, 2},
    {1, 3, 2, 0},

    {2, 1, 0, 3},
    {2, 0, 3, 1},
    {2, 3, 1, 0},
    {2, 3, 0, 1},

    {3, 1, 2, 0},
    {3, 2, 0, 1},
    {3, 0, 1, 2},
    {3, 0, 2, 1}
};


static const __constant float ccnt[16] = {
    1.34375f,
    1.28125f,
    1.359375f,
    1.3671875f,

    1.4296875f,
    1.3984375f,
    1.3828125f,
    1.3046875f,

    1.4140625f,
    1.2734375f,
    1.2578125f,
    1.2890625f,

    1.3203125f,
    1.3515625f,
    1.3359375f,
    1.4609375f
};


struct SharedMemChunk
{
    int4 out[16];
    float4 va[16];
};


__attribute__((reqd_work_group_size(WORKSIZE * 16, 1, 1)))
__kernel void cn1(__global int *lpad_in, __global int *spad, uint numThreads)
{
    const uint gIdx = getIdx();
    uint chunk = get_local_id(0) / 16;

    __global int* lpad = (__global int*)((__global char*)lpad_in + MEMORY * (gIdx/16));

    __local struct SharedMemChunk smem_in[WORKSIZE];
    __local struct SharedMemChunk* smem = smem_in + chunk;

    uint tid = get_local_id(0) % 16;

    uint idxHash = gIdx/16;
    uint s = ((__global uint*)spad)[idxHash * 50] >> 8;
    float4 vs = (float4)(0);

    // tid divided
    const uint tidd = tid / 4;
    // tid modulo
    const uint tidm = tid % 4;
    const uint block = tidd * 16 + tidm;

    #pragma unroll CN_UNROLL
    for (uint i = 0; i < ITERATIONS; i++)   {
        mem_fence(CLK_LOCAL_MEM_FENCE);
        int tmp = ((__global int*)scratchpad_ptr(s, tidd, lpad))[tidm];
        ((__local int*)(smem->out))[tid] = tmp;
        mem_fence(CLK_LOCAL_MEM_FENCE);

        {
            single_comupte_wrap(
                tidm,
                *(smem->out + look[tid][0]),
                *(smem->out + look[tid][1]),
                *(smem->out + look[tid][2]),
                *(smem->out + look[tid][3]),
                ccnt[tid], vs, smem->va + tid,
                smem->out + tid
            );
        }
        mem_fence(CLK_LOCAL_MEM_FENCE);

        int outXor = ((__local int*)smem->out)[block];
        for (uint dd = block + 4; dd < (tidd + 1) * 16; dd += 4) {
            outXor ^= ((__local int*)smem->out)[dd];
        }

        ((__global int*)scratchpad_ptr(s, tidd, lpad))[tidm] = outXor ^ tmp;
        ((__local int*)smem->out)[tid] = outXor;

        float va_tmp1 = ((__local float*)smem->va)[block] + ((__local float*)smem->va)[block + 4];
        float va_tmp2 = ((__local float*)smem->va)[block+ 8] + ((__local float*)smem->va)[block + 12];
        ((__local float*)smem->va)[tid] = va_tmp1 + va_tmp2;

        mem_fence(CLK_LOCAL_MEM_FENCE);

        int out2 = ((__local int*)smem->out)[tid] ^ ((__local int*)smem->out)[tid + 4 ] ^ ((__local int*)smem->out)[tid + 8] ^ ((__local int*)smem->out)[tid + 12];
        va_tmp1 = ((__local float*)smem->va)[block] + ((__local float*)smem->va)[block + 4];
        va_tmp2 = ((__local float*)smem->va)[block + 8] + ((__local float*)smem->va)[block + 12];
        va_tmp1 = va_tmp1 + va_tmp2;
        va_tmp1 = fabs(va_tmp1);

        float xx = va_tmp1 * 16777216.0f;
        int xx_int = (int)xx;
        ((__local int*)smem->out)[tid] = out2 ^ xx_int;
        ((__local float*)smem->va)[tid] = va_tmp1 / 64.0f;

        mem_fence(CLK_LOCAL_MEM_FENCE);

        vs = smem->va[0];
        s = smem->out[0].x ^ smem->out[0].y ^ smem->out[0].z ^ smem->out[0].w;
    }
}


static const __constant uint skip[3] = {
    20,22,22
};


inline void generate_512(uint idx, __local ulong* in, __global ulong* out)
{
    ulong hash[25];

    hash[0] = in[0] ^ idx;
    for (int i = 1; i < 25; ++i) {
        hash[i] = in[i];
    }

    for (int a = 0; a < 3; ++a) {
        keccakf1600_1(hash);
        for (int i = 0; i < skip[a]; ++i) {
            out[i] = hash[i];
        }

        out += skip[a];
    }
}


__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void cn0(__global ulong *input, __global int *Scratchpad, __global ulong *states, uint Threads)
{
    const uint gIdx = getIdx();
    __local ulong State_buf[8 * 25];
    __local ulong* State = State_buf + get_local_id(0) * 25;

    {
        states += 25 * gIdx;

        Scratchpad = (__global int*)((__global char*)Scratchpad + MEMORY * gIdx);

        if (get_local_id(1) == 0) {

#           ifdef __NV_CL_C_VERSION
            for(uint i = 0; i < 8; ++i)
                State[i] = input[i];
#           else
            ((__local ulong8 *)State)[0] = vload8(0, input);
#           endif

            State[8]  = input[8];
            State[9]  = input[9];
            State[10] = input[10];

            ((__local uint *)State)[9]  &= 0x00FFFFFFU;
            ((__local uint *)State)[9]  |= (((uint)get_global_id(0)) & 0xFF) << 24;
            ((__local uint *)State)[10] &= 0xFF000000U;
            /* explicit cast to `uint` is required because some OpenCL implementations (e.g. NVIDIA)
             * handle get_global_id and get_global_offset as signed long long int and add
             * 0xFFFFFFFF... to `get_global_id` if we set on host side a 32bit offset where the first bit is `1`
             * (even if it is correct casted to unsigned on the host)
             */
            ((__local uint *)State)[10] |= (((uint)get_global_id(0) >> 8));

            for (int i = 11; i < 25; ++i) {
                State[i] = 0x00UL;
            }

            // Last bit of padding
            State[16] = 0x8000000000000000UL;

            keccakf1600_2(State);

            #pragma unroll
            for (int i = 0; i < 25; ++i) {
                states[i] = State[i];
            }
        }
    }
}


__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void cn00(__global int *Scratchpad, __global ulong *states)
{
    const uint gIdx = getIdx() / 64;
    __local ulong State[25];

    states += 25 * gIdx;

    Scratchpad = (__global int*)((__global char*)Scratchpad + MEMORY * gIdx);

    for (int i = get_local_id(0); i < 25; i += get_local_size(0)) {
        State[i] = states[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = get_local_id(0); i < MEMORY / 512; i += get_local_size(0)) {
        generate_512(i, State, (__global ulong*)((__global uchar*)Scratchpad + i * 512));
    }
}


__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void cn2(__global uint4 *Scratchpad, __global ulong *states, __global uint *output, ulong Target, uint Threads)
{
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];
    uint ExpandedKey2[40];
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

    __local uint4 xin1[8][8];
    __local uint4 xin2[8][8];

    {
        states += 25 * gIdx;
        Scratchpad += gIdx * (MEMORY >> 4);

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

    __local uint4* xin1_store = &xin1[get_local_id(1)][get_local_id(0)];
    __local uint4* xin1_load = &xin1[(get_local_id(1) + 1) % 8][get_local_id(0)];
    __local uint4* xin2_store = &xin2[get_local_id(1)][get_local_id(0)];
    __local uint4* xin2_load = &xin2[(get_local_id(1) + 1) % 8][get_local_id(0)];
    *xin2_store = (uint4)(0, 0, 0, 0);

    {

        #pragma unroll 2
        for (int i = 0, i1 = get_local_id(1); i < (MEMORY >> 7); ++i, i1 = (i1 + 16) % (MEMORY >> 4)) {
            text ^= Scratchpad[(uint)i1];
            barrier(CLK_LOCAL_MEM_FENCE);
            text ^= *xin2_load;

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

            *xin1_store = text;

            text ^= Scratchpad[(uint)i1 + 8u];
            barrier(CLK_LOCAL_MEM_FENCE);
            text ^= *xin1_load;

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

            *xin2_store = text;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        text ^= *xin2_load;
    }

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

    __local ulong State_buf[8 * 25];
    {
        vstore2(as_ulong2(text), get_local_id(1) + 4, states);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    {
        if(!get_local_id(1))
        {
            __local ulong* State = State_buf + get_local_id(0) * 25;

            for(int i = 0; i < 25; ++i) State[i] = states[i];

            keccakf1600_2(State);

            if(State[3] <= Target)
            {
                ulong outIdx = atomic_inc(output + 0xFF);
                if(outIdx < 0xFF)
                    output[outIdx] = get_global_id(0);
            }
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}
