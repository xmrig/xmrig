#include "defs.h"

//
// DAG calculation logic
//


#define ETHASH_DATASET_PARENTS 512
#define NODE_WORDS (64 / 4)

__constant uint2 const Keccak_f1600_RC[24] = {
    (uint2)(0x00000001, 0x00000000),
    (uint2)(0x00008082, 0x00000000),
    (uint2)(0x0000808a, 0x80000000),
    (uint2)(0x80008000, 0x80000000),
    (uint2)(0x0000808b, 0x00000000),
    (uint2)(0x80000001, 0x00000000),
    (uint2)(0x80008081, 0x80000000),
    (uint2)(0x00008009, 0x80000000),
    (uint2)(0x0000008a, 0x00000000),
    (uint2)(0x00000088, 0x00000000),
    (uint2)(0x80008009, 0x00000000),
    (uint2)(0x8000000a, 0x00000000),
    (uint2)(0x8000808b, 0x00000000),
    (uint2)(0x0000008b, 0x80000000),
    (uint2)(0x00008089, 0x80000000),
    (uint2)(0x00008003, 0x80000000),
    (uint2)(0x00008002, 0x80000000),
    (uint2)(0x00000080, 0x80000000),
    (uint2)(0x0000800a, 0x00000000),
    (uint2)(0x8000000a, 0x80000000),
    (uint2)(0x80008081, 0x80000000),
    (uint2)(0x00008080, 0x80000000),
    (uint2)(0x80000001, 0x00000000),
    (uint2)(0x80008008, 0x80000000),
};

#if PLATFORM == OPENCL_PLATFORM_NVIDIA && COMPUTE >= 35
static uint2 ROL2(const uint2 a, const int offset)
{
    uint2 result;
    if (offset >= 32)
    {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
    }
    else
    {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    }
    return result;
}
#elif PLATFORM == OPENCL_PLATFORM_AMD
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
static uint2 ROL2(const uint2 vv, const int r)
{
    if (r <= 32)
    {
        return amd_bitalign((vv).xy, (vv).yx, 32 - r);
    }
    else
    {
        return amd_bitalign((vv).yx, (vv).xy, 64 - r);
    }
}
#else
static uint2 ROL2(const uint2 v, const int n)
{
    uint2 result;
    if (n <= 32)
    {
        result.y = ((v.y << (n)) | (v.x >> (32 - n)));
        result.x = ((v.x << (n)) | (v.y >> (32 - n)));
    }
    else
    {
        result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
        result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));
    }
    return result;
}
#endif

static void chi(uint2* a, const uint n, const uint2* t)
{
    a[n + 0] = bitselect(t[n + 0] ^ t[n + 2], t[n + 0], t[n + 1]);
    a[n + 1] = bitselect(t[n + 1] ^ t[n + 3], t[n + 1], t[n + 2]);
    a[n + 2] = bitselect(t[n + 2] ^ t[n + 4], t[n + 2], t[n + 3]);
    a[n + 3] = bitselect(t[n + 3] ^ t[n + 0], t[n + 3], t[n + 4]);
    a[n + 4] = bitselect(t[n + 4] ^ t[n + 1], t[n + 4], t[n + 0]);
}

static void keccak_f1600_round(uint2* a, uint r)
{
    uint2 t[25];
    uint2 u;

    // Theta
    t[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    t[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    t[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    t[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    t[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    u = t[4] ^ ROL2(t[1], 1);
    a[0] ^= u;
    a[5] ^= u;
    a[10] ^= u;
    a[15] ^= u;
    a[20] ^= u;
    u = t[0] ^ ROL2(t[2], 1);
    a[1] ^= u;
    a[6] ^= u;
    a[11] ^= u;
    a[16] ^= u;
    a[21] ^= u;
    u = t[1] ^ ROL2(t[3], 1);
    a[2] ^= u;
    a[7] ^= u;
    a[12] ^= u;
    a[17] ^= u;
    a[22] ^= u;
    u = t[2] ^ ROL2(t[4], 1);
    a[3] ^= u;
    a[8] ^= u;
    a[13] ^= u;
    a[18] ^= u;
    a[23] ^= u;
    u = t[3] ^ ROL2(t[0], 1);
    a[4] ^= u;
    a[9] ^= u;
    a[14] ^= u;
    a[19] ^= u;
    a[24] ^= u;

    // Rho Pi

    t[0] = a[0];
    t[10] = ROL2(a[1], 1);
    t[20] = ROL2(a[2], 62);
    t[5] = ROL2(a[3], 28);
    t[15] = ROL2(a[4], 27);

    t[16] = ROL2(a[5], 36);
    t[1] = ROL2(a[6], 44);
    t[11] = ROL2(a[7], 6);
    t[21] = ROL2(a[8], 55);
    t[6] = ROL2(a[9], 20);

    t[7] = ROL2(a[10], 3);
    t[17] = ROL2(a[11], 10);
    t[2] = ROL2(a[12], 43);
    t[12] = ROL2(a[13], 25);
    t[22] = ROL2(a[14], 39);

    t[23] = ROL2(a[15], 41);
    t[8] = ROL2(a[16], 45);
    t[18] = ROL2(a[17], 15);
    t[3] = ROL2(a[18], 21);
    t[13] = ROL2(a[19], 8);

    t[14] = ROL2(a[20], 18);
    t[24] = ROL2(a[21], 2);
    t[9] = ROL2(a[22], 61);
    t[19] = ROL2(a[23], 56);
    t[4] = ROL2(a[24], 14);

    // Chi
    chi(a, 0, t);

    // Iota
    a[0] ^= Keccak_f1600_RC[r];

    chi(a, 5, t);
    chi(a, 10, t);
    chi(a, 15, t);
    chi(a, 20, t);
}

static void keccak_f1600_no_absorb(uint2* a, uint out_size, uint isolate)
{
    // Originally I unrolled the first and last rounds to interface
    // better with surrounding code, however I haven't done this
    // without causing the AMD compiler to blow up the VGPR usage.


    // uint o = 25;
    for (uint r = 0; r < 24;)
    {
        // This dynamic branch stops the AMD compiler unrolling the loop
        // and additionally saves about 33% of the VGPRs, enough to gain another
        // wavefront. Ideally we'd get 4 in flight, but 3 is the best I can
        // massage out of the compiler. It doesn't really seem to matter how
        // much we try and help the compiler save VGPRs because it seems to throw
        // that information away, hence the implementation of keccak here
        // doesn't bother.
        if (isolate)
        {
            keccak_f1600_round(a, r++);
            // if (r == 23) o = out_size;
        }
    }


    // final round optimised for digest size
    // keccak_f1600_round(a, 23, out_size);
}

#define copy(dst, src, count)         \
    for (uint i = 0; i != count; ++i) \
    {                                 \
        (dst)[i] = (src)[i];          \
    }

static uint fnv(uint x, uint y)
{
    return x * FNV_PRIME ^ y;
}

static uint4 fnv4(uint4 x, uint4 y)
{
    return x * FNV_PRIME ^ y;
}

typedef union
{
    uint words[64 / sizeof(uint)];
    uint2 uint2s[64 / sizeof(uint2)];
    uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union
{
    uint words[200 / sizeof(uint)];
    uint2 uint2s[200 / sizeof(uint2)];
    uint4 uint4s[200 / sizeof(uint4)];
} hash200_t;

typedef struct
{
    uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

static void SHA3_512(uint2* s, uint isolate)
{
    for (uint i = 8; i != 25; ++i)
    {
        s[i] = (uint2){0, 0};
    }
    s[8].x = 0x00000001;
    s[8].y = 0x80000000;
    keccak_f1600_no_absorb(s, 8, isolate);
}

static uint fast_mod(uint a, uint4 d)
{
    const ulong t = a;
    const uint q = ((t + d.y) * d.x) >> d.z;
    return a - q * d.w;
}

__kernel void ethash_calculate_dag_item(uint start, __global hash64_t const* g_light, __global hash64_t* g_dag, uint isolate, uint dag_words, uint4 light_words)
{
    uint const node_index = start + get_global_id(0);
    if (node_index >= dag_words)
        return;

    hash200_t dag_node;
    copy(dag_node.uint4s, g_light[fast_mod(node_index, light_words)].uint4s, 4);
    dag_node.words[0] ^= node_index;
    SHA3_512(dag_node.uint2s, isolate);

    for (uint i = 0; i != ETHASH_DATASET_PARENTS; ++i)
    {
        uint parent_index = fast_mod(fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]), light_words);

        for (uint w = 0; w != 4; ++w)
            dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], g_light[parent_index].uint4s[w]);
    }

    SHA3_512(dag_node.uint2s, isolate);
    copy(g_dag[node_index].uint4s, dag_node.uint4s, 4);
}
