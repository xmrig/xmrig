#include "defs.h"

typedef struct __attribute__ ((aligned(16))) {uint32_t s[PROGPOW_DAG_LOADS];} dag_t;

// Implementation based on:
// https://github.com/mjosaarinen/tiny_sha3/blob/master/sha3.c

__constant const uint32_t keccakf_rndc[24] = {0x00000001, 0x00008082, 0x0000808a, 0x80008000,
    0x0000808b, 0x80000001, 0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080, 0x0000800a, 0x8000000a,
    0x80008081, 0x00008080, 0x80000001, 0x80008008};

__constant const uint32_t ravencoin_rndc[15] = {
        0x00000072, //R
        0x00000041, //A
        0x00000056, //V
        0x00000045, //E
        0x0000004E, //N
        0x00000043, //C
        0x0000004F, //O
        0x00000049, //I
        0x0000004E, //N
        0x0000004B, //K
        0x00000041, //A
        0x00000057, //W
        0x00000050, //P
        0x0000004F, //O
        0x00000057, //W
};

// Implementation of the Keccakf transformation with a width of 800
void keccak_f800_round(uint32_t st[25], const int r)
{
    const uint32_t keccakf_rotc[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    const uint32_t keccakf_piln[24] = {
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

    uint32_t t, bc[5];
    // Theta
    for (int i = 0; i < 5; i++)
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for (int i = 0; i < 5; i++)
    {
        t = bc[(i + 4) % 5] ^ ROTL32(bc[(i + 1) % 5], 1u);
        for (uint32_t j = 0; j < 25; j += 5)
            st[j + i] ^= t;
    }

    // Rho Pi
    t = st[1];
    for (int i = 0; i < 24; i++)
    {
        uint32_t j = keccakf_piln[i];
        bc[0] = st[j];
        st[j] = ROTL32(t, keccakf_rotc[i]);
        t = bc[0];
    }

    //  Chi
    for (uint32_t j = 0; j < 25; j += 5)
    {
        for (int i = 0; i < 5; i++)
            bc[i] = st[j + i];
        for (int i = 0; i < 5; i++)
            st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    //  Iota
    st[0] ^= keccakf_rndc[r];
}

// Keccak - implemented as a variant of SHAKE
// The width is 800, with a bitrate of 576, a capacity of 224, and no padding
// Only need 64 bits of output for mining
uint64_t keccak_f800(uint32_t* st)
{
    // Complete all 22 rounds as a separate impl to
    // evaluate only first 8 words is wasteful of regsters
    for (int r = 0; r < 22; r++) {
        keccak_f800_round(st, r);
    }
}

#define fnv1a(h, d) (h = (h ^ d) * FNV_PRIME)

typedef struct
{
    uint32_t z, w, jsr, jcong;
} kiss99_t;

// KISS99 is simple, fast, and passes the TestU01 suite
// https://en.wikipedia.org/wiki/KISS_(algorithm)
// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
uint32_t kiss99(kiss99_t* st)
{
    st->z = 36969 * (st->z & 65535) + (st->z >> 16);
    st->w = 18000 * (st->w & 65535) + (st->w >> 16);
    uint32_t MWC = ((st->z << 16) + st->w);
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    st->jcong = 69069 * st->jcong + 1234567;
    return ((MWC ^ st->jcong) + st->jsr);
}

void fill_mix(local uint32_t* seed, uint32_t lane_id, uint32_t* mix)
{
    // Use FNV to expand the per-warp seed to per-lane
    // Use KISS to expand the per-lane seed to fill mix
    uint32_t fnv_hash = FNV_OFFSET_BASIS;
    kiss99_t st;
    st.z = fnv1a(fnv_hash, seed[0]);
    st.w = fnv1a(fnv_hash, seed[1]);
    st.jsr = fnv1a(fnv_hash, lane_id);
    st.jcong = fnv1a(fnv_hash, lane_id);
#pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(&st);
}

typedef struct
{
    uint32_t uint32s[PROGPOW_LANES];
} shuffle_t;

typedef struct
{
    uint32_t uint32s[32 / sizeof(uint32_t)];
} hash32_t;

#if PLATFORM != OPENCL_PLATFORM_NVIDIA  // use maxrregs on nv
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
#endif
__kernel void progpow_search(__global dag_t const* g_dag, __global uint* job_blob, ulong target, uint hack_false, volatile __global uint* results, volatile __global uint* stop)
{
    const uint32_t lid = get_local_id(0);
    const uint32_t gid = get_global_id(0);

    if (stop[0]) {
        if (lid == 0) {
            // Count groups of skipped hashes (if we don't count them we'll break hashrate display)
            atomic_inc(stop + 1);
        }
        return;
    }

    __local shuffle_t share[HASHES_PER_GROUP];
    __local uint32_t c_dag[PROGPOW_CACHE_WORDS];

    const uint32_t lane_id = lid & (PROGPOW_LANES - 1);
    const uint32_t group_id = lid / PROGPOW_LANES;

    // Load the first portion of the DAG into the cache
    for (uint32_t word = lid * PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS; word += GROUP_SIZE * PROGPOW_DAG_LOADS)
    {
        dag_t load = g_dag[word / PROGPOW_DAG_LOADS];
        for (int i = 0; i < PROGPOW_DAG_LOADS; i++)
            c_dag[word + i] = load.s[i];
    }

    uint32_t hash_seed[2];  // KISS99 initiator
    hash32_t digest;        // Carry-over from mix output

    uint32_t state2[8];

    {
        // Absorb phase for initial round of keccak

        uint32_t state[25];     // Keccak's state

        // 1st fill with job data
        for (int i = 0; i < 10; i++)
            state[i] = job_blob[i];

        // Apply nonce
        state[8] = gid;

        // 3rd apply ravencoin input constraints
        for (int i = 10; i < 25; i++)
            state[i] = ravencoin_rndc[i-10];

        // Run intial keccak round
        keccak_f800(state);

        for (int i = 0; i < 8; i++)
            state2[i] = state[i];

    }

#pragma unroll 1
    for (uint32_t h = 0; h < PROGPOW_LANES; h++)
    {
        uint32_t mix[PROGPOW_REGS];

        // share the hash's seed across all lanes
        if (lane_id == h) {
            share[group_id].uint32s[0] = state2[0];
            share[group_id].uint32s[1] = state2[1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // initialize mix for all lanes
        fill_mix(share[group_id].uint32s, lane_id, mix);

	#pragma unroll 2
	for (uint32_t loop = 0; loop < PROGPOW_CNT_DAG; ++loop)
	{
		// global load
		if(lane_id == (loop % PROGPOW_LANES))
			share[0].uint32s[group_id] = mix[0];

		barrier(CLK_LOCAL_MEM_FENCE);

		uint32_t offset = share[0].uint32s[group_id];
		offset %= PROGPOW_DAG_ELEMENTS;
		offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;
		dag_t data_dag = g_dag[offset];

		// hack to prevent compiler from reordering LD and usage
		if (hack_false) barrier(CLK_LOCAL_MEM_FENCE);

		uint32_t data;
		XMRIG_INCLUDE_PROGPOW_RANDOM_MATH

		// consume global load data
		// hack to prevent compiler from reordering LD and usage
		if (hack_false) barrier(CLK_LOCAL_MEM_FENCE);

		XMRIG_INCLUDE_PROGPOW_DATA_LOADS
	}

        // Reduce mix data to a per-lane 32-bit digest
        uint32_t mix_hash = FNV_OFFSET_BASIS;
#pragma unroll
        for (int i = 0; i < PROGPOW_REGS; i++)
            fnv1a(mix_hash, mix[i]);

        // Reduce all lanes to a single 256-bit digest
        hash32_t digest_temp;
        for (int i = 0; i < 8; i++)
            digest_temp.uint32s[i] = FNV_OFFSET_BASIS;
        share[group_id].uint32s[lane_id] = mix_hash;
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int i = 0; i < PROGPOW_LANES; i++)
            fnv1a(digest_temp.uint32s[i % 8], share[group_id].uint32s[i]);
        if (h == lane_id)
            digest = digest_temp;
    }


    // Absorb phase for last round of keccak (256 bits)
    uint64_t result;

    {
        uint32_t state[25] = {0x0};     // Keccak's state

        // 1st initial 8 words of state are kept as carry-over from initial keccak
        for (int i = 0; i < 8; i++)
            state[i] = state2[i];

        // 2nd subsequent 8 words are carried from digest/mix
        for (int i = 8; i < 16; i++)
            state[i] = digest.uint32s[i - 8];

        // 3rd apply ravencoin input constraints
        for (int i = 16; i < 25; i++)
            state[i] = ravencoin_rndc[i - 16];

        // Run keccak loop
        keccak_f800(state);

        uint64_t res = (uint64_t)state[1] << 32 | state[0];
        result = as_ulong(as_uchar8(res).s76543210);
    }

    if (result <= target)
    {
        *stop = 1;

        const uint k = atomic_inc(results) + 1;
        if (k <= 15)
            results[k] = gid;
    }
}
