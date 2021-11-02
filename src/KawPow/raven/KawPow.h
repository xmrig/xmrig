R"===(
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

#if __CUDA_ARCH__ < 350
    #define ROTL32(x,n) (((x) << (n % 32)) | ((x) >> (32 - (n % 32))))
    #define ROTR32(x,n) (((x) >> (n % 32)) | ((x) << (32 - (n % 32))))
#else
    #define ROTL32(x,n) __funnelshift_l((x), (x), (n))
    #define ROTR32(x,n) __funnelshift_r((x), (x), (n))
#endif

#define min(a,b)     ((a<b) ? a : b)
#define mul_hi(a, b) __umulhi(a, b)
#define clz(a)       __clz(a)
#define popcount(a)  __popc(a)

#define DEV_INLINE __device__ __forceinline__

#if (__CUDACC_VER_MAJOR__ > 8)
    #define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))
#else
    #define SHFL(x, y, z) __shfl((x), (y), (z))
#endif

#define PROGPOW_LANES           16
#define PROGPOW_REGS            32
#define PROGPOW_DAG_LOADS       4
#define PROGPOW_CACHE_WORDS     4096
#define PROGPOW_CNT_DAG         64
#define PROGPOW_CNT_MATH        18

typedef struct __align__(16) {uint32_t s[PROGPOW_DAG_LOADS];} dag_t;

DEV_INLINE void progPowLoop(const uint32_t loop, uint32_t mix[PROGPOW_REGS], const dag_t *g_dag, const uint32_t c_dag[PROGPOW_CACHE_WORDS], const bool hack_false)
{
    dag_t data_dag;
    uint32_t offset, data;
    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES-1);

    // global load
    offset = SHFL(mix[0], loop % PROGPOW_LANES, PROGPOW_LANES);
    XMRIG_INCLUDE_OFFSET_MOD_DAG_ELEMENTS
    offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;
    data_dag = g_dag[offset];

    // hack to prevent compiler from reordering LD and usage
    if (hack_false) __threadfence_block();

    XMRIG_INCLUDE_PROGPOW_RANDOM_MATH

    // consume global load data
    // hack to prevent compiler from reordering LD and usage
    if (hack_false) __threadfence_block();

    XMRIG_INCLUDE_PROGPOW_DATA_LOADS
}

#define FNV_PRIME 0x1000193
#define FNV_OFFSET_BASIS 0x811c9dc5

typedef struct
{
    uint32_t uint32s[32 / sizeof(uint32_t)];
} hash32_t;

// Implementation based on:
// https://github.com/mjosaarinen/tiny_sha3/blob/master/sha3.c


__device__ __constant__ const uint32_t keccakf_rndc[24] = {
    0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
    0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800a, 0x8000000a, 0x80008081, 0x00008080, 0x80000001, 0x80008008
};

__device__ __constant__ const uint32_t ravencoin_rndc[15] = {
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

// Implementation of the permutation Keccakf with width 800.
__device__ __forceinline__ void keccak_f800_round(uint32_t st[25], const int r)
{

    const uint32_t keccakf_rotc[24] = {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
        27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
    };
    const uint32_t keccakf_piln[24] = {
        10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
    };

    uint32_t t, bc[5];
    // Theta
    for (int i = 0; i < 5; i++)
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for (int i = 0; i < 5; i++) {
        t = bc[(i + 4) % 5] ^ ROTL32(bc[(i + 1) % 5], 1);
        for (uint32_t j = 0; j < 25; j += 5)
            st[j + i] ^= t;
    }

    // Rho Pi
    t = st[1];
    for (int i = 0; i < 24; i++) {
        uint32_t j = keccakf_piln[i];
        bc[0] = st[j];
        st[j] = ROTL32(t, keccakf_rotc[i]);
        t = bc[0];
    }

    //  Chi
    for (uint32_t j = 0; j < 25; j += 5) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[j + i];
        for (int i = 0; i < 5; i++)
            st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    //  Iota
    st[0] ^= keccakf_rndc[r];
}

__device__ __forceinline__ uint32_t cuda_swab32(const uint32_t x)
{
    return __byte_perm(x, x, 0x0123);
}

// Keccak - implemented as a variant of SHAKE
// The width is 800, with a bitrate of 576, a capacity of 224, and no padding
// Only need 64 bits of output for mining
__device__ __forceinline__ void keccak_f800(uint32_t* st)
{
    // Assumes input state has already been filled
    // at higher level

    // Complete all 22 rounds as a separate impl to
    // evaluate only first 8 words is wasteful of regsters
    for (int r = 0; r < 22; r++) {
        keccak_f800_round(st, r);
    }
}

#define fnv1a(h, d) (h = (uint32_t(h) ^ uint32_t(d)) * uint32_t(FNV_PRIME))

typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

// KISS99 is simple, fast, and passes the TestU01 suite
// https://en.wikipedia.org/wiki/KISS_(algorithm)
// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
__device__ __forceinline__ uint32_t kiss99(kiss99_t &st)
{
    st.z = 36969 * (st.z & 65535) + (st.z >> 16);
    st.w = 18000 * (st.w & 65535) + (st.w >> 16);
    uint32_t MWC = ((st.z << 16) + st.w);
    st.jsr ^= (st.jsr << 17);
    st.jsr ^= (st.jsr >> 13);
    st.jsr ^= (st.jsr << 5);
    st.jcong = 69069 * st.jcong + 1234567;
    return ((MWC^st.jcong) + st.jsr);
}

__device__ __forceinline__ void fill_mix(uint32_t* hash_seed, uint32_t lane_id, uint32_t* mix)
{
    // Use FNV to expand the per-warp seed to per-lane
    // Use KISS to expand the per-lane seed to fill mix
    uint32_t fnv_hash = FNV_OFFSET_BASIS;
    kiss99_t st;
    st.z = fnv1a(fnv_hash, hash_seed[0]);
    st.w = fnv1a(fnv_hash, hash_seed[1]);
    st.jsr = fnv1a(fnv_hash, lane_id);
    st.jcong = fnv1a(fnv_hash, lane_id);
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(st);
}

__global__ void XMRIG_INCLUDE_LAUNCH_BOUNDS progpow_search(const dag_t *g_dag, const uint32_t* job_blob, const uint64_t target, bool hack_false, uint32_t* results, uint32_t* stop)
{
    if (*stop) {
        if ((threadIdx.x == 0) && ((blockIdx.x & 15) == 0)) {
            // Count groups of skipped hashes (if we don't count them we'll break hashrate display)
            atomicAdd(stop + 1, blockDim.x * 16);
        }
        return;
    }

    __shared__ uint32_t c_dag[PROGPOW_CACHE_WORDS];

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);

    // Load the first portion of the DAG into the shared cache
    for (uint32_t word = threadIdx.x*PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS; word += blockDim.x*PROGPOW_DAG_LOADS)
    {
        dag_t load = g_dag[word/PROGPOW_DAG_LOADS];
        for (int i = 0; i < PROGPOW_DAG_LOADS; ++i)
            c_dag[word + i] =  load.s[i];
    }

    uint32_t hash_seed[2];  // KISS99 initiator
    hash32_t digest;        // Carry-over from mix output

    uint32_t state2[8];

    {
        // Absorb phase for initial round of keccak

        uint32_t state[25] = {0x0};     // Keccak's state

        // 1st fill with job data
        for (int i = 0; i < 10; i++)
            state[i] = job_blob[i];

        // Apply nonce
        gid += state[8];
        state[8] = gid;

        // 3rd apply ravencoin input constraints
        for (int i = 10; i < 25; i++)
            state[i] = ravencoin_rndc[i-10];

        // Run intial keccak round
        keccak_f800(state);

        for (int i = 0; i < 8; i++)
            state2[i] = state[i];
    }

    // Force threads to sync and ensure shared mem is in sync
    __syncthreads();

    #pragma unroll 1
    for (uint32_t h = 0; h < PROGPOW_LANES; h++)
    {
        uint32_t mix[PROGPOW_REGS];

        hash_seed[0] = __shfl_sync(0xFFFFFFFF, state2[0], h, PROGPOW_LANES);
        hash_seed[1] = __shfl_sync(0xFFFFFFFF, state2[1], h, PROGPOW_LANES);

        // initialize mix for all lanes
        fill_mix(hash_seed, lane_id, mix);

        #pragma unroll 1
        for (uint32_t l = 0; l < PROGPOW_CNT_DAG; l++)
            progPowLoop(l, mix, g_dag, c_dag, hack_false);

        // Reduce mix data to a per-lane 32-bit digest
        uint32_t digest_lane = FNV_OFFSET_BASIS;
        #pragma unroll
        for (int i = 0; i < PROGPOW_REGS; i++)
            fnv1a(digest_lane, mix[i]);

        // Reduce all lanes to a single 256-bit digest
        hash32_t digest_temp;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            digest_temp.uint32s[i] = FNV_OFFSET_BASIS;

        for (int i = 0; i < PROGPOW_LANES; i += 8)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                fnv1a(digest_temp.uint32s[j], SHFL(digest_lane, i + j, PROGPOW_LANES));

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

        // Extract result, swap endianness, and compare with target
        result = (uint64_t) cuda_swab32(state[0]) << 32 | cuda_swab32(state[1]);
    }

    // Check result vs target
    if (result < target) {
        *stop = 1;

        const uint32_t index = atomicAdd(results, 1U) + 1U;
        if (index <= 15) {
            results[index] = gid;
        }
    }
}
)==="
