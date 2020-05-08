#include "argon2-avx512f.h"

#ifdef HAVE_AVX512F
#include <stdint.h>
#include <string.h>

#ifdef __GNUC__
#   include <x86intrin.h>
#else
#   include <intrin.h>
#endif

#define ror64(x, n) _mm512_ror_epi64((x), (n))

static __m512i f(__m512i x, __m512i y)
{
    __m512i z = _mm512_mul_epu32(x, y);
    return _mm512_add_epi64(_mm512_add_epi64(x, y), _mm512_add_epi64(z, z));
}

#define G1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = f(A0, B0); \
        A1 = f(A1, B1); \
\
        D0 = _mm512_xor_si512(D0, A0); \
        D1 = _mm512_xor_si512(D1, A1); \
\
        D0 = ror64(D0, 32); \
        D1 = ror64(D1, 32); \
\
        C0 = f(C0, D0); \
        C1 = f(C1, D1); \
\
        B0 = _mm512_xor_si512(B0, C0); \
        B1 = _mm512_xor_si512(B1, C1); \
\
        B0 = ror64(B0, 24); \
        B1 = ror64(B1, 24); \
    } while ((void)0, 0)

#define G2(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = f(A0, B0); \
        A1 = f(A1, B1); \
\
        D0 = _mm512_xor_si512(D0, A0); \
        D1 = _mm512_xor_si512(D1, A1); \
\
        D0 = ror64(D0, 16); \
        D1 = ror64(D1, 16); \
\
        C0 = f(C0, D0); \
        C1 = f(C1, D1); \
\
        B0 = _mm512_xor_si512(B0, C0); \
        B1 = _mm512_xor_si512(B1, C1); \
\
        B0 = ror64(B0, 63); \
        B1 = ror64(B1, 63); \
    } while ((void)0, 0)

#define DIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm512_permutex_epi64(B0, _MM_SHUFFLE(0, 3, 2, 1)); \
        B1 = _mm512_permutex_epi64(B1, _MM_SHUFFLE(0, 3, 2, 1)); \
\
        C0 = _mm512_permutex_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        C1 = _mm512_permutex_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
\
        D0 = _mm512_permutex_epi64(D0, _MM_SHUFFLE(2, 1, 0, 3)); \
        D1 = _mm512_permutex_epi64(D1, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while ((void)0, 0)

#define UNDIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm512_permutex_epi64(B0, _MM_SHUFFLE(2, 1, 0, 3)); \
        B1 = _mm512_permutex_epi64(B1, _MM_SHUFFLE(2, 1, 0, 3)); \
\
        C0 = _mm512_permutex_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        C1 = _mm512_permutex_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
\
        D0 = _mm512_permutex_epi64(D0, _MM_SHUFFLE(0, 3, 2, 1)); \
        D1 = _mm512_permutex_epi64(D1, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while ((void)0, 0)

#define BLAKE2_ROUND(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        G1(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        DIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        G1(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        UNDIAGONALIZE(A0, B0, C0, D0, A1, B1, C1, D1); \
    } while ((void)0, 0)

#define SWAP_HALVES(A0, A1) \
    do { \
        __m512i t0, t1; \
        t0 = _mm512_shuffle_i64x2(A0, A1, _MM_SHUFFLE(1, 0, 1, 0)); \
        t1 = _mm512_shuffle_i64x2(A0, A1, _MM_SHUFFLE(3, 2, 3, 2)); \
        A0 = t0; \
        A1 = t1; \
    } while((void)0, 0)

#define SWAP_QUARTERS(A0, A1) \
    do { \
        SWAP_HALVES(A0, A1); \
        A0 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A0); \
        A1 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A1); \
    } while((void)0, 0)

#define UNSWAP_QUARTERS(A0, A1) \
    do { \
        A0 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A0); \
        A1 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), A1); \
        SWAP_HALVES(A0, A1); \
    } while((void)0, 0)

#define BLAKE2_ROUND1(A0, C0, B0, D0, A1, C1, B1, D1) \
    do { \
        SWAP_HALVES(A0, B0); \
        SWAP_HALVES(C0, D0); \
        SWAP_HALVES(A1, B1); \
        SWAP_HALVES(C1, D1); \
        BLAKE2_ROUND(A0, B0, C0, D0, A1, B1, C1, D1); \
        SWAP_HALVES(A0, B0); \
        SWAP_HALVES(C0, D0); \
        SWAP_HALVES(A1, B1); \
        SWAP_HALVES(C1, D1); \
    } while ((void)0, 0)

#define BLAKE2_ROUND2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        SWAP_QUARTERS(A0, A1); \
        SWAP_QUARTERS(B0, B1); \
        SWAP_QUARTERS(C0, C1); \
        SWAP_QUARTERS(D0, D1); \
        BLAKE2_ROUND(A0, B0, C0, D0, A1, B1, C1, D1); \
        UNSWAP_QUARTERS(A0, A1); \
        UNSWAP_QUARTERS(B0, B1); \
        UNSWAP_QUARTERS(C0, C1); \
        UNSWAP_QUARTERS(D0, D1); \
    } while ((void)0, 0)

enum {
    ARGON2_VECS_IN_BLOCK = ARGON2_OWORDS_IN_BLOCK / 4,
};

static void fill_block(__m512i *s, const block *ref_block, block *next_block,
                       int with_xor)
{
    __m512i block_XY[ARGON2_VECS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_VECS_IN_BLOCK; i++) {
            s[i] =_mm512_xor_si512(
                s[i], _mm512_loadu_si512((const __m512i *)ref_block->v + i));
            block_XY[i] = _mm512_xor_si512(
                s[i], _mm512_loadu_si512((const __m512i *)next_block->v + i));
        }

    } else {
        for (i = 0; i < ARGON2_VECS_IN_BLOCK; i++) {
            block_XY[i] = s[i] =_mm512_xor_si512(
                s[i], _mm512_loadu_si512((const __m512i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 2; ++i) {
        BLAKE2_ROUND1(
            s[8 * i + 0], s[8 * i + 1], s[8 * i + 2], s[8 * i + 3],
            s[8 * i + 4], s[8 * i + 5], s[8 * i + 6], s[8 * i + 7]);
    }

    for (i = 0; i < 2; ++i) {
        BLAKE2_ROUND2(
            s[2 * 0 + i], s[2 * 1 + i], s[2 * 2 + i], s[2 * 3 + i],
            s[2 * 4 + i], s[2 * 5 + i], s[2 * 6 + i], s[2 * 7 + i]);
    }

    for (i = 0; i < ARGON2_VECS_IN_BLOCK; i++) {
        s[i] = _mm512_xor_si512(s[i], block_XY[i]);
        _mm512_storeu_si512((__m512i *)next_block->v + i, s[i]);
    }
}

static void next_addresses(block *address_block, block *input_block)
{
    /*Temporary zero-initialized blocks*/
    __m512i zero_block[ARGON2_VECS_IN_BLOCK];
    __m512i zero2_block[ARGON2_VECS_IN_BLOCK];

    memset(zero_block, 0, sizeof(zero_block));
    memset(zero2_block, 0, sizeof(zero2_block));

    /*Increasing index counter*/
    input_block->v[6]++;

    /*First iteration of G*/
    fill_block(zero_block, input_block, address_block, 0);

    /*Second iteration of G*/
    fill_block(zero2_block, address_block, address_block, 0);
}

void xmrig_ar2_fill_segment_avx512f(const argon2_instance_t *instance, argon2_position_t position)
{
    block *ref_block = NULL, *curr_block = NULL;
    block address_block, input_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
    __m512i state[ARGON2_VECS_IN_BLOCK];
    int data_independent_addressing;

    if (instance == NULL) {
        return;
    }

    data_independent_addressing = (instance->type == Argon2_i) ||
            (instance->type == Argon2_id && (position.pass == 0) &&
             (position.slice < ARGON2_SYNC_POINTS / 2));

    if (data_independent_addressing) {
        init_block_value(&input_block, 0);

        input_block.v[0] = position.pass;
        input_block.v[1] = position.lane;
        input_block.v[2] = position.slice;
        input_block.v[3] = instance->memory_blocks;
        input_block.v[4] = instance->passes;
        input_block.v[5] = instance->type;
    }

    starting_index = 0;

    if ((0 == position.pass) && (0 == position.slice)) {
        starting_index = 2; /* we have already generated the first two blocks */

        /* Don't forget to generate the first block of addresses: */
        if (data_independent_addressing) {
            next_addresses(&address_block, &input_block);
        }
    }

    /* Offset of the current block */
    curr_offset = position.lane * instance->lane_length +
                  position.slice * instance->segment_length + starting_index;

    if (0 == curr_offset % instance->lane_length) {
        /* Last block in this lane */
        prev_offset = curr_offset + instance->lane_length - 1;
    } else {
        /* Previous block */
        prev_offset = curr_offset - 1;
    }

    memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

    for (i = starting_index; i < instance->segment_length;
         ++i, ++curr_offset, ++prev_offset) {
        /*1.1 Rotating prev_offset if needed */
        if (curr_offset % instance->lane_length == 1) {
            prev_offset = curr_offset - 1;
        }

        /* 1.2 Computing the index of the reference block */
        /* 1.2.1 Taking pseudo-random value from the previous block */
        if (data_independent_addressing) {
            if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
                next_addresses(&address_block, &input_block);
            }
            pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];
        } else {
            pseudo_rand = instance->memory[prev_offset].v[0];
        }

        /* 1.2.2 Computing the lane of the reference block */
        ref_lane = ((pseudo_rand >> 32)) % instance->lanes;

        if ((position.pass == 0) && (position.slice == 0)) {
            /* Can not reference other lanes yet */
            ref_lane = position.lane;
        }

        /* 1.2.3 Computing the number of possible reference block within the
         * lane.
         */
        position.index = i;
        ref_index = xmrig_ar2_index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF, ref_lane == position.lane);

        /* 2 Creating a new block */
        ref_block =
            instance->memory + instance->lane_length * ref_lane + ref_index;
        curr_block = instance->memory + curr_offset;

        /* version 1.2.1 and earlier: overwrite, not XOR */
        if (0 == position.pass || ARGON2_VERSION_10 == instance->version) {
            fill_block(state, ref_block, curr_block, 0);
        } else {
            fill_block(state, ref_block, curr_block, 1);
        }
    }
}

extern int cpu_flags_has_avx512f(void);
int xmrig_ar2_check_avx512f(void) { return cpu_flags_has_avx512f(); }

#else

void xmrig_ar2_fill_segment_avx512f(const argon2_instance_t *instance, argon2_position_t position) {}
int xmrig_ar2_check_avx512f(void) { return 0; }

#endif
