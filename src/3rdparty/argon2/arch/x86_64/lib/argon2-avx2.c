#include "argon2-avx2.h"

#ifdef HAVE_AVX2
#include <string.h>

#include <x86intrin.h>

#include "cpu-flags.h"

#define r16 (_mm256_setr_epi8( \
     2,  3,  4,  5,  6,  7,  0,  1, \
    10, 11, 12, 13, 14, 15,  8,  9, \
    18, 19, 20, 21, 22, 23, 16, 17, \
    26, 27, 28, 29, 30, 31, 24, 25))

#define r24 (_mm256_setr_epi8( \
     3,  4,  5,  6,  7,  0,  1,  2, \
    11, 12, 13, 14, 15,  8,  9, 10, \
    19, 20, 21, 22, 23, 16, 17, 18, \
    27, 28, 29, 30, 31, 24, 25, 26))

#define ror64_16(x) _mm256_shuffle_epi8((x), r16)
#define ror64_24(x) _mm256_shuffle_epi8((x), r24)
#define ror64_32(x) _mm256_shuffle_epi32((x), _MM_SHUFFLE(2, 3, 0, 1))
#define ror64_63(x) \
    _mm256_xor_si256(_mm256_srli_epi64((x), 63), _mm256_add_epi64((x), (x)))

static __m256i f(__m256i x, __m256i y)
{
    __m256i z = _mm256_mul_epu32(x, y);
    return _mm256_add_epi64(_mm256_add_epi64(x, y), _mm256_add_epi64(z, z));
}

#define G1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = f(A0, B0); \
        A1 = f(A1, B1); \
\
        D0 = _mm256_xor_si256(D0, A0); \
        D1 = _mm256_xor_si256(D1, A1); \
\
        D0 = ror64_32(D0); \
        D1 = ror64_32(D1); \
\
        C0 = f(C0, D0); \
        C1 = f(C1, D1); \
\
        B0 = _mm256_xor_si256(B0, C0); \
        B1 = _mm256_xor_si256(B1, C1); \
\
        B0 = ror64_24(B0); \
        B1 = ror64_24(B1); \
    } while ((void)0, 0)

#define G2(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        A0 = f(A0, B0); \
        A1 = f(A1, B1); \
\
        D0 = _mm256_xor_si256(D0, A0); \
        D1 = _mm256_xor_si256(D1, A1); \
\
        D0 = ror64_16(D0); \
        D1 = ror64_16(D1); \
\
        C0 = f(C0, D0); \
        C1 = f(C1, D1); \
\
        B0 = _mm256_xor_si256(B0, C0); \
        B1 = _mm256_xor_si256(B1, C1); \
\
        B0 = ror64_63(B0); \
        B1 = ror64_63(B1); \
    } while ((void)0, 0)

#define DIAGONALIZE1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm256_permute4x64_epi64(B0, _MM_SHUFFLE(0, 3, 2, 1)); \
        B1 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(0, 3, 2, 1)); \
\
        C0 = _mm256_permute4x64_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        C1 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
\
        D0 = _mm256_permute4x64_epi64(D0, _MM_SHUFFLE(2, 1, 0, 3)); \
        D1 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(2, 1, 0, 3)); \
    } while ((void)0, 0)

#define UNDIAGONALIZE1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        B0 = _mm256_permute4x64_epi64(B0, _MM_SHUFFLE(2, 1, 0, 3)); \
        B1 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(2, 1, 0, 3)); \
\
        C0 = _mm256_permute4x64_epi64(C0, _MM_SHUFFLE(1, 0, 3, 2)); \
        C1 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(1, 0, 3, 2)); \
\
        D0 = _mm256_permute4x64_epi64(D0, _MM_SHUFFLE(0, 3, 2, 1)); \
        D1 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(0, 3, 2, 1)); \
    } while ((void)0, 0)

#define DIAGONALIZE2(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        __m256i tmp1, tmp2; \
        tmp1 = _mm256_blend_epi32(B0, B1, 0xCC); \
        tmp2 = _mm256_blend_epi32(B0, B1, 0x33); \
        B1 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        B0 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
\
        tmp1 = C0; \
        C0 = C1; \
        C1 = tmp1; \
\
        tmp1 = _mm256_blend_epi32(D0, D1, 0xCC); \
        tmp2 = _mm256_blend_epi32(D0, D1, 0x33); \
        D0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        D1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
    } while ((void)0, 0)

#define UNDIAGONALIZE2(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        __m256i tmp1, tmp2; \
        tmp1 = _mm256_blend_epi32(B0, B1, 0xCC); \
        tmp2 = _mm256_blend_epi32(B0, B1, 0x33); \
        B0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        B1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
\
        tmp1 = C0; \
        C0 = C1; \
        C1 = tmp1; \
\
        tmp1 = _mm256_blend_epi32(D0, D1, 0xCC); \
        tmp2 = _mm256_blend_epi32(D0, D1, 0x33); \
        D1 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); \
        D0 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); \
    } while ((void)0, 0)

#define BLAKE2_ROUND1(A0, B0, C0, D0, A1, B1, C1, D1) \
    do { \
        G1(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        DIAGONALIZE1(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        G1(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        UNDIAGONALIZE1(A0, B0, C0, D0, A1, B1, C1, D1); \
    } while ((void)0, 0)

#define BLAKE2_ROUND2(A0, A1, B0, B1, C0, C1, D0, D1) \
    do { \
        G1(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        DIAGONALIZE2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        G1(A0, B0, C0, D0, A1, B1, C1, D1); \
        G2(A0, B0, C0, D0, A1, B1, C1, D1); \
\
        UNDIAGONALIZE2(A0, B0, C0, D0, A1, B1, C1, D1); \
    } while ((void)0, 0)

enum {
    ARGON2_HWORDS_IN_BLOCK = ARGON2_OWORDS_IN_BLOCK / 2,
};

static void fill_block(__m256i *s, const block *ref_block, block *next_block,
                       int with_xor)
{
    __m256i block_XY[ARGON2_HWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            s[i] =_mm256_xor_si256(
                s[i], _mm256_loadu_si256((const __m256i *)ref_block->v + i));
            block_XY[i] = _mm256_xor_si256(
                s[i], _mm256_loadu_si256((const __m256i *)next_block->v + i));
        }

    } else {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            block_XY[i] = s[i] =_mm256_xor_si256(
                s[i], _mm256_loadu_si256((const __m256i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND1(
            s[8 * i + 0], s[8 * i + 1], s[8 * i + 2], s[8 * i + 3],
            s[8 * i + 4], s[8 * i + 5], s[8 * i + 6], s[8 * i + 7]);
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND2(
            s[4 * 0 + i], s[4 * 1 + i], s[4 * 2 + i], s[4 * 3 + i],
            s[4 * 4 + i], s[4 * 5 + i], s[4 * 6 + i], s[4 * 7 + i]);
    }

    for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
        s[i] = _mm256_xor_si256(s[i], block_XY[i]);
        _mm256_storeu_si256((__m256i *)next_block->v + i, s[i]);
    }
}

static void next_addresses(block *address_block, block *input_block)
{
    /*Temporary zero-initialized blocks*/
    __m256i zero_block[ARGON2_HWORDS_IN_BLOCK];
    __m256i zero2_block[ARGON2_HWORDS_IN_BLOCK];

    memset(zero_block, 0, sizeof(zero_block));
    memset(zero2_block, 0, sizeof(zero2_block));

    /*Increasing index counter*/
    input_block->v[6]++;

    /*First iteration of G*/
    fill_block(zero_block, input_block, address_block, 0);

    /*Second iteration of G*/
    fill_block(zero2_block, address_block, address_block, 0);
}

void fill_segment_avx2(const argon2_instance_t *instance,
                       argon2_position_t position)
{
    block *ref_block = NULL, *curr_block = NULL;
    block address_block, input_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
    __m256i state[ARGON2_HWORDS_IN_BLOCK];
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
        ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
                                ref_lane == position.lane);

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

int check_avx2(void)
{
    return cpu_flags_have_avx2();
}

#else

void fill_segment_avx2(const argon2_instance_t *instance,
                       argon2_position_t position)
{
}

int check_avx2(void)
{
    return 0;
}

#endif
