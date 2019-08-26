//
// Created by Haifa Bogdan Adnan on 06/08/2018.
//

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../../../common/DLLImport.h"
#include "../../argon2/Defs.h"
#include "../../../common/DLLExport.h"

#if !defined(BUILD_REF) && (defined(__x86_64__) || defined(_WIN64) || defined(__NEON__))
#include "blamka-round-opt.h"
#else
#include "blamka-round-ref.h"
#endif

void copy_block(block *dst, const block *src) {
    memcpy(dst->v, src->v, sizeof(uint64_t) * ARGON2_QWORDS_IN_BLOCK);
}

void xor_block(block *dst, const block *src) {
    int i;
    for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
        dst->v[i] ^= src->v[i];
    }
}

#ifndef BUILD_REF

#if defined(__AVX512F__)
static void fill_block(__m512i *state, const block *ref_block,
                       block *next_block, int with_xor, int keep) {
    __m512i block_XY[ARGON2_512BIT_WORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
            state[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i *)ref_block->v + i));
            block_XY[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 2; ++i) {
        BLAKE2_ROUND_1(
            state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
            state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 2; ++i) {
        BLAKE2_ROUND_2(
            state[2 * 0 + i], state[2 * 1 + i], state[2 * 2 + i], state[2 * 3 + i],
            state[2 * 4 + i], state[2 * 5 + i], state[2 * 6 + i], state[2 * 7 + i]);
    }

    if(keep) {
        for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
            state[i] = _mm512_xor_si512(state[i], block_XY[i]);
            _mm512_storeu_si512((__m512i *)next_block->v + i, state[i]);
        }
    }
    else {
        for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
            state[i] = _mm512_xor_si512(state[i], block_XY[i]);
        }
    }
}
#elif defined(__AVX2__)
static void fill_block(__m256i *state, const block *ref_block,
                       block *next_block, int with_xor, int keep) {
    __m256i block_XY[ARGON2_HWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            state[i] = _mm256_xor_si256(
                    state[i], _mm256_loadu_si256((const __m256i *)ref_block->v + i));
            block_XY[i] = _mm256_xor_si256(
                    state[i], _mm256_loadu_si256((const __m256i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm256_xor_si256(
                    state[i], _mm256_loadu_si256((const __m256i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_1(state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
                       state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]);
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_2(state[ 0 + i], state[ 4 + i], state[ 8 + i], state[12 + i],
                       state[16 + i], state[20 + i], state[24 + i], state[28 + i]);
    }

    if(keep) {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            state[i] = _mm256_xor_si256(state[i], block_XY[i]);
            _mm256_store_si256((__m256i *)next_block->v + i, state[i]);
        }
    }
    else {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            state[i] = _mm256_xor_si256(state[i], block_XY[i]);
        }
    }
}
#elif defined(__AVX__)

#define I2D(x) _mm256_castsi256_pd(x)
#define D2I(x) _mm256_castpd_si256(x)

static void fill_block(__m128i *state, const block *ref_block,
                       block *next_block, int with_xor, int keep) {
    __m128i block_XY[ARGON2_OWORDS_IN_BLOCK];
    unsigned int i;

    __m256i t;
    __m256i *s256 = (__m256i *) state, *block256 = (__m256i *) block_XY;

    if (with_xor) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK / 2; i++) {
            t = D2I(_mm256_xor_pd(I2D(_mm256_loadu_si256(s256 + i)), \
                I2D(_mm256_loadu_si256((const __m256i *)ref_block->v + i))));
            _mm256_storeu_si256(s256 + i, t);
            t = D2I(_mm256_xor_pd(I2D(t), \
                I2D(_mm256_loadu_si256((const __m256i *)next_block->v + i))));
            _mm256_storeu_si256(block256 + i, t);
        }
    } else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK / 2; i++) {
            t = D2I(_mm256_xor_pd(I2D(_mm256_loadu_si256(s256 + i)), \
                I2D(_mm256_loadu_si256((const __m256i *)ref_block->v + i))));
            _mm256_storeu_si256(s256 + i, t);
            _mm256_storeu_si256(block256 + i, t);
        }
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
                     state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
                     state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
                     state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
                     state[8 * 6 + i], state[8 * 7 + i]);
    }

    if(keep) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK / 2; i++) {
            t = D2I(_mm256_xor_pd(I2D(_mm256_loadu_si256(s256 + i)), \
                I2D(_mm256_loadu_si256(block256 + i))));

            _mm256_storeu_si256(s256 + i, t);
            _mm256_storeu_si256((__m256i *)next_block->v + i, t);
        }
    }
    else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK / 2; i++) {
            t = D2I(_mm256_xor_pd(I2D(_mm256_loadu_si256(s256 + i)), \
                I2D(_mm256_loadu_si256(block256 + i))));

            _mm256_storeu_si256(s256 + i, t);
        }
    }

}
#elif defined(__NEON__)
static void fill_block(uint64x2_t *state, const block *ref_block,
                       block *next_block, int with_xor, int keep) {
    uint64x2_t block_XY[ARGON2_OWORDS_IN_BLOCK];
    uint64x2_t t0, t1;

    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = veorq_u64(state[i], vld1q_u64(ref_block->v + i*2));
            block_XY[i] = veorq_u64(state[i], vld1q_u64(next_block->v + i*2));
        }
    } else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = veorq_u64(state[i], vld1q_u64(ref_block->v + i*2));
        }
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
                     state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
                     state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
                     state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
                     state[8 * 6 + i], state[8 * 7 + i]);
    }

    if(keep) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = veorq_u64(state[i], block_XY[i]);
            vst1q_u64(next_block->v + i*2, state[i]);
        }
    }
    else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = veorq_u64(state[i], block_XY[i]);
        }
    }
}
#else
static void fill_block(__m128i *state, const block *ref_block,
                       block *next_block, int with_xor, int keep) {
    __m128i block_XY[ARGON2_OWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
            block_XY[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
            state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
            state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
            state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
            state[8 * 6 + i], state[8 * 7 + i]);
    }

    if(keep) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(state[i], block_XY[i]);
            _mm_storeu_si128((__m128i *)next_block->v + i, state[i]);
        }
    }
    else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(state[i], block_XY[i]);
        }
    }
}
#endif

#else
static void fill_block(block *prev_block, const block *ref_block,
                       block *next_block, int with_xor, int keep) {
    block block_tmp;
    unsigned i;

    xor_block(prev_block, ref_block);
    copy_block(&block_tmp, prev_block);

    if (with_xor && next_block != NULL) {
        xor_block(&block_tmp, next_block);
    }

    /* Apply Blake2 on columns of 64-bit words: (0,1,...,15) , then
     (16,17,..31)... finally (112,113,...127) */
    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND_NOMSG(
                           prev_block->v[16 * i], prev_block->v[16 * i + 1], prev_block->v[16 * i + 2],
                           prev_block->v[16 * i + 3], prev_block->v[16 * i + 4], prev_block->v[16 * i + 5],
                           prev_block->v[16 * i + 6], prev_block->v[16 * i + 7], prev_block->v[16 * i + 8],
                           prev_block->v[16 * i + 9], prev_block->v[16 * i + 10], prev_block->v[16 * i + 11],
                           prev_block->v[16 * i + 12], prev_block->v[16 * i + 13], prev_block->v[16 * i + 14],
                           prev_block->v[16 * i + 15]);
    }

    /* Apply Blake2 on rows of 64-bit words: (0,1,16,17,...112,113), then
     (2,3,18,19,...,114,115).. finally (14,15,30,31,...,126,127) */
    for (i = 0; i < 8; i++) {
        BLAKE2_ROUND_NOMSG(
                           prev_block->v[2 * i], prev_block->v[2 * i + 1], prev_block->v[2 * i + 16],
                           prev_block->v[2 * i + 17], prev_block->v[2 * i + 32], prev_block->v[2 * i + 33],
                           prev_block->v[2 * i + 48], prev_block->v[2 * i + 49], prev_block->v[2 * i + 64],
                           prev_block->v[2 * i + 65], prev_block->v[2 * i + 80], prev_block->v[2 * i + 81],
                           prev_block->v[2 * i + 96], prev_block->v[2 * i + 97], prev_block->v[2 * i + 112],
                           prev_block->v[2 * i + 113]);
    }

    xor_block(prev_block, &block_tmp);
    if(keep)
        copy_block(next_block, prev_block);
}

#endif

DLLEXPORT void *fill_memory_blocks(int threads, Argon2Profile *profile, void *user_data) {
    void *memory = user_data;
#ifndef  BUILD_REF
#if defined(__AVX512F__)
    __m512i state[ARGON2_512BIT_WORDS_IN_BLOCK];
    uint64_t buff_512[8];
#elif defined(__AVX2__)
    __m256i state[ARGON2_HWORDS_IN_BLOCK];
    uint64_t buff_256[4];
#elif defined(__x86_64__) || defined(_WIN64)
    __m128i state[ARGON2_OWORDS_IN_BLOCK];
#elif defined(__NEON__)
    uint64x2_t state[ARGON2_OWORDS_IN_BLOCK];
#endif
#else
    block state_;
    block *state = &state_;
#endif
    int lane_length = profile->segSize * 4;
    int seg_length = profile->segSize;
    int suc_idx = profile->succesiveIdxs;

    for(int thr = 0; thr < threads;thr++) {
        block *ref_block = NULL, *curr_block = NULL;

        int32_t ref_idx = 0;
        int32_t cur_idx = 0;
        int32_t prev_idx = 0;
        int32_t seg_type = 0;
        int32_t idx = 0;
        int32_t keep = 1;
        int32_t with_xor = 0;

        block *blocks = (block *)((uint8_t*)memory + thr * profile->memSize);

        int32_t *address = profile->blockRefs;

        for(uint32_t s = 0; s < profile->segCount; s++) {
            cur_idx = profile->segments[s * 3];
            prev_idx = profile->segments[s * 3 + 1];
            seg_type = profile->segments[s * 3 + 2];
            keep = 1;
            with_xor = (s >= profile->thrCost * 4) ? 1 : 0;

            idx = (s < profile->thrCost) ? 2 : 0;

            int32_t lane = s % profile->thrCost;
            int32_t slice = (s / profile->thrCost) % 4;
            int32_t pass = (s / profile->thrCost) / 4;

            memcpy(state, (void *) (blocks + prev_idx), ARGON2_BLOCK_SIZE);

            if(seg_type == 0) {
                if(s < profile->thrCost)
                    address = &profile->blockRefs[(s * (profile->segSize - 2)) * 3];
                else
                    address = &profile->blockRefs[(profile->thrCost * (profile->segSize - 2) + (s - profile->thrCost) * profile->segSize) * 3];
            }

            for (int i = idx; i < seg_length; ++i, cur_idx ++) {
                if (seg_type == 1) { // data dependent addressing
#ifndef  BUILD_REF
#if defined(__AVX512F__)
                    _mm512_storeu_si512(buff_512, state[0]);
                    uint64_t pseudo_rand = buff_512[0];
#elif defined(__AVX2__)
                    _mm256_storeu_si256(buff_256, state[0]);
                    uint64_t pseudo_rand = buff_256[0];
#elif defined(__x86_64__) || defined(_WIN64)
                    uint64_t pseudo_rand = _mm_cvtsi128_si64(state[0]);
#elif defined(__NEON__)
                    uint64_t pseudo_rand = 0;
                    vst1q_lane_u64(&pseudo_rand, state[0], 0);
#endif
#else
                    uint64_t pseudo_rand = state->v[0];
#endif
                    uint64_t ref_lane = ((pseudo_rand >> 32)) % profile->thrCost;
                    uint32_t reference_area_size = 0;
                    if(pass > 0) {
                        if (lane == ref_lane) {
                            reference_area_size = lane_length - seg_length + i - 1;
                        } else {
                            reference_area_size = lane_length - seg_length + ((i == 0) ? (-1) : 0);
                        }
                    }
                    else {
                        if (lane == ref_lane) {
                            reference_area_size = slice * seg_length + i - 1;
                        } else {
                            reference_area_size = slice * seg_length + ((i == 0) ? (-1) : 0);
                        }
                    }
                    uint64_t relative_position = pseudo_rand & 0xFFFFFFFF;
                    relative_position = relative_position * relative_position >> 32;

                    relative_position = reference_area_size - 1 -
                                        (reference_area_size * relative_position >> 32);

                    ref_idx = ref_lane * lane_length + (((pass > 0 && slice < 3) ? ((slice + 1) * seg_length) : 0) + relative_position) % lane_length;
                }
                else {
                    ref_idx = address[1];
                    if(suc_idx == 0)
                        cur_idx = address[0];
                    keep = address[2];

                    address += 3;
                }

                ref_block = blocks + ref_idx;
                curr_block = blocks + cur_idx;

                fill_block(state, ref_block, curr_block, with_xor, keep);
            }
        }

        uint32_t dst = -1;
        for(; address < (profile->blockRefs + profile->blockRefsSize * 3); address += 3) {
            if (address[2] == -1) {
                curr_block = blocks + address[0];
                ref_block = blocks + address[1];
                dst = address[0];
                xor_block(curr_block, ref_block);
            }
        }
        if(dst != -1)
            copy_block(blocks, blocks + dst);
        else
            copy_block(blocks, state);
    }

    return memory;
}

