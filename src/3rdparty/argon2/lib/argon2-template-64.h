#include <string.h>

#include "core.h"

#define MASK_32 UINT64_C(0xFFFFFFFF)

#define F(x, y) ((x) + (y) + 2 * ((x) & MASK_32) * ((y) & MASK_32))

#define G(a, b, c, d) \
    do { \
        a = F(a, b); \
        d = rotr64(d ^ a, 32); \
        c = F(c, d); \
        b = rotr64(b ^ c, 24); \
        a = F(a, b); \
        d = rotr64(d ^ a, 16); \
        c = F(c, d); \
        b = rotr64(b ^ c, 63); \
    } while ((void)0, 0)

#define BLAKE2_ROUND_NOMSG(v0, v1, v2, v3, v4, v5, v6, v7, \
                           v8, v9, v10, v11, v12, v13, v14, v15) \
    do { \
        G(v0, v4, v8,  v12); \
        G(v1, v5, v9,  v13); \
        G(v2, v6, v10, v14); \
        G(v3, v7, v11, v15); \
        G(v0, v5, v10, v15); \
        G(v1, v6, v11, v12); \
        G(v2, v7, v8,  v13); \
        G(v3, v4, v9,  v14); \
    } while ((void)0, 0)

#define BLAKE2_ROUND_NOMSG1(v) \
    BLAKE2_ROUND_NOMSG( \
        (v)[ 0], (v)[ 1], (v)[ 2], (v)[ 3], \
        (v)[ 4], (v)[ 5], (v)[ 6], (v)[ 7], \
        (v)[ 8], (v)[ 9], (v)[10], (v)[11], \
        (v)[12], (v)[13], (v)[14], (v)[15])

#define BLAKE2_ROUND_NOMSG2(v) \
    BLAKE2_ROUND_NOMSG( \
        (v)[  0], (v)[  1], (v)[ 16], (v)[ 17], \
        (v)[ 32], (v)[ 33], (v)[ 48], (v)[ 49], \
        (v)[ 64], (v)[ 65], (v)[ 80], (v)[ 81], \
        (v)[ 96], (v)[ 97], (v)[112], (v)[113])

static void fill_block(const block *prev_block, const block *ref_block,
                       block *next_block, int with_xor)
{
    block blockR, block_tmp;

    copy_block(&blockR, ref_block);
    xor_block(&blockR, prev_block);
    copy_block(&block_tmp, &blockR);
    if (with_xor) {
        xor_block(&block_tmp, next_block);
    }

    /* Apply Blake2 on columns of 64-bit words: (0,1,...,15) , then
    (16,17,..31)... finally (112,113,...127) */
    BLAKE2_ROUND_NOMSG1(blockR.v + 0 * 16);
    BLAKE2_ROUND_NOMSG1(blockR.v + 1 * 16);
    BLAKE2_ROUND_NOMSG1(blockR.v + 2 * 16);
    BLAKE2_ROUND_NOMSG1(blockR.v + 3 * 16);
    BLAKE2_ROUND_NOMSG1(blockR.v + 4 * 16);
    BLAKE2_ROUND_NOMSG1(blockR.v + 5 * 16);
    BLAKE2_ROUND_NOMSG1(blockR.v + 6 * 16);
    BLAKE2_ROUND_NOMSG1(blockR.v + 7 * 16);

    /* Apply Blake2 on rows of 64-bit words: (0,1,16,17,...112,113), then
    (2,3,18,19,...,114,115).. finally (14,15,30,31,...,126,127) */
    BLAKE2_ROUND_NOMSG2(blockR.v + 0 * 2);
    BLAKE2_ROUND_NOMSG2(blockR.v + 1 * 2);
    BLAKE2_ROUND_NOMSG2(blockR.v + 2 * 2);
    BLAKE2_ROUND_NOMSG2(blockR.v + 3 * 2);
    BLAKE2_ROUND_NOMSG2(blockR.v + 4 * 2);
    BLAKE2_ROUND_NOMSG2(blockR.v + 5 * 2);
    BLAKE2_ROUND_NOMSG2(blockR.v + 6 * 2);
    BLAKE2_ROUND_NOMSG2(blockR.v + 7 * 2);

    copy_block(next_block, &block_tmp);
    xor_block(next_block, &blockR);
}

static void next_addresses(block *address_block, block *input_block,
                           const block *zero_block)
{
    input_block->v[6]++;
    fill_block(zero_block, input_block, address_block, 0);
    fill_block(zero_block, address_block, address_block, 0);
}

static void fill_segment_64(const argon2_instance_t *instance,
                            argon2_position_t position)
{
    block *ref_block, *curr_block, *prev_block;
    block address_block, input_block, zero_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
    int data_independent_addressing;

    if (instance == NULL) {
        return;
    }

    data_independent_addressing = (instance->type == Argon2_i) ||
            (instance->type == Argon2_id && (position.pass == 0) &&
             (position.slice < ARGON2_SYNC_POINTS / 2));

    if (data_independent_addressing) {
        init_block_value(&zero_block, 0);
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
            next_addresses(&address_block, &input_block, &zero_block);
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
                next_addresses(&address_block, &input_block, &zero_block);
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
        prev_block = instance->memory + prev_offset;

        /* version 1.2.1 and earlier: overwrite, not XOR */
        if (0 == position.pass || ARGON2_VERSION_10 == instance->version) {
            fill_block(prev_block, ref_block, curr_block, 0);
        } else {
            fill_block(prev_block, ref_block, curr_block, 1);
        }
    }
}
