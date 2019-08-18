/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Original code from Argon2 reference source code package used under CC0 Licence
 * https://github.com/P-H-C/phc-winner-argon2
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
*/

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "crypto/randomx/argon2.h"
#include "crypto/randomx/argon2_core.h"

#include "crypto/randomx/blake2/blamka-round-ref.h"
#include "crypto/randomx/blake2/blake2-impl.h"
#include "crypto/randomx/blake2/blake2.h"

 /*
  * Function fills a new memory block and optionally XORs the old block over the new one.
  * @next_block must be initialized.
  * @param prev_block Pointer to the previous block
  * @param ref_block Pointer to the reference block
  * @param next_block Pointer to the block to be constructed
  * @param with_xor Whether to XOR into the new block (1) or just overwrite (0)
  * @pre all block pointers must be valid
  */
static void fill_block(const block *prev_block, const block *ref_block,
	block *next_block, int with_xor) {
	block blockR, block_tmp;
	unsigned i;

	rxa2_copy_block(&blockR, ref_block);
	rxa2_xor_block(&blockR, prev_block);
	rxa2_copy_block(&block_tmp, &blockR);
	/* Now blockR = ref_block + prev_block and block_tmp = ref_block + prev_block */
	if (with_xor) {
		/* Saving the next block contents for XOR over: */
		rxa2_xor_block(&block_tmp, next_block);
		/* Now blockR = ref_block + prev_block and
		   block_tmp = ref_block + prev_block + next_block */
	}

	/* Apply Blake2 on columns of 64-bit words: (0,1,...,15) , then
	   (16,17,..31)... finally (112,113,...127) */
	for (i = 0; i < 8; ++i) {
		BLAKE2_ROUND_NOMSG(
			blockR.v[16 * i], blockR.v[16 * i + 1], blockR.v[16 * i + 2],
			blockR.v[16 * i + 3], blockR.v[16 * i + 4], blockR.v[16 * i + 5],
			blockR.v[16 * i + 6], blockR.v[16 * i + 7], blockR.v[16 * i + 8],
			blockR.v[16 * i + 9], blockR.v[16 * i + 10], blockR.v[16 * i + 11],
			blockR.v[16 * i + 12], blockR.v[16 * i + 13], blockR.v[16 * i + 14],
			blockR.v[16 * i + 15]);
	}

	/* Apply Blake2 on rows of 64-bit words: (0,1,16,17,...112,113), then
	   (2,3,18,19,...,114,115).. finally (14,15,30,31,...,126,127) */
	for (i = 0; i < 8; i++) {
		BLAKE2_ROUND_NOMSG(
			blockR.v[2 * i], blockR.v[2 * i + 1], blockR.v[2 * i + 16],
			blockR.v[2 * i + 17], blockR.v[2 * i + 32], blockR.v[2 * i + 33],
			blockR.v[2 * i + 48], blockR.v[2 * i + 49], blockR.v[2 * i + 64],
			blockR.v[2 * i + 65], blockR.v[2 * i + 80], blockR.v[2 * i + 81],
			blockR.v[2 * i + 96], blockR.v[2 * i + 97], blockR.v[2 * i + 112],
			blockR.v[2 * i + 113]);
	}

	rxa2_copy_block(next_block, &block_tmp);
	rxa2_xor_block(next_block, &blockR);
}

static void next_addresses(block *address_block, block *input_block,
	const block *zero_block) {
	input_block->v[6]++;
	fill_block(zero_block, input_block, address_block, 0);
	fill_block(zero_block, address_block, address_block, 0);
}

void rxa2_fill_segment(const argon2_instance_t *instance,
	argon2_position_t position) {
	block *ref_block = NULL, *curr_block = NULL;
	block address_block, input_block, zero_block;
	uint64_t pseudo_rand, ref_index, ref_lane;
	uint32_t prev_offset, curr_offset;
	uint32_t starting_index;
	uint32_t i;
	int data_independent_addressing;

	if (instance == NULL) {
		return;
	}

	data_independent_addressing =
		(instance->type == Argon2_i) ||
		(instance->type == Argon2_id && (position.pass == 0) &&
		(position.slice < ARGON2_SYNC_POINTS / 2));

	if (data_independent_addressing) {
		rxa2_init_block_value(&zero_block, 0);
		rxa2_init_block_value(&input_block, 0);

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
	}
	else {
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
		}
		else {
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
		ref_index = rxa2_index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
			ref_lane == position.lane);

		/* 2 Creating a new block */
		ref_block =
			instance->memory + instance->lane_length * ref_lane + ref_index;
		curr_block = instance->memory + curr_offset;
		if (ARGON2_VERSION_10 == instance->version) {
			/* version 1.2.1 and earlier: overwrite, not XOR */
			fill_block(instance->memory + prev_offset, ref_block, curr_block, 0);
		}
		else {
			if (0 == position.pass) {
				fill_block(instance->memory + prev_offset, ref_block,
					curr_block, 0);
			}
			else {
				fill_block(instance->memory + prev_offset, ref_block,
					curr_block, 1);
			}
		}
	}
}
