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

 /*For memory wiping*/
#ifdef _MSC_VER
#include <windows.h>
#include <winbase.h> /* For SecureZeroMemory */
#endif
#if defined __STDC_LIB_EXT1__
#define __STDC_WANT_LIB_EXT1__ 1
#endif
#define VC_GE_2005(version) (version >= 1400)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "crypto/randomx/argon2_core.h"
#include "crypto/randomx/blake2/blake2.h"
#include "crypto/randomx/blake2/blake2-impl.h"

#ifdef GENKAT
#include "genkat.h"
#endif

#if defined(__clang__)
#if __has_attribute(optnone)
#define NOT_OPTIMIZED __attribute__((optnone))
#endif
#elif defined(__GNUC__)
#define GCC_VERSION                                                            \
    (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if GCC_VERSION >= 40400
#define NOT_OPTIMIZED __attribute__((optimize("O0")))
#endif
#endif
#ifndef NOT_OPTIMIZED
#define NOT_OPTIMIZED
#endif

/***************Instance and Position constructors**********/
void rxa2_init_block_value(block *b, uint8_t in) { memset(b->v, in, sizeof(b->v)); }

void rxa2_copy_block(block *dst, const block *src) {
	memcpy(dst->v, src->v, sizeof(uint64_t) * ARGON2_QWORDS_IN_BLOCK);
}

void rxa2_xor_block(block *dst, const block *src) {
	int i;
	for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
		dst->v[i] ^= src->v[i];
	}
}

static void load_block(block *dst, const void *input) {
	unsigned i;
	for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
		dst->v[i] = load64((const uint8_t *)input + i * sizeof(dst->v[i]));
	}
}

//static void store_block(void *output, const block *src) {
//	unsigned i;
//	for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
//		store64((uint8_t *)output + i * sizeof(src->v[i]), src->v[i]);
//	}
//}

/***************Memory functions*****************/

int rxa2_allocate_memory(const argon2_context *context, uint8_t **memory,
	size_t num, size_t size) {
	size_t memory_size = num * size;
	if (memory == NULL) {
		return ARGON2_MEMORY_ALLOCATION_ERROR;
	}

	/* 1. Check for multiplication overflow */
	if (size != 0 && memory_size / size != num) {
		return ARGON2_MEMORY_ALLOCATION_ERROR;
	}

	/* 2. Try to allocate with appropriate allocator */
	if (context->allocate_cbk) {
		(context->allocate_cbk)(memory, memory_size);
	}
	else {
		*memory = (uint8_t*)malloc(memory_size);
	}

	if (*memory == NULL) {
		return ARGON2_MEMORY_ALLOCATION_ERROR;
	}

	return ARGON2_OK;
}

void rxa2_free_memory(const argon2_context *context, uint8_t *memory,
	size_t num, size_t size) {
	size_t memory_size = num * size;
	rxa2_clear_internal_memory(memory, memory_size);
	if (context->free_cbk) {
		(context->free_cbk)(memory, memory_size);
	}
	else {
		free(memory);
	}
}

void NOT_OPTIMIZED rxa2_secure_wipe_memory(void *v, size_t n) {
#if defined(_MSC_VER) && VC_GE_2005(_MSC_VER)
	SecureZeroMemory(v, n);
#elif defined memset_s
	memset_s(v, n, 0, n);
#elif defined(__OpenBSD__)
	explicit_bzero(v, n);
#else
	static void *(*const volatile memset_sec)(void *, int, size_t) = &memset;
	memset_sec(v, 0, n);
#endif
}

/* Memory clear flag defaults to true. */
#define FLAG_clear_internal_memory 0
void rxa2_clear_internal_memory(void *v, size_t n) {
	if (FLAG_clear_internal_memory && v) {
		rxa2_secure_wipe_memory(v, n);
	}
}

uint32_t rxa2_index_alpha(const argon2_instance_t *instance,
	const argon2_position_t *position, uint32_t pseudo_rand,
	int same_lane) {
	/*
	 * Pass 0:
	 *      This lane : all already finished segments plus already constructed
	 * blocks in this segment
	 *      Other lanes : all already finished segments
	 * Pass 1+:
	 *      This lane : (SYNC_POINTS - 1) last segments plus already constructed
	 * blocks in this segment
	 *      Other lanes : (SYNC_POINTS - 1) last segments
	 */
	uint32_t reference_area_size;
	uint64_t relative_position;
	uint32_t start_position, absolute_position;

	if (0 == position->pass) {
		/* First pass */
		if (0 == position->slice) {
			/* First slice */
			reference_area_size =
				position->index - 1; /* all but the previous */
		}
		else {
			if (same_lane) {
				/* The same lane => add current segment */
				reference_area_size =
					position->slice * instance->segment_length +
					position->index - 1;
			}
			else {
				reference_area_size =
					position->slice * instance->segment_length +
					((position->index == 0) ? (-1) : 0);
			}
		}
	}
	else {
		/* Second pass */
		if (same_lane) {
			reference_area_size = instance->lane_length -
				instance->segment_length + position->index -
				1;
		}
		else {
			reference_area_size = instance->lane_length -
				instance->segment_length +
				((position->index == 0) ? (-1) : 0);
		}
	}

	/* 1.2.4. Mapping pseudo_rand to 0..<reference_area_size-1> and produce
	 * relative position */
	relative_position = pseudo_rand;
	relative_position = relative_position * relative_position >> 32;
	relative_position = reference_area_size - 1 -
		(reference_area_size * relative_position >> 32);

	/* 1.2.5 Computing starting position */
	start_position = 0;

	if (0 != position->pass) {
		start_position = (position->slice == ARGON2_SYNC_POINTS - 1)
			? 0
			: (position->slice + 1) * instance->segment_length;
	}

	/* 1.2.6. Computing absolute position */
	absolute_position = (start_position + relative_position) %
		instance->lane_length; /* absolute position */
	return absolute_position;
}

/* Single-threaded version for p=1 case */
static int fill_memory_blocks_st(argon2_instance_t *instance) {
	uint32_t r, s, l;

	for (r = 0; r < instance->passes; ++r) {
		for (s = 0; s < ARGON2_SYNC_POINTS; ++s) {
			for (l = 0; l < instance->lanes; ++l) {
				argon2_position_t position = { r, l, (uint8_t)s, 0 };
				rxa2_fill_segment(instance, position);
			}
		}
#ifdef GENKAT
		internal_kat(instance, r); /* Print all memory blocks */
#endif
	}
	return ARGON2_OK;
}

int rxa2_fill_memory_blocks(argon2_instance_t *instance) {
	if (instance == NULL || instance->lanes == 0) {
		return ARGON2_INCORRECT_PARAMETER;
	}
	return fill_memory_blocks_st(instance);
}

int rxa2_validate_inputs(const argon2_context *context) {
	if (NULL == context) {
		return ARGON2_INCORRECT_PARAMETER;
	}

	/* Validate password (required param) */
	if (NULL == context->pwd) {
		if (0 != context->pwdlen) {
			return ARGON2_PWD_PTR_MISMATCH;
		}
	}

	if (ARGON2_MIN_PWD_LENGTH > context->pwdlen) {
		return ARGON2_PWD_TOO_SHORT;
	}

	if (ARGON2_MAX_PWD_LENGTH < context->pwdlen) {
		return ARGON2_PWD_TOO_LONG;
	}

	/* Validate salt (required param) */
	if (NULL == context->salt) {
		if (0 != context->saltlen) {
			return ARGON2_SALT_PTR_MISMATCH;
		}
	}

	if (ARGON2_MIN_SALT_LENGTH > context->saltlen) {
		return ARGON2_SALT_TOO_SHORT;
	}

	if (ARGON2_MAX_SALT_LENGTH < context->saltlen) {
		return ARGON2_SALT_TOO_LONG;
	}

	/* Validate secret (optional param) */
	if (NULL == context->secret) {
		if (0 != context->secretlen) {
			return ARGON2_SECRET_PTR_MISMATCH;
		}
	}
	else {
		if (ARGON2_MIN_SECRET > context->secretlen) {
			return ARGON2_SECRET_TOO_SHORT;
		}
		if (ARGON2_MAX_SECRET < context->secretlen) {
			return ARGON2_SECRET_TOO_LONG;
		}
	}

	/* Validate associated data (optional param) */
	if (NULL == context->ad) {
		if (0 != context->adlen) {
			return ARGON2_AD_PTR_MISMATCH;
		}
	}
	else {
		if (ARGON2_MIN_AD_LENGTH > context->adlen) {
			return ARGON2_AD_TOO_SHORT;
		}
		if (ARGON2_MAX_AD_LENGTH < context->adlen) {
			return ARGON2_AD_TOO_LONG;
		}
	}

	/* Validate memory cost */
	if (ARGON2_MIN_MEMORY > context->m_cost) {
		return ARGON2_MEMORY_TOO_LITTLE;
	}

	if (ARGON2_MAX_MEMORY < context->m_cost) {
		return ARGON2_MEMORY_TOO_MUCH;
	}

	if (context->m_cost < 8 * context->lanes) {
		return ARGON2_MEMORY_TOO_LITTLE;
	}

	/* Validate time cost */
	if (ARGON2_MIN_TIME > context->t_cost) {
		return ARGON2_TIME_TOO_SMALL;
	}

	if (ARGON2_MAX_TIME < context->t_cost) {
		return ARGON2_TIME_TOO_LARGE;
	}

	/* Validate lanes */
	if (ARGON2_MIN_LANES > context->lanes) {
		return ARGON2_LANES_TOO_FEW;
	}

	if (ARGON2_MAX_LANES < context->lanes) {
		return ARGON2_LANES_TOO_MANY;
	}

	/* Validate threads */
	if (ARGON2_MIN_THREADS > context->threads) {
		return ARGON2_THREADS_TOO_FEW;
	}

	if (ARGON2_MAX_THREADS < context->threads) {
		return ARGON2_THREADS_TOO_MANY;
	}

	if (NULL != context->allocate_cbk && NULL == context->free_cbk) {
		return ARGON2_FREE_MEMORY_CBK_NULL;
	}

	if (NULL == context->allocate_cbk && NULL != context->free_cbk) {
		return ARGON2_ALLOCATE_MEMORY_CBK_NULL;
	}

	return ARGON2_OK;
}

void rxa2_fill_first_blocks(uint8_t *blockhash, const argon2_instance_t *instance) {
	uint32_t l;
	/* Make the first and second block in each lane as G(H0||0||i) or
	   G(H0||1||i) */
	uint8_t blockhash_bytes[ARGON2_BLOCK_SIZE];
	for (l = 0; l < instance->lanes; ++l) {

		store32(blockhash + ARGON2_PREHASH_DIGEST_LENGTH, 0);
		store32(blockhash + ARGON2_PREHASH_DIGEST_LENGTH + 4, l);
		rxa2_blake2b_long(blockhash_bytes, ARGON2_BLOCK_SIZE, blockhash,
			ARGON2_PREHASH_SEED_LENGTH);
		load_block(&instance->memory[l * instance->lane_length + 0],
			blockhash_bytes);

		store32(blockhash + ARGON2_PREHASH_DIGEST_LENGTH, 1);
		rxa2_blake2b_long(blockhash_bytes, ARGON2_BLOCK_SIZE, blockhash,
			ARGON2_PREHASH_SEED_LENGTH);
		load_block(&instance->memory[l * instance->lane_length + 1],
			blockhash_bytes);
	}
	rxa2_clear_internal_memory(blockhash_bytes, ARGON2_BLOCK_SIZE);
}

void rxa2_initial_hash(uint8_t *blockhash, argon2_context *context, argon2_type type) {
	blake2b_state BlakeHash;
	uint8_t value[sizeof(uint32_t)];

	if (NULL == context || NULL == blockhash) {
		return;
	}

	rx_blake2b_init(&BlakeHash, ARGON2_PREHASH_DIGEST_LENGTH);

	store32(&value, context->lanes);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->outlen);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->m_cost);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->t_cost);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->version);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, (uint32_t)type);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->pwdlen);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->pwd != NULL) {
		rx_blake2b_update(&BlakeHash, (const uint8_t *)context->pwd,
			context->pwdlen);

		if (context->flags & ARGON2_FLAG_CLEAR_PASSWORD) {
			rxa2_secure_wipe_memory(context->pwd, context->pwdlen);
			context->pwdlen = 0;
		}
	}

	store32(&value, context->saltlen);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->salt != NULL) {
		rx_blake2b_update(&BlakeHash, (const uint8_t *)context->salt, context->saltlen);
	}

	store32(&value, context->secretlen);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->secret != NULL) {
		rx_blake2b_update(&BlakeHash, (const uint8_t *)context->secret,
			context->secretlen);

		if (context->flags & ARGON2_FLAG_CLEAR_SECRET) {
			rxa2_secure_wipe_memory(context->secret, context->secretlen);
			context->secretlen = 0;
		}
	}

	store32(&value, context->adlen);
	rx_blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->ad != NULL) {
		rx_blake2b_update(&BlakeHash, (const uint8_t *)context->ad,
			context->adlen);
	}

	rx_blake2b_final(&BlakeHash, blockhash, ARGON2_PREHASH_DIGEST_LENGTH);
}

int rxa2_argon_initialize(argon2_instance_t *instance, argon2_context *context) {
	uint8_t blockhash[ARGON2_PREHASH_SEED_LENGTH];

	if (instance == NULL || context == NULL)
		return ARGON2_INCORRECT_PARAMETER;
	instance->context_ptr = context;

	/* 1. Memory allocation */
	/*result = allocate_memory(context, (uint8_t **)&(instance->memory), instance->memory_blocks, sizeof(block));
	if (result != ARGON2_OK) {
		return result;
	}*/

	/* 2. Initial hashing */
	/* H_0 + 8 extra bytes to produce the first blocks */
	/* uint8_t blockhash[ARGON2_PREHASH_SEED_LENGTH]; */
	/* Hashing all inputs */
	rxa2_initial_hash(blockhash, context, instance->type);
	/* Zeroing 8 extra bytes */
	rxa2_clear_internal_memory(blockhash + ARGON2_PREHASH_DIGEST_LENGTH,
		ARGON2_PREHASH_SEED_LENGTH -
		ARGON2_PREHASH_DIGEST_LENGTH);

	/* 3. Creating first blocks, we always have at least two blocks in a slice
	 */
	rxa2_fill_first_blocks(blockhash, instance);
	/* Clearing the hash */
	rxa2_clear_internal_memory(blockhash, ARGON2_PREHASH_SEED_LENGTH);

	return ARGON2_OK;
}
