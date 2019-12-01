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

#ifndef DEFYX_H
#define DEFYX_H

#include "crypto/randomx/randomx.h"

struct RandomX_ConfigurationScala : public RandomX_ConfigurationBase { RandomX_ConfigurationScala(); };

extern RandomX_ConfigurationScala RandomX_ScalaConfig;

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Calculates a RandomX hash value.
 *
 * @param machine is a pointer to a randomx_vm structure. Must not be NULL.
 * @param input is a pointer to memory to be hashed. Must not be NULL.
 * @param inputSize is the number of bytes to be hashed.
 * @param output is a pointer to memory where the hash will be stored. Must not
 *        be NULL and at least RANDOMX_HASH_SIZE bytes must be available for writing.
*/
RANDOMX_EXPORT void defyx_calculate_hash(randomx_vm *machine, const void *input, size_t inputSize, void *output);

RANDOMX_EXPORT void defyx_calculate_hash_first(randomx_vm* machine, uint64_t (&tempHash)[8], const void* input, size_t inputSize);
RANDOMX_EXPORT void defyx_calculate_hash_next(randomx_vm* machine, uint64_t (&tempHash)[8], const void* nextInput, size_t nextInputSize, void* output);

#if defined(__cplusplus)
}
#endif

#endif
