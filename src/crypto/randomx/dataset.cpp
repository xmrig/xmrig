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

#include <new>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <cstring>

#include "crypto/randomx/common.hpp"
#include "crypto/randomx/dataset.hpp"
#include "crypto/randomx/virtual_memory.hpp"
#include "crypto/randomx/superscalar.hpp"
#include "crypto/randomx/blake2_generator.hpp"
#include "crypto/randomx/reciprocal.h"
#include "crypto/randomx/blake2/endian.h"
#include "crypto/randomx/jit_compiler.hpp"
#include "crypto/randomx/intrin_portable.h"

#include "3rdparty/argon2/include/argon2.h"
#include "3rdparty/argon2/lib/core.h"

//static_assert(RANDOMX_ARGON_MEMORY % (RANDOMX_ARGON_LANES * ARGON2_SYNC_POINTS) == 0, "RANDOMX_ARGON_MEMORY - invalid value");
static_assert(ARGON2_BLOCK_SIZE == randomx::ArgonBlockSize, "Unexpected value of ARGON2_BLOCK_SIZE");

namespace randomx {

	template<class Allocator>
	void deallocCache(randomx_cache* cache) {
		if (cache->memory != nullptr)
			Allocator::freeMemory(cache->memory, RANDOMX_CACHE_MAX_SIZE);
		if (cache->jit != nullptr)
			delete cache->jit;
	}

	template void deallocCache<DefaultAllocator>(randomx_cache* cache);
	template void deallocCache<LargePageAllocator>(randomx_cache* cache);

	void initCache(randomx_cache* cache, const void* key, size_t keySize) {
		argon2_context context;

		context.out = nullptr;
		context.outlen = 0;
		context.pwd = CONST_CAST(uint8_t *)key;
		context.pwdlen = (uint32_t)keySize;
		context.salt = CONST_CAST(uint8_t *)RandomX_CurrentConfig.ArgonSalt;
		context.saltlen = (uint32_t)strlen(RandomX_CurrentConfig.ArgonSalt);
		context.secret = NULL;
		context.secretlen = 0;
		context.ad = NULL;
		context.adlen = 0;
		context.t_cost = RandomX_CurrentConfig.ArgonIterations;
		context.m_cost = RandomX_CurrentConfig.ArgonMemory;
		context.lanes = RandomX_CurrentConfig.ArgonLanes;
		context.threads = 1;
		context.allocate_cbk = NULL;
		context.free_cbk = NULL;
		context.flags = ARGON2_DEFAULT_FLAGS;
		context.version = ARGON2_VERSION_NUMBER;

		argon2_ctx_mem(&context, Argon2_d, cache->memory, RandomX_CurrentConfig.ArgonMemory * 1024);

		randomx::Blake2Generator gen(key, keySize);
		for (uint32_t i = 0; i < RandomX_CurrentConfig.CacheAccesses; ++i) {
			randomx::generateSuperscalar(cache->programs[i], gen);
		}
	}

	void initCacheCompile(randomx_cache* cache, const void* key, size_t keySize) {
		initCache(cache, key, keySize);
		cache->jit->generateSuperscalarHash(cache->programs);
		cache->jit->generateDatasetInitCode();
	}

	constexpr uint64_t superscalarMul0 = 6364136223846793005ULL;
	constexpr uint64_t superscalarAdd1 = 9298411001130361340ULL;
	constexpr uint64_t superscalarAdd2 = 12065312585734608966ULL;
	constexpr uint64_t superscalarAdd3 = 9306329213124626780ULL;
	constexpr uint64_t superscalarAdd4 = 5281919268842080866ULL;
	constexpr uint64_t superscalarAdd5 = 10536153434571861004ULL;
	constexpr uint64_t superscalarAdd6 = 3398623926847679864ULL;
	constexpr uint64_t superscalarAdd7 = 9549104520008361294ULL;

	static inline uint8_t* getMixBlock(uint64_t registerValue, uint8_t *memory) {
		const uint32_t mask = (RandomX_CurrentConfig.ArgonMemory * randomx::ArgonBlockSize) / CacheLineSize - 1;
		return memory + (registerValue & mask) * CacheLineSize;
	}

	void initDatasetItem(randomx_cache* cache, uint8_t* out, uint64_t itemNumber) {
		int_reg_t rl[8];
		uint8_t* mixBlock;
		uint64_t registerValue = itemNumber;
		rl[0] = (itemNumber + 1) * superscalarMul0;
		rl[1] = rl[0] ^ superscalarAdd1;
		rl[2] = rl[0] ^ superscalarAdd2;
		rl[3] = rl[0] ^ superscalarAdd3;
		rl[4] = rl[0] ^ superscalarAdd4;
		rl[5] = rl[0] ^ superscalarAdd5;
		rl[6] = rl[0] ^ superscalarAdd6;
		rl[7] = rl[0] ^ superscalarAdd7;
		for (unsigned i = 0; i < RandomX_CurrentConfig.CacheAccesses; ++i) {
			mixBlock = getMixBlock(registerValue, cache->memory);
			rx_prefetch_nta(mixBlock);
			SuperscalarProgram& prog = cache->programs[i];

			executeSuperscalar(rl, prog);

			for (unsigned q = 0; q < 8; ++q)
				rl[q] ^= load64_native(mixBlock + 8 * q);

			registerValue = rl[prog.getAddressRegister()];
		}

		memcpy(out, &rl, CacheLineSize);
	}

	void initDataset(randomx_cache* cache, uint8_t* dataset, uint32_t startItem, uint32_t endItem) {
		for (uint32_t itemNumber = startItem; itemNumber < endItem; ++itemNumber, dataset += CacheLineSize)
			initDatasetItem(cache, dataset, itemNumber);
	}
}
