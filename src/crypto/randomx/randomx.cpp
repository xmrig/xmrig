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

#include "crypto/randomx/common.hpp"
#include "crypto/randomx/randomx.h"
#include "crypto/randomx/dataset.hpp"
#include "crypto/randomx/vm_interpreted.hpp"
#include "crypto/randomx/vm_interpreted_light.hpp"
#include "crypto/randomx/vm_compiled.hpp"
#include "crypto/randomx/vm_compiled_light.hpp"
#include "crypto/randomx/blake2/blake2.h"

#if defined(_M_X64) || defined(__x86_64__)
#include "crypto/randomx/jit_compiler_x86_static.hpp"
#elif defined(XMRIG_ARM)
#include "crypto/randomx/jit_compiler_a64_static.hpp"
#endif

#include <cassert>

RandomX_ConfigurationWownero::RandomX_ConfigurationWownero()
{
	ArgonSalt = "RandomWOW\x01";
	ProgramIterations = 1024;
	ProgramCount = 16;
	ScratchpadL2_Size = 131072;
	ScratchpadL3_Size = 1048576;

	RANDOMX_FREQ_IADD_RS = 25;
	RANDOMX_FREQ_IROR_R = 10;
	RANDOMX_FREQ_IROL_R = 0;
	RANDOMX_FREQ_FSWAP_R = 8;
	RANDOMX_FREQ_FADD_R = 20;
	RANDOMX_FREQ_FSUB_R = 20;
	RANDOMX_FREQ_FMUL_R = 20;
	RANDOMX_FREQ_CBRANCH = 16;

	fillAes4Rx4_Key[0] = rx_set_int_vec_i128(0xcf359e95, 0x141f82b7, 0x7ffbe4a6, 0xf890465d);
	fillAes4Rx4_Key[1] = rx_set_int_vec_i128(0x6741ffdc, 0xbd5c5ac3, 0xfee8278a, 0x6a55c450);
	fillAes4Rx4_Key[2] = rx_set_int_vec_i128(0x3d324aac, 0xa7279ad2, 0xd524fde4, 0x114c47a4);
	fillAes4Rx4_Key[3] = rx_set_int_vec_i128(0x76f6db08, 0x42d3dbd9, 0x99a9aeff, 0x810c3a2a);
	fillAes4Rx4_Key[4] = fillAes4Rx4_Key[0];
	fillAes4Rx4_Key[5] = fillAes4Rx4_Key[1];
	fillAes4Rx4_Key[6] = fillAes4Rx4_Key[2];
	fillAes4Rx4_Key[7] = fillAes4Rx4_Key[3];
}

RandomX_ConfigurationLoki::RandomX_ConfigurationLoki()
{
	ArgonIterations = 4;
	ArgonLanes = 2;
	ArgonSalt = "RandomXL\x12";
	ProgramSize = 320;
	ProgramCount = 7;

	RANDOMX_FREQ_IADD_RS = 25;
	RANDOMX_FREQ_CBRANCH = 16;
}

RandomX_ConfigurationArqma::RandomX_ConfigurationArqma()
{
	ArgonIterations = 1;
	ArgonSalt = "RandomARQ\x01";
	ProgramIterations = 1024;
	ProgramCount = 4;
	ScratchpadL2_Size = 131072;
	ScratchpadL3_Size = 262144;
}

RandomX_ConfigurationBase::RandomX_ConfigurationBase()
	: ArgonMemory(262144)
	, ArgonIterations(3)
	, ArgonLanes(1)
	, ArgonSalt("RandomX\x03")
	, CacheAccesses(8)
	, SuperscalarLatency(170)
	, DatasetBaseSize(2147483648)
	, DatasetExtraSize(33554368)
	, ScratchpadL1_Size(16384)
	, ScratchpadL2_Size(262144)
	, ScratchpadL3_Size(2097152)
	, ProgramSize(256)
	, ProgramIterations(2048)
	, ProgramCount(8)
	, JumpBits(8)
	, JumpOffset(8)
	, RANDOMX_FREQ_IADD_RS(16)
	, RANDOMX_FREQ_IADD_M(7)
	, RANDOMX_FREQ_ISUB_R(16)
	, RANDOMX_FREQ_ISUB_M(7)
	, RANDOMX_FREQ_IMUL_R(16)
	, RANDOMX_FREQ_IMUL_M(4)
	, RANDOMX_FREQ_IMULH_R(4)
	, RANDOMX_FREQ_IMULH_M(1)
	, RANDOMX_FREQ_ISMULH_R(4)
	, RANDOMX_FREQ_ISMULH_M(1)
	, RANDOMX_FREQ_IMUL_RCP(8)
	, RANDOMX_FREQ_INEG_R(2)
	, RANDOMX_FREQ_IXOR_R(15)
	, RANDOMX_FREQ_IXOR_M(5)
	, RANDOMX_FREQ_IROR_R(8)
	, RANDOMX_FREQ_IROL_R(2)
	, RANDOMX_FREQ_ISWAP_R(4)
	, RANDOMX_FREQ_FSWAP_R(4)
	, RANDOMX_FREQ_FADD_R(16)
	, RANDOMX_FREQ_FADD_M(5)
	, RANDOMX_FREQ_FSUB_R(16)
	, RANDOMX_FREQ_FSUB_M(5)
	, RANDOMX_FREQ_FSCAL_R(6)
	, RANDOMX_FREQ_FMUL_R(32)
	, RANDOMX_FREQ_FDIV_M(4)
	, RANDOMX_FREQ_FSQRT_R(6)
	, RANDOMX_FREQ_CBRANCH(25)
	, RANDOMX_FREQ_CFROUND(1)
	, RANDOMX_FREQ_ISTORE(16)
	, RANDOMX_FREQ_NOP(0)
{
	fillAes4Rx4_Key[0] = rx_set_int_vec_i128(0x99e5d23f, 0x2f546d2b, 0xd1833ddb, 0x6421aadd);
	fillAes4Rx4_Key[1] = rx_set_int_vec_i128(0xa5dfcde5, 0x06f79d53, 0xb6913f55, 0xb20e3450);
	fillAes4Rx4_Key[2] = rx_set_int_vec_i128(0x171c02bf, 0x0aa4679f, 0x515e7baf, 0x5c3ed904);
	fillAes4Rx4_Key[3] = rx_set_int_vec_i128(0xd8ded291, 0xcd673785, 0xe78f5d08, 0x85623763);
	fillAes4Rx4_Key[4] = rx_set_int_vec_i128(0x229effb4, 0x3d518b6d, 0xe3d6a7a6, 0xb5826f73);
	fillAes4Rx4_Key[5] = rx_set_int_vec_i128(0xb272b7d2, 0xe9024d4e, 0x9c10b3d9, 0xc7566bf3);
	fillAes4Rx4_Key[6] = rx_set_int_vec_i128(0xf63befa7, 0x2ba9660a, 0xf765a38b, 0xf273c9e7);
	fillAes4Rx4_Key[7] = rx_set_int_vec_i128(0xc0b0762d, 0x0c06d1fd, 0x915839de, 0x7a7cd609);

#if defined(_M_X64) || defined(__x86_64__)
	{
		const uint8_t* a = (const uint8_t*)&randomx_sshash_prefetch;
		const uint8_t* b = (const uint8_t*)&randomx_sshash_end;
		memcpy(codeShhPrefetchTweaked, a, b - a);
	}
	{
		const uint8_t* a = (const uint8_t*)&randomx_program_read_dataset;
		const uint8_t* b = (const uint8_t*)&randomx_program_read_dataset_sshash_init;
		memcpy(codeReadDatasetTweaked, a, b - a);
	}
	{
		const uint8_t* a = (const uint8_t*)&randomx_program_read_dataset_sshash_init;
		const uint8_t* b = (const uint8_t*)&randomx_program_read_dataset_sshash_fin;
		memcpy(codeReadDatasetLightSshInitTweaked, a, b - a);
	}
	{
		const uint8_t* a = (const uint8_t*)&randomx_prefetch_scratchpad;
		const uint8_t* b = (const uint8_t*)&randomx_prefetch_scratchpad_end;
		memcpy(codePrefetchScratchpadTweaked, a, b - a);
	}
#endif
}

static uint32_t Log2(size_t value) { return (value > 1) ? (Log2(value / 2) + 1) : 0; }

void RandomX_ConfigurationBase::Apply()
{
	ScratchpadL1Mask_Calculated = (ScratchpadL1_Size / sizeof(uint64_t) - 1) * 8;
	ScratchpadL1Mask16_Calculated = (ScratchpadL1_Size / sizeof(uint64_t) / 2 - 1) * 16;
	ScratchpadL2Mask_Calculated = (ScratchpadL2_Size / sizeof(uint64_t) - 1) * 8;
	ScratchpadL2Mask16_Calculated = (ScratchpadL2_Size / sizeof(uint64_t) / 2 - 1) * 16;
	ScratchpadL3Mask_Calculated = (((ScratchpadL3_Size / sizeof(uint64_t)) - 1) * 8);
	ScratchpadL3Mask64_Calculated = ((ScratchpadL3_Size / sizeof(uint64_t)) / 8 - 1) * 64;

	CacheLineAlignMask_Calculated = (DatasetBaseSize - 1) & ~(RANDOMX_DATASET_ITEM_SIZE - 1);
	DatasetExtraItems_Calculated = DatasetExtraSize / RANDOMX_DATASET_ITEM_SIZE;

	ConditionMask_Calculated = (1 << JumpBits) - 1;

#if defined(_M_X64) || defined(__x86_64__)
	*(uint32_t*)(codeShhPrefetchTweaked + 3) = ArgonMemory * 16 - 1;
	const uint32_t DatasetBaseMask = DatasetBaseSize - RANDOMX_DATASET_ITEM_SIZE;
	*(uint32_t*)(codeReadDatasetTweaked + 7) = DatasetBaseMask;
	*(uint32_t*)(codeReadDatasetTweaked + 23) = DatasetBaseMask;
	*(uint32_t*)(codeReadDatasetLightSshInitTweaked + 59) = DatasetBaseMask;

	*(uint32_t*)(codePrefetchScratchpadTweaked + 4) = ScratchpadL3Mask64_Calculated;
	*(uint32_t*)(codePrefetchScratchpadTweaked + 18) = ScratchpadL3Mask64_Calculated;

#define JIT_HANDLE(x, prev) randomx::JitCompilerX86::engine[k] = &randomx::JitCompilerX86::h_##x

#elif defined(XMRIG_ARM)

	Log2_ScratchpadL1 = Log2(ScratchpadL1_Size);
	Log2_ScratchpadL2 = Log2(ScratchpadL2_Size);
	Log2_ScratchpadL3 = Log2(ScratchpadL3_Size);
	Log2_DatasetBaseSize = Log2(DatasetBaseSize);
	Log2_CacheSize = Log2((ArgonMemory * randomx::ArgonBlockSize) / randomx::CacheLineSize);

#define JIT_HANDLE(x, prev) randomx::JitCompilerA64::engine[k] = &randomx::JitCompilerA64::h_##x

#else
#define JIT_HANDLE(x, prev)
#endif

	constexpr int CEIL_NULL = 0;
	int k = 0;

#define INST_HANDLE(x, prev) \
	CEIL_##x = CEIL_##prev + RANDOMX_FREQ_##x; \
	for (; k < CEIL_##x; ++k) { JIT_HANDLE(x, prev); }

	INST_HANDLE(IADD_RS, NULL);
	INST_HANDLE(IADD_M, IADD_RS);
	INST_HANDLE(ISUB_R, IADD_M);
	INST_HANDLE(ISUB_M, ISUB_R);
	INST_HANDLE(IMUL_R, ISUB_M);
	INST_HANDLE(IMUL_M, IMUL_R);
	INST_HANDLE(IMULH_R, IMUL_M);
	INST_HANDLE(IMULH_M, IMULH_R);
	INST_HANDLE(ISMULH_R, IMULH_M);
	INST_HANDLE(ISMULH_M, ISMULH_R);
	INST_HANDLE(IMUL_RCP, ISMULH_M);
	INST_HANDLE(INEG_R, IMUL_RCP);
	INST_HANDLE(IXOR_R, INEG_R);
	INST_HANDLE(IXOR_M, IXOR_R);
	INST_HANDLE(IROR_R, IXOR_M);
	INST_HANDLE(IROL_R, IROR_R);
	INST_HANDLE(ISWAP_R, IROL_R);
	INST_HANDLE(FSWAP_R, ISWAP_R);
	INST_HANDLE(FADD_R, FSWAP_R);
	INST_HANDLE(FADD_M, FADD_R);
	INST_HANDLE(FSUB_R, FADD_M);
	INST_HANDLE(FSUB_M, FSUB_R);
	INST_HANDLE(FSCAL_R, FSUB_M);
	INST_HANDLE(FMUL_R, FSCAL_R);
	INST_HANDLE(FDIV_M, FMUL_R);
	INST_HANDLE(FSQRT_R, FDIV_M);
	INST_HANDLE(CBRANCH, FSQRT_R);
	INST_HANDLE(CFROUND, CBRANCH);
	INST_HANDLE(ISTORE, CFROUND);
	INST_HANDLE(NOP, ISTORE);
#undef INST_HANDLE
}

RandomX_ConfigurationMonero RandomX_MoneroConfig;
RandomX_ConfigurationWownero RandomX_WowneroConfig;
RandomX_ConfigurationLoki RandomX_LokiConfig;
RandomX_ConfigurationArqma RandomX_ArqmaConfig;

RandomX_ConfigurationBase RandomX_CurrentConfig;

extern "C" {

	randomx_cache *randomx_alloc_cache(randomx_flags flags) {
		randomx_cache *cache = nullptr;

		try {
			cache = new randomx_cache();
			switch (flags & (RANDOMX_FLAG_JIT | RANDOMX_FLAG_LARGE_PAGES)) {
				case RANDOMX_FLAG_DEFAULT:
					cache->dealloc = &randomx::deallocCache<randomx::DefaultAllocator>;
					cache->jit = nullptr;
					cache->initialize = &randomx::initCache;
					cache->datasetInit = &randomx::initDataset;
					cache->memory = (uint8_t*)randomx::DefaultAllocator::allocMemory(RANDOMX_CACHE_MAX_SIZE);
					break;

				case RANDOMX_FLAG_JIT:
					cache->dealloc = &randomx::deallocCache<randomx::DefaultAllocator>;
					cache->jit = new randomx::JitCompiler();
					cache->initialize = &randomx::initCacheCompile;
					cache->datasetInit = cache->jit->getDatasetInitFunc();
					cache->memory = (uint8_t*)randomx::DefaultAllocator::allocMemory(RANDOMX_CACHE_MAX_SIZE);
					break;

				case RANDOMX_FLAG_LARGE_PAGES:
					cache->dealloc = &randomx::deallocCache<randomx::LargePageAllocator>;
					cache->jit = nullptr;
					cache->initialize = &randomx::initCache;
					cache->datasetInit = &randomx::initDataset;
					cache->memory = (uint8_t*)randomx::LargePageAllocator::allocMemory(RANDOMX_CACHE_MAX_SIZE);
					break;

				case RANDOMX_FLAG_JIT | RANDOMX_FLAG_LARGE_PAGES:
					cache->dealloc = &randomx::deallocCache<randomx::LargePageAllocator>;
					cache->jit = new randomx::JitCompiler();
					cache->initialize = &randomx::initCacheCompile;
					cache->datasetInit = cache->jit->getDatasetInitFunc();
					cache->memory = (uint8_t*)randomx::LargePageAllocator::allocMemory(RANDOMX_CACHE_MAX_SIZE);
					break;

				default:
					UNREACHABLE;
			}
		}
		catch (std::exception &ex) {
			if (cache != nullptr) {
				randomx_release_cache(cache);
				cache = nullptr;
			}
		}

		return cache;
	}

	void randomx_init_cache(randomx_cache *cache, const void *key, size_t keySize) {
		assert(cache != nullptr);
		assert(keySize == 0 || key != nullptr);
		cache->initialize(cache, key, keySize);
	}

	void randomx_release_cache(randomx_cache* cache) {
		assert(cache != nullptr);
		cache->dealloc(cache);
		delete cache;
	}

	randomx_dataset *randomx_alloc_dataset(randomx_flags flags) {
		randomx_dataset *dataset = nullptr;

		try {
			dataset = new randomx_dataset();
			if (flags & RANDOMX_FLAG_LARGE_PAGES) {
				dataset->dealloc = &randomx::deallocDataset<randomx::LargePageAllocator>;
				dataset->memory = (uint8_t*)randomx::LargePageAllocator::allocMemory(RANDOMX_DATASET_MAX_SIZE);
			}
			else {
				dataset->dealloc = &randomx::deallocDataset<randomx::DefaultAllocator>;
				dataset->memory = (uint8_t*)randomx::DefaultAllocator::allocMemory(RANDOMX_DATASET_MAX_SIZE);
			}
		}
		catch (std::exception &ex) {
			if (dataset != nullptr) {
				randomx_release_dataset(dataset);
				dataset = nullptr;
			}
		}

		return dataset;
	}

	#define DatasetItemCount ((RandomX_CurrentConfig.DatasetBaseSize + RandomX_CurrentConfig.DatasetExtraSize) / RANDOMX_DATASET_ITEM_SIZE)

	unsigned long randomx_dataset_item_count() {
		return DatasetItemCount;
	}

	void randomx_init_dataset(randomx_dataset *dataset, randomx_cache *cache, unsigned long startItem, unsigned long itemCount) {
		assert(dataset != nullptr);
		assert(cache != nullptr);
		assert(startItem < DatasetItemCount && itemCount <= DatasetItemCount);
		assert(startItem + itemCount <= DatasetItemCount);
		cache->datasetInit(cache, dataset->memory + startItem * randomx::CacheLineSize, startItem, startItem + itemCount);
	}

	void *randomx_get_dataset_memory(randomx_dataset *dataset) {
		assert(dataset != nullptr);
		return dataset->memory;
	}

	void randomx_release_dataset(randomx_dataset *dataset) {
		assert(dataset != nullptr);
		dataset->dealloc(dataset);
		delete dataset;
	}

	randomx_vm *randomx_create_vm(randomx_flags flags, randomx_cache *cache, randomx_dataset *dataset, uint8_t *scratchpad) {
		assert(cache != nullptr || (flags & RANDOMX_FLAG_FULL_MEM));
		assert(cache == nullptr || cache->isInitialized());
		assert(dataset != nullptr || !(flags & RANDOMX_FLAG_FULL_MEM));

		randomx_vm *vm = nullptr;

		try {
			switch (flags & (RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES)) {
				case RANDOMX_FLAG_DEFAULT:
					vm = new randomx::InterpretedLightVmDefault();
					break;

				case RANDOMX_FLAG_FULL_MEM:
					vm = new randomx::InterpretedVmDefault();
					break;

				case RANDOMX_FLAG_JIT:
					vm = new randomx::CompiledLightVmDefault();
					break;

				case RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT:
					vm = new randomx::CompiledVmDefault();
					break;

				case RANDOMX_FLAG_HARD_AES:
					vm = new randomx::InterpretedLightVmHardAes();
					break;

				case RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_HARD_AES:
					vm = new randomx::InterpretedVmHardAes();
					break;

				case RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES:
					vm = new randomx::CompiledLightVmHardAes();
					break;

				case RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES:
					vm = new randomx::CompiledVmHardAes();
					break;

				default:
					UNREACHABLE;
			}

			if (cache != nullptr) {
				vm->setCache(cache);
			}

			if (dataset != nullptr) {
				vm->setDataset(dataset);
			}

			vm->setScratchpad(scratchpad);
		}
		catch (std::exception &ex) {
			delete vm;
			vm = nullptr;
		}

		return vm;
	}

	void randomx_vm_set_cache(randomx_vm *machine, randomx_cache* cache) {
		assert(machine != nullptr);
		assert(cache != nullptr && cache->isInitialized());
		machine->setCache(cache);
	}

	void randomx_vm_set_dataset(randomx_vm *machine, randomx_dataset *dataset) {
		assert(machine != nullptr);
		assert(dataset != nullptr);
		machine->setDataset(dataset);
	}

	void randomx_destroy_vm(randomx_vm *machine) {
		assert(machine != nullptr);
		delete machine;
	}

	void randomx_calculate_hash(randomx_vm *machine, const void *input, size_t inputSize, void *output) {
		assert(machine != nullptr);
		assert(inputSize == 0 || input != nullptr);
		assert(output != nullptr);
		alignas(16) uint64_t tempHash[8];
		rx_blake2b(tempHash, sizeof(tempHash), input, inputSize, nullptr, 0);
		machine->initScratchpad(&tempHash);
		machine->resetRoundingMode();
		for (uint32_t chain = 0; chain < RandomX_CurrentConfig.ProgramCount - 1; ++chain) {
			machine->run(&tempHash);
			rx_blake2b(tempHash, sizeof(tempHash), machine->getRegisterFile(), sizeof(randomx::RegisterFile), nullptr, 0);
		}
		machine->run(&tempHash);
		machine->getFinalResult(output, RANDOMX_HASH_SIZE);
	}

}
