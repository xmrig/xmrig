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

#ifndef RANDOMX_H
#define RANDOMX_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "crypto/randomx/intrin_portable.h"

#define RANDOMX_HASH_SIZE 32
#define RANDOMX_DATASET_ITEM_SIZE 64

#ifndef RANDOMX_EXPORT
#define RANDOMX_EXPORT
#endif


enum randomx_flags {
  RANDOMX_FLAG_DEFAULT = 0,
  RANDOMX_FLAG_LARGE_PAGES = 1,
  RANDOMX_FLAG_HARD_AES = 2,
  RANDOMX_FLAG_FULL_MEM = 4,
  RANDOMX_FLAG_JIT = 8,
  RANDOMX_FLAG_1GB_PAGES = 16,
  RANDOMX_FLAG_RYZEN = 64,
};


struct randomx_dataset;
struct randomx_cache;
class randomx_vm;


struct RandomX_ConfigurationBase
{
	RandomX_ConfigurationBase();

	void Apply();

	uint32_t ArgonMemory;
	uint32_t ArgonIterations;
	uint32_t ArgonLanes;
	const char* ArgonSalt;
	uint32_t CacheAccesses;
	uint32_t SuperscalarLatency;

	uint32_t DatasetBaseSize;
	uint32_t DatasetExtraSize;

	uint32_t ScratchpadL1_Size;
	uint32_t ScratchpadL2_Size;
	uint32_t ScratchpadL3_Size;

	uint32_t ProgramSize;
	uint32_t ProgramIterations;
	uint32_t ProgramCount;

	uint32_t JumpBits;
	uint32_t JumpOffset;

	uint32_t RANDOMX_FREQ_IADD_RS;
	uint32_t RANDOMX_FREQ_IADD_M;
	uint32_t RANDOMX_FREQ_ISUB_R;
	uint32_t RANDOMX_FREQ_ISUB_M;
	uint32_t RANDOMX_FREQ_IMUL_R;
	uint32_t RANDOMX_FREQ_IMUL_M;
	uint32_t RANDOMX_FREQ_IMULH_R;
	uint32_t RANDOMX_FREQ_IMULH_M;
	uint32_t RANDOMX_FREQ_ISMULH_R;
	uint32_t RANDOMX_FREQ_ISMULH_M;
	uint32_t RANDOMX_FREQ_IMUL_RCP;
	uint32_t RANDOMX_FREQ_INEG_R;
	uint32_t RANDOMX_FREQ_IXOR_R;
	uint32_t RANDOMX_FREQ_IXOR_M;
	uint32_t RANDOMX_FREQ_IROR_R;
	uint32_t RANDOMX_FREQ_IROL_R;
	uint32_t RANDOMX_FREQ_ISWAP_R;
	uint32_t RANDOMX_FREQ_FSWAP_R;
	uint32_t RANDOMX_FREQ_FADD_R;
	uint32_t RANDOMX_FREQ_FADD_M;
	uint32_t RANDOMX_FREQ_FSUB_R;
	uint32_t RANDOMX_FREQ_FSUB_M;
	uint32_t RANDOMX_FREQ_FSCAL_R;
	uint32_t RANDOMX_FREQ_FMUL_R;
	uint32_t RANDOMX_FREQ_FDIV_M;
	uint32_t RANDOMX_FREQ_FSQRT_R;
	uint32_t RANDOMX_FREQ_CBRANCH;
	uint32_t RANDOMX_FREQ_CFROUND;
	uint32_t RANDOMX_FREQ_ISTORE;
	uint32_t RANDOMX_FREQ_NOP;

	rx_vec_i128 fillAes4Rx4_Key[8];

	uint8_t codeShhPrefetchTweaked[20];
	uint8_t codeReadDatasetTweaked[256];
	uint32_t codeReadDatasetTweakedSize;
	uint8_t codeReadDatasetRyzenTweaked[256];
	uint32_t codeReadDatasetRyzenTweakedSize;
	uint8_t codeReadDatasetLightSshInitTweaked[68];
	uint8_t codePrefetchScratchpadTweaked[32];

	uint32_t CacheLineAlignMask_Calculated;
	uint32_t DatasetExtraItems_Calculated;

	uint32_t ScratchpadL1Mask_Calculated;
	uint32_t ScratchpadL1Mask16_Calculated;
	uint32_t ScratchpadL2Mask_Calculated;
	uint32_t ScratchpadL2Mask16_Calculated;
	uint32_t ScratchpadL3Mask_Calculated;
	uint32_t ScratchpadL3Mask64_Calculated;

	uint32_t ConditionMask_Calculated;

#if defined(XMRIG_ARMv8)
	uint32_t Log2_ScratchpadL1;
	uint32_t Log2_ScratchpadL2;
	uint32_t Log2_ScratchpadL3;
	uint32_t Log2_DatasetBaseSize;
	uint32_t Log2_CacheSize;
#endif

	int CEIL_IADD_RS;
	int CEIL_IADD_M;
	int CEIL_ISUB_R;
	int CEIL_ISUB_M;
	int CEIL_IMUL_R;
	int CEIL_IMUL_M;
	int CEIL_IMULH_R;
	int CEIL_IMULH_M;
	int CEIL_ISMULH_R;
	int CEIL_ISMULH_M;
	int CEIL_IMUL_RCP;
	int CEIL_INEG_R;
	int CEIL_IXOR_R;
	int CEIL_IXOR_M;
	int CEIL_IROR_R;
	int CEIL_IROL_R;
	int CEIL_ISWAP_R;
	int CEIL_FSWAP_R;
	int CEIL_FADD_R;
	int CEIL_FADD_M;
	int CEIL_FSUB_R;
	int CEIL_FSUB_M;
	int CEIL_FSCAL_R;
	int CEIL_FMUL_R;
	int CEIL_FDIV_M;
	int CEIL_FSQRT_R;
	int CEIL_CBRANCH;
	int CEIL_CFROUND;
	int CEIL_ISTORE;
	int CEIL_NOP;
};

struct RandomX_ConfigurationMonero : public RandomX_ConfigurationBase {};
struct RandomX_ConfigurationWownero : public RandomX_ConfigurationBase { RandomX_ConfigurationWownero(); };
struct RandomX_ConfigurationLoki : public RandomX_ConfigurationBase { RandomX_ConfigurationLoki(); };
struct RandomX_ConfigurationArqma : public RandomX_ConfigurationBase { RandomX_ConfigurationArqma(); };

extern RandomX_ConfigurationMonero RandomX_MoneroConfig;
extern RandomX_ConfigurationWownero RandomX_WowneroConfig;
extern RandomX_ConfigurationLoki RandomX_LokiConfig;
extern RandomX_ConfigurationArqma RandomX_ArqmaConfig;

extern RandomX_ConfigurationBase RandomX_CurrentConfig;

template<typename T>
void randomx_apply_config(const T& config)
{
	static_assert(sizeof(T) == sizeof(RandomX_ConfigurationBase), "Invalid RandomX configuration struct size");
	static_assert(std::is_base_of<RandomX_ConfigurationBase, T>::value, "Incompatible RandomX configuration struct");
	RandomX_CurrentConfig = config;
	RandomX_CurrentConfig.Apply();
}

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Creates a randomx_cache structure and allocates memory for RandomX Cache.
 *
 * @param flags is any combination of these 2 flags (each flag can be set or not set):
 *        RANDOMX_FLAG_LARGE_PAGES - allocate memory in large pages
 *        RANDOMX_FLAG_JIT - create cache structure with JIT compilation support; this makes
 *                           subsequent Dataset initialization faster
 *
 * @return Pointer to an allocated randomx_cache structure.
 *         NULL is returned if memory allocation fails or if the RANDOMX_FLAG_JIT
 *         is set and JIT compilation is not supported on the current platform.
 */
RANDOMX_EXPORT randomx_cache *randomx_create_cache(randomx_flags flags, uint8_t *memory);

/**
 * Initializes the cache memory and SuperscalarHash using the provided key value.
 *
 * @param cache is a pointer to a previously allocated randomx_cache structure. Must not be NULL.
 * @param key is a pointer to memory which contains the key value. Must not be NULL.
 * @param keySize is the number of bytes of the key.
*/
RANDOMX_EXPORT void randomx_init_cache(randomx_cache *cache, const void *key, size_t keySize);

/**
 * Releases all memory occupied by the randomx_cache structure.
 *
 * @param cache is a pointer to a previously allocated randomx_cache structure.
*/
RANDOMX_EXPORT void randomx_release_cache(randomx_cache* cache);

/**
 * Creates a randomx_dataset structure and allocates memory for RandomX Dataset.
 *
 * @param flags is the initialization flags. Only one flag is supported (can be set or not set):
 *        RANDOMX_FLAG_LARGE_PAGES - allocate memory in large pages
 *
 * @return Pointer to an allocated randomx_dataset structure.
 *         NULL is returned if memory allocation fails.
 */
RANDOMX_EXPORT randomx_dataset *randomx_create_dataset(uint8_t *memory);

/**
 * Gets the number of items contained in the dataset.
 *
 * @return the number of items contained in the dataset.
*/
RANDOMX_EXPORT unsigned long randomx_dataset_item_count(void);

/**
 * Initializes dataset items.
 *
 * Note: In order to use the Dataset, all items from 0 to (randomx_dataset_item_count() - 1) must be initialized.
 * This may be done by several calls to this function using non-overlapping item sequences.
 *
 * @param dataset is a pointer to a previously allocated randomx_dataset structure. Must not be NULL.
 * @param cache is a pointer to a previously allocated and initialized randomx_cache structure. Must not be NULL.
 * @param startItem is the item number where intialization should start.
 * @param itemCount is the number of items that should be initialized.
*/
RANDOMX_EXPORT void randomx_init_dataset(randomx_dataset *dataset, randomx_cache *cache, unsigned long startItem, unsigned long itemCount);

/**
 * Returns a pointer to the internal memory buffer of the dataset structure. The size
 * of the internal memory buffer is randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE.
 *
 * @param dataset is dataset is a pointer to a previously allocated randomx_dataset structure. Must not be NULL.
 *
 * @return Pointer to the internal memory buffer of the dataset structure.
*/
RANDOMX_EXPORT void *randomx_get_dataset_memory(randomx_dataset *dataset);

/**
 * Releases all memory occupied by the randomx_dataset structure.
 *
 * @param dataset is a pointer to a previously allocated randomx_dataset structure.
*/
RANDOMX_EXPORT void randomx_release_dataset(randomx_dataset *dataset);

/**
 * Creates and initializes a RandomX virtual machine.
 *
 * @param flags is any combination of these 4 flags (each flag can be set or not set):
 *        RANDOMX_FLAG_LARGE_PAGES - allocate scratchpad memory in large pages
 *        RANDOMX_FLAG_HARD_AES - virtual machine will use hardware accelerated AES
 *        RANDOMX_FLAG_FULL_MEM - virtual machine will use the full dataset
 *        RANDOMX_FLAG_JIT - virtual machine will use a JIT compiler
 *        The numeric values of the flags are ordered so that a higher value will provide
 *        faster hash calculation and a lower numeric value will provide higher portability.
 *        Using RANDOMX_FLAG_DEFAULT (all flags not set) works on all platforms, but is the slowest.
 * @param cache is a pointer to an initialized randomx_cache structure. Can be
 *        NULL if RANDOMX_FLAG_FULL_MEM is set.
 * @param dataset is a pointer to a randomx_dataset structure. Can be NULL
 *        if RANDOMX_FLAG_FULL_MEM is not set.
 *
 * @return Pointer to an initialized randomx_vm structure.
 *         Returns NULL if:
 *         (1) Scratchpad memory allocation fails.
 *         (2) The requested initialization flags are not supported on the current platform.
 *         (3) cache parameter is NULL and RANDOMX_FLAG_FULL_MEM is not set
 *         (4) dataset parameter is NULL and RANDOMX_FLAG_FULL_MEM is set
*/
RANDOMX_EXPORT randomx_vm *randomx_create_vm(randomx_flags flags, randomx_cache *cache, randomx_dataset *dataset, uint8_t *scratchpad);

/**
 * Reinitializes a virtual machine with a new Cache. This function should be called anytime
 * the Cache is reinitialized with a new key.
 *
 * @param machine is a pointer to a randomx_vm structure that was initialized
 *        without RANDOMX_FLAG_FULL_MEM. Must not be NULL.
 * @param cache is a pointer to an initialized randomx_cache structure. Must not be NULL.
*/
RANDOMX_EXPORT void randomx_vm_set_cache(randomx_vm *machine, randomx_cache* cache);

/**
 * Reinitializes a virtual machine with a new Dataset.
 *
 * @param machine is a pointer to a randomx_vm structure that was initialized
 *        with RANDOMX_FLAG_FULL_MEM. Must not be NULL.
 * @param dataset is a pointer to an initialized randomx_dataset structure. Must not be NULL.
*/
RANDOMX_EXPORT void randomx_vm_set_dataset(randomx_vm *machine, randomx_dataset *dataset);

/**
 * Releases all memory occupied by the randomx_vm structure.
 *
 * @param machine is a pointer to a previously created randomx_vm structure.
*/
RANDOMX_EXPORT void randomx_destroy_vm(randomx_vm *machine);

/**
 * Calculates a RandomX hash value.
 *
 * @param machine is a pointer to a randomx_vm structure. Must not be NULL.
 * @param input is a pointer to memory to be hashed. Must not be NULL.
 * @param inputSize is the number of bytes to be hashed.
 * @param output is a pointer to memory where the hash will be stored. Must not
 *        be NULL and at least RANDOMX_HASH_SIZE bytes must be available for writing.
*/
RANDOMX_EXPORT void randomx_calculate_hash(randomx_vm *machine, const void *input, size_t inputSize, void *output);

RANDOMX_EXPORT void randomx_calculate_hash_first(randomx_vm* machine, uint64_t (&tempHash)[8], const void* input, size_t inputSize);
RANDOMX_EXPORT void randomx_calculate_hash_next(randomx_vm* machine, uint64_t (&tempHash)[8], const void* nextInput, size_t nextInputSize, void* output);

#if defined(__cplusplus)
}
#endif

#endif
