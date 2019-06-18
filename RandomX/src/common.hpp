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

#pragma once

#include <cstdint>
#include <iostream>
#include <climits>
#include "blake2/endian.h"
#include "configuration.h"
#include "randomx.h"

namespace randomx {

	static_assert(RANDOMX_ARGON_MEMORY > 0, "RANDOMX_ARGON_MEMORY must be greater than 0.");
	static_assert((RANDOMX_ARGON_MEMORY & (RANDOMX_ARGON_MEMORY - 1)) == 0, "RANDOMX_ARGON_MEMORY must be a power of 2.");
	static_assert(RANDOMX_DATASET_BASE_SIZE >= 64, "RANDOMX_DATASET_BASE_SIZE must be at least 64.");
	static_assert((RANDOMX_DATASET_BASE_SIZE & (RANDOMX_DATASET_BASE_SIZE - 1)) == 0, "RANDOMX_DATASET_BASE_SIZE must be a power of 2.");
	static_assert(RANDOMX_DATASET_BASE_SIZE <= 4294967296ULL, "RANDOMX_DATASET_BASE_SIZE must not exceed 4294967296.");
	static_assert(RANDOMX_DATASET_EXTRA_SIZE % 64 == 0, "RANDOMX_DATASET_EXTRA_SIZE must be divisible by 64.");
	static_assert(RANDOMX_PROGRAM_SIZE > 0, "RANDOMX_PROGRAM_SIZE must be greater than 0");
	static_assert(RANDOMX_PROGRAM_ITERATIONS > 0, "RANDOMX_PROGRAM_ITERATIONS must be greater than 0");
	static_assert(RANDOMX_PROGRAM_COUNT > 0, "RANDOMX_PROGRAM_COUNT must be greater than 0");
	static_assert((RANDOMX_SCRATCHPAD_L3 & (RANDOMX_SCRATCHPAD_L3 - 1)) == 0, "RANDOMX_SCRATCHPAD_L3 must be a power of 2.");
	static_assert(RANDOMX_SCRATCHPAD_L3 >= RANDOMX_SCRATCHPAD_L2, "RANDOMX_SCRATCHPAD_L3 must be greater than or equal to RANDOMX_SCRATCHPAD_L2.");
	static_assert((RANDOMX_SCRATCHPAD_L2 & (RANDOMX_SCRATCHPAD_L2 - 1)) == 0, "RANDOMX_SCRATCHPAD_L2 must be a power of 2.");
	static_assert(RANDOMX_SCRATCHPAD_L2 >= RANDOMX_SCRATCHPAD_L1, "RANDOMX_SCRATCHPAD_L2 must be greater than or equal to RANDOMX_SCRATCHPAD_L1.");
	static_assert(RANDOMX_SCRATCHPAD_L1 >= 64, "RANDOMX_SCRATCHPAD_L1 must be at least 64.");
	static_assert((RANDOMX_SCRATCHPAD_L1 & (RANDOMX_SCRATCHPAD_L1 - 1)) == 0, "RANDOMX_SCRATCHPAD_L1 must be a power of 2.");
	static_assert(RANDOMX_CACHE_ACCESSES > 1, "RANDOMX_CACHE_ACCESSES must be greater than 1");
	static_assert(RANDOMX_SUPERSCALAR_LATENCY > 0, "RANDOMX_SUPERSCALAR_LATENCY must be greater than 0");
	static_assert(RANDOMX_JUMP_BITS > 0, "RANDOMX_JUMP_BITS must be greater than 0.");
	static_assert(RANDOMX_JUMP_OFFSET >= 0, "RANDOMX_JUMP_OFFSET must be greater than or equal to 0.");
	static_assert(RANDOMX_JUMP_BITS + RANDOMX_JUMP_OFFSET <= 16, "RANDOMX_JUMP_BITS + RANDOMX_JUMP_OFFSET must not exceed 16.");

	constexpr int wtSum = RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + \
		RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + \
		RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M + RANDOMX_FREQ_IMUL_RCP + \
		RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_ISWAP_R + \
		RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R + RANDOMX_FREQ_FADD_M + RANDOMX_FREQ_FSUB_R + RANDOMX_FREQ_FSUB_M + \
		RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R + RANDOMX_FREQ_FDIV_M + RANDOMX_FREQ_FSQRT_R + RANDOMX_FREQ_CBRANCH + \
		RANDOMX_FREQ_CFROUND + RANDOMX_FREQ_ISTORE + RANDOMX_FREQ_NOP;

	static_assert(wtSum == 256,	"Sum of instruction frequencies must be 256.");


	constexpr uint32_t ArgonBlockSize = 1024;
	constexpr int ArgonSaltSize = sizeof("" RANDOMX_ARGON_SALT) - 1;
	constexpr int SuperscalarMaxSize = 3 * RANDOMX_SUPERSCALAR_LATENCY + 2;
	constexpr int CacheLineSize = RANDOMX_DATASET_ITEM_SIZE;
	constexpr int ScratchpadSize = RANDOMX_SCRATCHPAD_L3;
	constexpr uint32_t CacheLineAlignMask = (RANDOMX_DATASET_BASE_SIZE - 1) & ~(CacheLineSize - 1);
	constexpr uint32_t CacheSize = RANDOMX_ARGON_MEMORY * ArgonBlockSize;
	constexpr uint64_t DatasetSize = RANDOMX_DATASET_BASE_SIZE + RANDOMX_DATASET_EXTRA_SIZE;
	constexpr uint32_t DatasetExtraItems = RANDOMX_DATASET_EXTRA_SIZE / RANDOMX_DATASET_ITEM_SIZE;
	constexpr uint32_t ConditionMask = ((1 << RANDOMX_JUMP_BITS) - 1);
	constexpr int ConditionOffset = RANDOMX_JUMP_OFFSET;
	constexpr int StoreL3Condition = 14;

	//Prevent some unsafe configurations.
#ifndef RANDOMX_UNSAFE
	static_assert((uint64_t)ArgonBlockSize * RANDOMX_CACHE_ACCESSES * RANDOMX_ARGON_MEMORY + 33554432 >= (uint64_t)RANDOMX_DATASET_BASE_SIZE + RANDOMX_DATASET_EXTRA_SIZE, "Unsafe configuration: Memory-time tradeoffs");
	static_assert((128 + RANDOMX_PROGRAM_SIZE * RANDOMX_FREQ_ISTORE / 256) * (RANDOMX_PROGRAM_COUNT * RANDOMX_PROGRAM_ITERATIONS) >= RANDOMX_SCRATCHPAD_L3, "Unsafe configuration: Insufficient Scratchpad writes");
	static_assert(RANDOMX_PROGRAM_COUNT > 1, "Unsafe configuration: Program filtering strategies");
	static_assert(RANDOMX_PROGRAM_SIZE >= 64, "Unsafe configuration: Low program entropy");
	static_assert(RANDOMX_PROGRAM_ITERATIONS >= 400, "Unsafe configuration: High compilation overhead");
#endif

#ifdef TRACE
	constexpr bool trace = true;
#else
	constexpr bool trace = false;
#endif

#ifndef UNREACHABLE
#ifdef __GNUC__
#define UNREACHABLE __builtin_unreachable()
#elif _MSC_VER
#define UNREACHABLE __assume(false)
#else
#define UNREACHABLE
#endif
#endif

#if defined(_M_X64) || defined(__x86_64__)
	class JitCompilerX86;
	using JitCompiler = JitCompilerX86;
#elif defined(__aarch64__)
	class JitCompilerA64;
	using JitCompiler = JitCompilerA64;
#else
	class JitCompilerFallback;
	using JitCompiler = JitCompilerFallback;
#endif

	using addr_t = uint32_t;

	using int_reg_t = uint64_t;

	struct fpu_reg_t {
		double lo;
		double hi;
	};

	constexpr uint32_t ScratchpadL1 = RANDOMX_SCRATCHPAD_L1 / sizeof(int_reg_t);
	constexpr uint32_t ScratchpadL2 = RANDOMX_SCRATCHPAD_L2 / sizeof(int_reg_t);
	constexpr uint32_t ScratchpadL3 = RANDOMX_SCRATCHPAD_L3 / sizeof(int_reg_t);
	constexpr int ScratchpadL1Mask = (ScratchpadL1 - 1) * 8;
	constexpr int ScratchpadL2Mask = (ScratchpadL2 - 1) * 8;
	constexpr int ScratchpadL1Mask16 = (ScratchpadL1 / 2 - 1) * 16;
	constexpr int ScratchpadL2Mask16 = (ScratchpadL2 / 2 - 1) * 16;
	constexpr int ScratchpadL3Mask = (ScratchpadL3 - 1) * 8;
	constexpr int ScratchpadL3Mask64 = (ScratchpadL3 / 8 - 1) * 64;
	constexpr int RegistersCount = 8;
	constexpr int RegisterCountFlt = RegistersCount / 2;
	constexpr int RegisterNeedsDisplacement = 5; //x86 r13 register
	constexpr int RegisterNeedsSib = 4; //x86 r12 register

	inline bool isPowerOf2(uint64_t x) {
		return (x & (x - 1)) == 0;
	}

	constexpr int mantissaSize = 52;
	constexpr int exponentSize = 11;
	constexpr uint64_t mantissaMask = (1ULL << mantissaSize) - 1;
	constexpr uint64_t exponentMask = (1ULL << exponentSize) - 1;
	constexpr int exponentBias = 1023;
	constexpr int dynamicExponentBits = 4;
	constexpr int staticExponentBits = 4;
	constexpr uint64_t constExponentBits = 0x300;
	constexpr uint64_t dynamicMantissaMask = (1ULL << (mantissaSize + dynamicExponentBits)) - 1;

	struct MemoryRegisters {
		addr_t mx, ma;
		uint8_t* memory = nullptr;
	};

	struct RegisterFile {
		int_reg_t r[RegistersCount];
		fpu_reg_t f[RegistersCount / 2];
		fpu_reg_t e[RegistersCount / 2];
		fpu_reg_t a[RegistersCount / 2];
	};

	typedef void(DatasetReadFunc)(addr_t, MemoryRegisters&, int_reg_t(&reg)[RegistersCount]);
	typedef void(ProgramFunc)(RegisterFile&, MemoryRegisters&, uint8_t* /* scratchpad */, uint64_t);
	typedef void(DatasetInitFunc)(randomx_cache* cache, uint8_t* dataset, uint32_t startBlock, uint32_t endBlock);

	typedef void(DatasetDeallocFunc)(randomx_dataset*);
	typedef void(CacheDeallocFunc)(randomx_cache*);
	typedef void(CacheInitializeFunc)(randomx_cache*, const void*, size_t);
}
