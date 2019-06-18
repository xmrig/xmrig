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

#include <new>
#include <vector>
#include "common.hpp"
#include "virtual_machine.hpp"
#include "intrin_portable.h"
#include "allocator.hpp"

namespace randomx {

	struct InstructionByteCode {
		union {
			int_reg_t* idst;
			rx_vec_f128* fdst;
		};
		union {
			int_reg_t* isrc;
			rx_vec_f128* fsrc;
		};
		union {
			uint64_t imm;
			int64_t simm;
		};
		InstructionType type;
		union {
			int16_t target;
			uint16_t shift;
		};
		uint32_t memMask;
	};

	static_assert(sizeof(InstructionByteCode) == 32, "Invalid packing of struct InstructionByteCode");

	template<class Allocator, bool softAes>
	class InterpretedVm : public VmBase<Allocator, softAes> {
	public:
		using VmBase<Allocator, softAes>::mem;
		using VmBase<Allocator, softAes>::scratchpad;
		using VmBase<Allocator, softAes>::program;
		using VmBase<Allocator, softAes>::config;
		using VmBase<Allocator, softAes>::reg;
		using VmBase<Allocator, softAes>::datasetPtr;
		using VmBase<Allocator, softAes>::datasetOffset;
		void* operator new(size_t size) {
			void* ptr = AlignedAllocator<CacheLineSize>::allocMemory(size);
			if (ptr == nullptr)
				throw std::bad_alloc();
			return ptr;
		}
		void operator delete(void* ptr) {
			AlignedAllocator<CacheLineSize>::freeMemory(ptr, sizeof(InterpretedVm));
		}
		void run(void* seed) override;
		void setDataset(randomx_dataset* dataset) override;
	protected:
		virtual void datasetRead(uint64_t blockNumber, int_reg_t(&r)[RegistersCount]);
		virtual void datasetPrefetch(uint64_t blockNumber);
	private:
		void execute();
		void precompileProgram(int_reg_t(&r)[RegistersCount], rx_vec_f128(&f)[RegisterCountFlt], rx_vec_f128(&e)[RegisterCountFlt], rx_vec_f128(&a)[RegisterCountFlt]);
		void executeBytecode(int_reg_t(&r)[RegistersCount], rx_vec_f128(&f)[RegisterCountFlt], rx_vec_f128(&e)[RegisterCountFlt], rx_vec_f128(&a)[RegisterCountFlt]);
		void executeBytecode(int& i, int_reg_t(&r)[RegistersCount], rx_vec_f128(&f)[RegisterCountFlt], rx_vec_f128(&e)[RegisterCountFlt], rx_vec_f128(&a)[RegisterCountFlt]);
		void* getScratchpadAddress(InstructionByteCode& ibc);
		rx_vec_f128 maskRegisterExponentMantissa(rx_vec_f128);

		InstructionByteCode byteCode[RANDOMX_PROGRAM_SIZE];
	};

	using InterpretedVmDefault = InterpretedVm<AlignedAllocator<CacheLineSize>, true>;
	using InterpretedVmHardAes = InterpretedVm<AlignedAllocator<CacheLineSize>, false>;
	using InterpretedVmLargePage = InterpretedVm<LargePageAllocator, true>;
	using InterpretedVmLargePageHardAes = InterpretedVm<LargePageAllocator, false>;
}