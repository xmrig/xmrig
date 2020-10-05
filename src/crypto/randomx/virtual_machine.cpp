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

#include <cstring>
#include <iomanip>
#include <stdexcept>
#include "crypto/randomx/virtual_machine.hpp"
#include "crypto/randomx/common.hpp"
#include "crypto/randomx/aes_hash.hpp"
#include "crypto/randomx/blake2/blake2.h"
#include "crypto/randomx/intrin_portable.h"
#include "crypto/randomx/allocator.hpp"
#include "crypto/randomx/soft_aes.h"
#include "base/tools/Profiler.h"

randomx_vm::~randomx_vm() {

}

void randomx_vm::resetRoundingMode() {
	rx_reset_float_state();
}

namespace randomx {

	static inline uint64_t getSmallPositiveFloatBits(uint64_t entropy) {
		auto exponent = entropy >> 59; //0..31
		auto mantissa = entropy & mantissaMask;
		exponent += exponentBias;
		exponent &= exponentMask;
		exponent <<= mantissaSize;
		return exponent | mantissa;
	}

	static inline uint64_t getStaticExponent(uint64_t entropy) {
		auto exponent = constExponentBits;
		exponent |= (entropy >> (64 - staticExponentBits)) << dynamicExponentBits;
		exponent <<= mantissaSize;
		return exponent;
	}

	static inline uint64_t getFloatMask(uint64_t entropy) {
		constexpr uint64_t mask22bit = (1ULL << 22) - 1;
		return (entropy & mask22bit) | getStaticExponent(entropy);
	}

}

void randomx_vm::initialize() {
	store64(&reg.a[0].lo, randomx::getSmallPositiveFloatBits(program.getEntropy(0)));
	store64(&reg.a[0].hi, randomx::getSmallPositiveFloatBits(program.getEntropy(1)));
	store64(&reg.a[1].lo, randomx::getSmallPositiveFloatBits(program.getEntropy(2)));
	store64(&reg.a[1].hi, randomx::getSmallPositiveFloatBits(program.getEntropy(3)));
	store64(&reg.a[2].lo, randomx::getSmallPositiveFloatBits(program.getEntropy(4)));
	store64(&reg.a[2].hi, randomx::getSmallPositiveFloatBits(program.getEntropy(5)));
	store64(&reg.a[3].lo, randomx::getSmallPositiveFloatBits(program.getEntropy(6)));
	store64(&reg.a[3].hi, randomx::getSmallPositiveFloatBits(program.getEntropy(7)));
	mem.ma = program.getEntropy(8) & CacheLineAlignMask;
	mem.mx = program.getEntropy(10);
	auto addressRegisters = program.getEntropy(12);
	config.readReg0 = 0 + (addressRegisters & 1);
	addressRegisters >>= 1;
	config.readReg1 = 2 + (addressRegisters & 1);
	addressRegisters >>= 1;
	config.readReg2 = 4 + (addressRegisters & 1);
	addressRegisters >>= 1;
	config.readReg3 = 6 + (addressRegisters & 1);
	datasetOffset = (program.getEntropy(13) % (DatasetExtraItems + 1)) * randomx::CacheLineSize;
	store64(&config.eMask[0], randomx::getFloatMask(program.getEntropy(14)));
	store64(&config.eMask[1], randomx::getFloatMask(program.getEntropy(15)));
}

namespace randomx {

	template<int softAes>
	VmBase<softAes>::~VmBase() {
	}

	template<int softAes>
	void VmBase<softAes>::setScratchpad(uint8_t *scratchpad) {
		if (datasetPtr == nullptr) {
			throw std::invalid_argument("Cache/Dataset not set");
		}

		this->scratchpad = scratchpad;
	}

	template<int softAes>
	void VmBase<softAes>::getFinalResult(void* out) {
		hashAes1Rx4<softAes>(scratchpad, ScratchpadSize, &reg.a);
		rx_blake2b_wrapper::run(out, RANDOMX_HASH_SIZE, &reg, sizeof(RegisterFile));
	}

	template<int softAes>
	void VmBase<softAes>::hashAndFill(void* out, uint64_t (&fill_state)[8]) {
		if (!softAes) {
			hashAndFillAes1Rx4<0, 2>(scratchpad, ScratchpadSize, &reg.a, fill_state);
		}
		else {
			(*GetSoftAESImpl())(scratchpad, ScratchpadSize, &reg.a, fill_state);
		}

		rx_blake2b_wrapper::run(out, RANDOMX_HASH_SIZE, &reg, sizeof(RegisterFile));
	}

	template<int softAes>
	void VmBase<softAes>::initScratchpad(void* seed) {
		fillAes1Rx4<softAes>(seed, ScratchpadSize, scratchpad);
	}

	template<int softAes>
	void VmBase<softAes>::generateProgram(void* seed) {
		PROFILE_SCOPE(RandomX_generate_program);
		fillAes4Rx4<softAes>(seed, 128 + RandomX_CurrentConfig.ProgramSize * 8, &program);
	}

	template class VmBase<false>;
	template class VmBase<true>;
}
