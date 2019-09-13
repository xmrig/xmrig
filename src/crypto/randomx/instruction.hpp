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
#include <type_traits>
#include "crypto/randomx/blake2/endian.h"

namespace randomx {

	class Instruction;

	enum class InstructionType : uint16_t {
		IADD_RS = 0,
		IADD_M = 1,
		ISUB_R = 2,
		ISUB_M = 3,
		IMUL_R = 4,
		IMUL_M = 5,
		IMULH_R = 6,
		IMULH_M = 7,
		ISMULH_R = 8,
		ISMULH_M = 9,
		IMUL_RCP = 10,
		INEG_R = 11,
		IXOR_R = 12,
		IXOR_M = 13,
		IROR_R = 14,
		IROL_R = 15,
		ISWAP_R = 16,
		FSWAP_R = 17,
		FADD_R = 18,
		FADD_M = 19,
		FSUB_R = 20,
		FSUB_M = 21,
		FSCAL_R = 22,
		FMUL_R = 23,
		FDIV_M = 24,
		FSQRT_R = 25,
		CBRANCH = 26,
		CFROUND = 27,
		ISTORE = 28,
		NOP = 29,
	};

	class Instruction {
	public:
		uint32_t getImm32() const {
			return load32(&imm32);
		}
		void setImm32(uint32_t val) {
			return store32(&imm32, val);
		}
		uint32_t getModMem() const {
			return mod & 3; //bits 0-1
		}
		uint32_t getModShift() const {
			return (mod >> 2) & 3; //bits 2-3
		}
		uint32_t getModCond() const {
			return mod >> 4; //bits 4-7
		}
		void setMod(uint8_t val) {
			mod = val;
		}

		uint8_t opcode;
		uint8_t dst;
		uint8_t src;
		uint8_t mod;
		uint32_t imm32;
	};

	static_assert(sizeof(Instruction) == 8, "Invalid size of struct randomx::Instruction");
	static_assert(std::is_standard_layout<Instruction>(), "randomx::Instruction must be a standard-layout struct");
}
