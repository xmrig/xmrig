/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>
Copyright (c) 2019, SChernykh    <https://github.com/SChernykh>

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
#include <vector>
#include <stdexcept>
#include "crypto/randomx/common.hpp"
#include "crypto/randomx/jit_compiler_a64_static.hpp"

namespace randomx {

	class Program;
	class ProgramConfiguration;
	class SuperscalarProgram;
	class Instruction;

	typedef void(JitCompilerA64::*InstructionGeneratorA64)(Instruction&, uint32_t&);

	class JitCompilerA64 {
	public:
		JitCompilerA64();
		~JitCompilerA64();

		void generateProgram(Program&, ProgramConfiguration&, uint32_t);
		void generateProgramLight(Program&, ProgramConfiguration&, uint32_t);

		template<size_t N>
		void generateSuperscalarHash(SuperscalarProgram(&programs)[N], std::vector<uint64_t> &);

		void generateDatasetInitCode() {}

		ProgramFunc* getProgramFunc() { return reinterpret_cast<ProgramFunc*>(code); }
		DatasetInitFunc* getDatasetInitFunc();
		uint8_t* getCode() { return code; }
		size_t getCodeSize();

		static InstructionGeneratorA64 engine[256];
		uint32_t reg_changed_offset[8];
		uint8_t* code;
		uint32_t literalPos;
		uint32_t num32bitLiterals;

		static void emit32(uint32_t val, uint8_t* code, uint32_t& codePos)
		{
			*(uint32_t*)(code + codePos) = val;
			codePos += sizeof(val);
		}

		static void emit64(uint64_t val, uint8_t* code, uint32_t& codePos)
		{
			*(uint64_t*)(code + codePos) = val;
			codePos += sizeof(val);
		}

		void emitMovImmediate(uint32_t dst, uint32_t imm, uint8_t* code, uint32_t& codePos);
		void emitAddImmediate(uint32_t dst, uint32_t src, uint32_t imm, uint8_t* code, uint32_t& codePos);

		template<uint32_t tmp_reg>
		void emitMemLoad(uint32_t dst, uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos);

		template<uint32_t tmp_reg_fp>
		void emitMemLoadFP(uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos);

		void h_IADD_RS(Instruction&, uint32_t&);
		void h_IADD_M(Instruction&, uint32_t&);
		void h_ISUB_R(Instruction&, uint32_t&);
		void h_ISUB_M(Instruction&, uint32_t&);
		void h_IMUL_R(Instruction&, uint32_t&);
		void h_IMUL_M(Instruction&, uint32_t&);
		void h_IMULH_R(Instruction&, uint32_t&);
		void h_IMULH_M(Instruction&, uint32_t&);
		void h_ISMULH_R(Instruction&, uint32_t&);
		void h_ISMULH_M(Instruction&, uint32_t&);
		void h_IMUL_RCP(Instruction&, uint32_t&);
		void h_INEG_R(Instruction&, uint32_t&);
		void h_IXOR_R(Instruction&, uint32_t&);
		void h_IXOR_M(Instruction&, uint32_t&);
		void h_IROR_R(Instruction&, uint32_t&);
		void h_IROL_R(Instruction&, uint32_t&);
		void h_ISWAP_R(Instruction&, uint32_t&);
		void h_FSWAP_R(Instruction&, uint32_t&);
		void h_FADD_R(Instruction&, uint32_t&);
		void h_FADD_M(Instruction&, uint32_t&);
		void h_FSUB_R(Instruction&, uint32_t&);
		void h_FSUB_M(Instruction&, uint32_t&);
		void h_FSCAL_R(Instruction&, uint32_t&);
		void h_FMUL_R(Instruction&, uint32_t&);
		void h_FDIV_M(Instruction&, uint32_t&);
		void h_FSQRT_R(Instruction&, uint32_t&);
		void h_CBRANCH(Instruction&, uint32_t&);
		void h_CFROUND(Instruction&, uint32_t&);
		void h_ISTORE(Instruction&, uint32_t&);
		void h_NOP(Instruction&, uint32_t&);
	};
}
