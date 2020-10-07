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
#include <cstring>
#include <vector>
#include "crypto/randomx/common.hpp"

namespace randomx {

	class Program;
	struct ProgramConfiguration;
	class SuperscalarProgram;
	class JitCompilerX86;
	class Instruction;

	typedef void(*InstructionGeneratorX86)(JitCompilerX86*, const Instruction&);

	constexpr uint32_t CodeSize = 64 * 1024;

	class JitCompilerX86 {
	public:
		explicit JitCompilerX86(bool hugePagesEnable);
		~JitCompilerX86();
		void prepare();
		void generateProgram(Program&, ProgramConfiguration&, uint32_t);
		void generateProgramLight(Program&, ProgramConfiguration&, uint32_t);
		template<size_t N>
		void generateSuperscalarHash(SuperscalarProgram (&programs)[N], std::vector<uint64_t> &);
		void generateDatasetInitCode();
		ProgramFunc* getProgramFunc() {
			return (ProgramFunc*)code;
		}
		DatasetInitFunc* getDatasetInitFunc() {
			return (DatasetInitFunc*)code;
		}
		uint8_t* getCode() {
			return code;
		}
		size_t getCodeSize();

		alignas(64) static InstructionGeneratorX86 engine[256];

		int registerUsage[RegistersCount];
		uint8_t* code;
		uint32_t codePos;
		uint32_t codePosFirst;
		uint32_t vm_flags;

#		ifdef XMRIG_FIX_RYZEN
		std::pair<const void*, const void*> mainLoopBounds;
#		endif

		bool BranchesWithin32B = false;
		bool hasAVX;
		bool hasXOP;

		uint8_t* allocatedCode;

		void generateProgramPrologue(Program&, ProgramConfiguration&);
		void generateProgramEpilogue(Program&, ProgramConfiguration&);
		template<bool rax>
		static void genAddressReg(const Instruction&, const uint32_t src, uint8_t* code, uint32_t& codePos);
		static void genAddressRegDst(const Instruction&, uint8_t* code, uint32_t& codePos);
		static void genAddressImm(const Instruction&, uint8_t* code, uint32_t& codePos);
		static void genSIB(int scale, int index, int base, uint8_t* code, uint32_t& codePos);

		void generateSuperscalarCode(Instruction &, std::vector<uint64_t> &);

		static void emitByte(uint8_t val, uint8_t* code, uint32_t& codePos) {
			code[codePos] = val;
			++codePos;
		}

		static void emit32(uint32_t val, uint8_t* code, uint32_t& codePos) {
			memcpy(code + codePos, &val, sizeof val);
			codePos += sizeof val;
		}

		static void emit64(uint64_t val, uint8_t* code, uint32_t& codePos) {
			memcpy(code + codePos, &val, sizeof val);
			codePos += sizeof val;
		}

		template<size_t N>
		static void emit(const uint8_t (&src)[N], uint8_t* code, uint32_t& codePos) {
			emit(src, N, code, codePos);
		}

		static void emit(const uint8_t* src, size_t count, uint8_t* code, uint32_t& codePos) {
			memcpy(code + codePos, src, count);
			codePos += count;
		}

		void h_IADD_RS(const Instruction&);
		void h_IADD_M(const Instruction&);
		void h_ISUB_R(const Instruction&);
		void h_ISUB_M(const Instruction&);
		void h_IMUL_R(const Instruction&);
		void h_IMUL_M(const Instruction&);
		void h_IMULH_R(const Instruction&);
		void h_IMULH_R_BMI2(const Instruction&);
		void h_IMULH_M(const Instruction&);
		void h_IMULH_M_BMI2(const Instruction&);
		void h_ISMULH_R(const Instruction&);
		void h_ISMULH_M(const Instruction&);
		void h_IMUL_RCP(const Instruction&);
		void h_INEG_R(const Instruction&);
		void h_IXOR_R(const Instruction&);
		void h_IXOR_M(const Instruction&);
		void h_IROR_R(const Instruction&);
		void h_IROL_R(const Instruction&);
		void h_ISWAP_R(const Instruction&);
		void h_FSWAP_R(const Instruction&);
		void h_FADD_R(const Instruction&);
		void h_FADD_M(const Instruction&);
		void h_FSUB_R(const Instruction&);
		void h_FSUB_M(const Instruction&);
		void h_FSCAL_R(const Instruction&);
		void h_FMUL_R(const Instruction&);
		void h_FDIV_M(const Instruction&);
		void h_FSQRT_R(const Instruction&);

		template<bool jccErratum>
		void h_CBRANCH(const Instruction&);

		void h_CFROUND(const Instruction&);
		void h_CFROUND_BMI2(const Instruction&);
		void h_ISTORE(const Instruction&);
		void h_NOP(const Instruction&);
	};
}
