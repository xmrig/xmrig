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

#include <climits>
#include "assembly_generator_x86.hpp"
#include "common.hpp"
#include "reciprocal.h"
#include "program.hpp"
#include "superscalar.hpp"

namespace randomx {

	static const char* regR[] = { "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15" };
	static const char* regR32[] = { "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d" };
	static const char* regFE[] = { "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7" };
	static const char* regF[] = { "xmm0", "xmm1", "xmm2", "xmm3" };
	static const char* regE[] = { "xmm4", "xmm5", "xmm6", "xmm7" };
	static const char* regA[] = { "xmm8", "xmm9", "xmm10", "xmm11" };

	static const char* tempRegx = "xmm12";
	static const char* mantissaMaskReg = "xmm13";
	static const char* exponentMaskReg = "xmm14";
	static const char* scaleMaskReg = "xmm15";
	static const char* regIc = "rbx";
	static const char* regIc32 = "ebx";
	static const char* regIc8 = "bl";
	static const char* regScratchpadAddr = "rsi";

	void AssemblyGeneratorX86::generateProgram(Program& prog) {
		for (unsigned i = 0; i < RegistersCount; ++i) {
			registerUsage[i] = -1;
		}
		asmCode.str(std::string()); //clear
		for (unsigned i = 0; i < prog.getSize(); ++i) {
			asmCode << "randomx_isn_" << i << ":" << std::endl;
			Instruction& instr = prog(i);
			instr.src %= RegistersCount;
			instr.dst %= RegistersCount;
			generateCode(instr, i);
		}
	}

	void AssemblyGeneratorX86::generateAsm(SuperscalarProgram& prog) {
		asmCode.str(std::string()); //clear
#ifdef RANDOMX_ALIGN
		asmCode << "ALIGN 16" << std::endl;
#endif
		for (unsigned i = 0; i < prog.getSize(); ++i) {
			Instruction& instr = prog(i);
			switch ((SuperscalarInstructionType)instr.opcode)
			{
			case SuperscalarInstructionType::ISUB_R:
				asmCode << "sub " << regR[instr.dst] << ", " << regR[instr.src] << std::endl;
				break;
			case SuperscalarInstructionType::IXOR_R:
				asmCode << "xor " << regR[instr.dst] << ", " << regR[instr.src] << std::endl;
				break;
			case SuperscalarInstructionType::IADD_RS:
				asmCode << "lea " << regR[instr.dst] << ", [" << regR[instr.dst] << "+" << regR[instr.src] << "*" << (1 << (instr.getModShift())) << "]" << std::endl;
				break;
			case SuperscalarInstructionType::IMUL_R:
				asmCode << "imul " << regR[instr.dst] << ", " << regR[instr.src] << std::endl;
				break;
			case SuperscalarInstructionType::IROR_C:
				asmCode << "ror " << regR[instr.dst] << ", " << instr.getImm32() << std::endl;
				break;
			case SuperscalarInstructionType::IADD_C7:
				asmCode << "add " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
				break;
			case SuperscalarInstructionType::IXOR_C7:
				asmCode << "xor " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
				break;
			case SuperscalarInstructionType::IADD_C8:
				asmCode << "add " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
#ifdef RANDOMX_ALIGN
				asmCode << "nop" << std::endl;
#endif
				break;
			case SuperscalarInstructionType::IXOR_C8:
				asmCode << "xor " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
#ifdef RANDOMX_ALIGN
				asmCode << "nop" << std::endl;
#endif
				break;
			case SuperscalarInstructionType::IADD_C9:
				asmCode << "add " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
#ifdef RANDOMX_ALIGN
				asmCode << "xchg ax, ax ;nop" << std::endl;
#endif
				break;
			case SuperscalarInstructionType::IXOR_C9:
				asmCode << "xor " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
#ifdef RANDOMX_ALIGN
				asmCode << "xchg ax, ax ;nop" << std::endl;
#endif
				break;
			case SuperscalarInstructionType::IMULH_R:
				asmCode << "mov rax, " << regR[instr.dst] << std::endl;
				asmCode << "mul " << regR[instr.src] << std::endl;
				asmCode << "mov " << regR[instr.dst] << ", rdx" << std::endl;
				break;
			case SuperscalarInstructionType::ISMULH_R:
				asmCode << "mov rax, " << regR[instr.dst] << std::endl;
				asmCode << "imul " << regR[instr.src] << std::endl;
				asmCode << "mov " << regR[instr.dst] << ", rdx" << std::endl;
				break;
			case SuperscalarInstructionType::IMUL_RCP:
				asmCode << "mov rax, " << (int64_t)randomx_reciprocal(instr.getImm32()) << std::endl;
				asmCode << "imul " << regR[instr.dst] << ", rax" << std::endl;
				break;
			default:
				UNREACHABLE;
			}
		}
	}

	void AssemblyGeneratorX86::generateC(SuperscalarProgram& prog) {
		asmCode.str(std::string()); //clear
		asmCode << "#include <stdint.h>" << std::endl;
		asmCode << "#if defined(__SIZEOF_INT128__)" << std::endl;
		asmCode << "	static inline uint64_t mulh(uint64_t a, uint64_t b) {" << std::endl;
		asmCode << "		return ((unsigned __int128)a * b) >> 64;" << std::endl;
		asmCode << "	}" << std::endl;
		asmCode << "	static inline int64_t smulh(int64_t a, int64_t b) {" << std::endl;
		asmCode << "		return ((__int128)a * b) >> 64;" << std::endl;
		asmCode << "	}" << std::endl;
		asmCode << "	#define HAVE_MULH" << std::endl;
		asmCode << "	#define HAVE_SMULH" << std::endl;
		asmCode << "#endif" << std::endl;
		asmCode << "#if defined(_MSC_VER)" << std::endl;
		asmCode << "	#define HAS_VALUE(X) X ## 0" << std::endl;
		asmCode << "	#define EVAL_DEFINE(X) HAS_VALUE(X)" << std::endl;
		asmCode << "	#include <intrin.h>" << std::endl;
		asmCode << "	#include <stdlib.h>" << std::endl;
		asmCode << "	static __inline uint64_t rotr(uint64_t x , int c) {" << std::endl;
		asmCode << "		return _rotr64(x, c);" << std::endl;
		asmCode << "	}" << std::endl;
		asmCode << "	#define HAVE_ROTR" << std::endl;
		asmCode << "	#if EVAL_DEFINE(__MACHINEARM64_X64(1))" << std::endl;
		asmCode << "		static __inline uint64_t mulh(uint64_t a, uint64_t b) {" << std::endl;
		asmCode << "			return __umulh(a, b);" << std::endl;
		asmCode << "		}" << std::endl;
		asmCode << "		#define HAVE_MULH" << std::endl;
		asmCode << "	#endif" << std::endl;
		asmCode << "	#if EVAL_DEFINE(__MACHINEX64(1))" << std::endl;
		asmCode << "		static __inline int64_t smulh(int64_t a, int64_t b) {" << std::endl;
		asmCode << "			int64_t hi;" << std::endl;
		asmCode << "			_mul128(a, b, &hi);" << std::endl;
		asmCode << "			return hi;" << std::endl;
		asmCode << "		}" << std::endl;
		asmCode << "		#define HAVE_SMULH" << std::endl;
		asmCode << "	#endif" << std::endl;
		asmCode << "#endif" << std::endl;
		asmCode << "#ifndef HAVE_ROTR" << std::endl;
		asmCode << "	static inline uint64_t rotr(uint64_t a, int b) {" << std::endl;
		asmCode << "		return (a >> b) | (a << (64 - b));" << std::endl;
		asmCode << "	}" << std::endl;
		asmCode << "	#define HAVE_ROTR" << std::endl;
		asmCode << "#endif" << std::endl;
		asmCode << "#if !defined(HAVE_MULH) || !defined(HAVE_SMULH) || !defined(HAVE_ROTR)" << std::endl;
		asmCode << "	#error \"Required functions are not defined\"" << std::endl;
		asmCode << "#endif" << std::endl;
		asmCode << "void superScalar(uint64_t r[8]) {" << std::endl;
		asmCode << "uint64_t r8 = r[0], r9 = r[1], r10 = r[2], r11 = r[3], r12 = r[4], r13 = r[5], r14 = r[6], r15 = r[7];" << std::endl;
		for (unsigned i = 0; i < prog.getSize(); ++i) {
			Instruction& instr = prog(i);
			switch ((SuperscalarInstructionType)instr.opcode)
			{
			case SuperscalarInstructionType::ISUB_R:
				asmCode << regR[instr.dst] << " -= " << regR[instr.src] << ";" << std::endl;
				break;
			case SuperscalarInstructionType::IXOR_R:
				asmCode << regR[instr.dst] << " ^= " << regR[instr.src] << ";" << std::endl;
				break;
			case SuperscalarInstructionType::IADD_RS:
				asmCode << regR[instr.dst] << " += " << regR[instr.src] << "*" << (1 << (instr.getModShift())) << ";" << std::endl;
				break;
			case SuperscalarInstructionType::IMUL_R:
				asmCode << regR[instr.dst] << " *= " << regR[instr.src] << ";" << std::endl;
				break;
			case SuperscalarInstructionType::IROR_C:
				asmCode << regR[instr.dst] << " = rotr(" << regR[instr.dst] << ", " << instr.getImm32() << ");" << std::endl;
				break;
			case SuperscalarInstructionType::IADD_C7:
			case SuperscalarInstructionType::IADD_C8:
			case SuperscalarInstructionType::IADD_C9:
				asmCode << regR[instr.dst] << " += " << (int32_t)instr.getImm32() << ";" << std::endl;
				break;
			case SuperscalarInstructionType::IXOR_C7:
			case SuperscalarInstructionType::IXOR_C8:
			case SuperscalarInstructionType::IXOR_C9:
				asmCode << regR[instr.dst] << " ^= " << (int32_t)instr.getImm32() << ";" << std::endl;
				break;
			case SuperscalarInstructionType::IMULH_R:
				asmCode << regR[instr.dst] << " = mulh(" << regR[instr.dst] << ", " << regR[instr.src] << ");" << std::endl;
				break;
			case SuperscalarInstructionType::ISMULH_R:
				asmCode << regR[instr.dst] << " = smulh(" << regR[instr.dst] << ", " << regR[instr.src] << ");" << std::endl;
				break;
			case SuperscalarInstructionType::IMUL_RCP:
				asmCode << regR[instr.dst] << " *= " << (int64_t)randomx_reciprocal(instr.getImm32()) << ";" << std::endl;
				break;
			default:
				UNREACHABLE;
			}
		}
		asmCode << "r[0] = r8; r[1] = r9; r[2] = r10; r[3] = r11; r[4] = r12; r[5] = r13; r[6] = r14; r[7] = r15;" << std::endl;
		asmCode << "}" << std::endl;
	}

	void AssemblyGeneratorX86::traceint(Instruction& instr) {
		if (trace) {
			asmCode << "\tpush " << regR[instr.dst] << std::endl;
		}
	}

	void AssemblyGeneratorX86::traceflt(Instruction& instr) {
		if (trace) {
			asmCode << "\tpush 0" << std::endl;
		}
	}

	void AssemblyGeneratorX86::tracenop(Instruction& instr) {
		if (trace) {
			asmCode << "\tpush 0" << std::endl;
		}
	}

	void AssemblyGeneratorX86::generateCode(Instruction& instr, int i) {
		asmCode << "\t; " << instr;
		auto generator = engine[instr.opcode];
		(this->*generator)(instr, i);
	}

	void AssemblyGeneratorX86::genAddressReg(Instruction& instr, const char* reg = "eax") {
		asmCode << "\tlea " << reg << ", [" << regR32[instr.src] << std::showpos << (int32_t)instr.getImm32() << std::noshowpos << "]" << std::endl;
		asmCode << "\tand " << reg << ", " << ((instr.getModMem()) ? ScratchpadL1Mask : ScratchpadL2Mask) << std::endl;
	}

	void AssemblyGeneratorX86::genAddressRegDst(Instruction& instr, int maskAlign = 8) {
		asmCode << "\tlea eax, [" << regR32[instr.dst] << std::showpos << (int32_t)instr.getImm32() << std::noshowpos << "]" << std::endl;
		int mask;
		if (instr.getModCond() < StoreL3Condition) {
			mask = instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask;
		}
		else {
			mask = ScratchpadL3Mask;
		}
		asmCode << "\tand eax" << ", " << (mask & (-maskAlign)) << std::endl;
	}

	int32_t AssemblyGeneratorX86::genAddressImm(Instruction& instr) {
		return (int32_t)instr.getImm32() & ScratchpadL3Mask;
	}

	void AssemblyGeneratorX86::h_IADD_RS(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if(instr.dst == RegisterNeedsDisplacement)
			asmCode << "\tlea " << regR[instr.dst] << ", [" << regR[instr.dst] << "+" << regR[instr.src] << "*" << (1 << (instr.getModShift())) << std::showpos << (int32_t)instr.getImm32() << std::noshowpos << "]" << std::endl;
		else
			asmCode << "\tlea " << regR[instr.dst] << ", [" << regR[instr.dst] << "+" << regR[instr.src] << "*" << (1 << (instr.getModShift())) << "]" << std::endl;
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IADD_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			asmCode << "\tadd " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+rax]" << std::endl;
		}
		else {
			asmCode << "\tadd " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+" << genAddressImm(instr) << "]" << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_ISUB_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			asmCode << "\tsub " << regR[instr.dst] << ", " << regR[instr.src] << std::endl;
		}
		else {
			asmCode << "\tsub " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_ISUB_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			asmCode << "\tsub " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+rax]" << std::endl;
		}
		else {
			asmCode << "\tsub " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+" << genAddressImm(instr) << "]" << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IMUL_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			asmCode << "\timul " << regR[instr.dst] << ", " << regR[instr.src] << std::endl;
		}
		else {
			asmCode << "\timul " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IMUL_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			asmCode << "\timul " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+rax]" << std::endl;
		}
		else {
			asmCode << "\timul " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+" << genAddressImm(instr) << "]" << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IMULH_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		asmCode << "\tmov rax, " << regR[instr.dst] << std::endl;
		asmCode << "\tmul " << regR[instr.src] << std::endl;
		asmCode << "\tmov " << regR[instr.dst] << ", rdx" << std::endl;
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IMULH_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr, "ecx");
			asmCode << "\tmov rax, " << regR[instr.dst] << std::endl;
			asmCode << "\tmul qword ptr [" << regScratchpadAddr << "+rcx]" << std::endl;
		}
		else {
			asmCode << "\tmov rax, " << regR[instr.dst] << std::endl;
			asmCode << "\tmul qword ptr [" << regScratchpadAddr << "+" << genAddressImm(instr) << "]" << std::endl;
		}
		asmCode << "\tmov " << regR[instr.dst] << ", rdx" << std::endl;
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_ISMULH_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		asmCode << "\tmov rax, " << regR[instr.dst] << std::endl;
		asmCode << "\timul " << regR[instr.src] << std::endl;
		asmCode << "\tmov " << regR[instr.dst] << ", rdx" << std::endl;
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_ISMULH_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr, "ecx");
			asmCode << "\tmov rax, " << regR[instr.dst] << std::endl;
			asmCode << "\timul qword ptr [" << regScratchpadAddr << "+rcx]" << std::endl;
		}
		else {
			asmCode << "\tmov rax, " << regR[instr.dst] << std::endl;
			asmCode << "\timul qword ptr [" << regScratchpadAddr << "+" << genAddressImm(instr) << "]" << std::endl;
		}
		asmCode << "\tmov " << regR[instr.dst] << ", rdx" << std::endl;
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_INEG_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		asmCode << "\tneg " << regR[instr.dst] << std::endl;
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IXOR_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			asmCode << "\txor " << regR[instr.dst] << ", " << regR[instr.src] << std::endl;
		}
		else {
			asmCode << "\txor " << regR[instr.dst] << ", " << (int32_t)instr.getImm32() << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IXOR_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			asmCode << "\txor " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+rax]" << std::endl;
		}
		else {
			asmCode << "\txor " << regR[instr.dst] << ", qword ptr [" << regScratchpadAddr << "+" << genAddressImm(instr) << "]" << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IROR_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			asmCode << "\tmov ecx, " << regR32[instr.src] << std::endl;
			asmCode << "\tror " << regR[instr.dst] << ", cl" << std::endl;
		}
		else {
			asmCode << "\tror " << regR[instr.dst] << ", " << (instr.getImm32() & 63) << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IROL_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			asmCode << "\tmov ecx, " << regR32[instr.src] << std::endl;
			asmCode << "\trol " << regR[instr.dst] << ", cl" << std::endl;
		}
		else {
			asmCode << "\trol " << regR[instr.dst] << ", " << (instr.getImm32() & 63) << std::endl;
		}
		traceint(instr);
	}

	void AssemblyGeneratorX86::h_IMUL_RCP(Instruction& instr, int i) {
		uint64_t divisor = instr.getImm32();
		if (!isPowerOf2(divisor)) {
			registerUsage[instr.dst] = i;
			asmCode << "\tmov rax, " << randomx_reciprocal(divisor) << std::endl;
			asmCode << "\timul " << regR[instr.dst] << ", rax" << std::endl;
			traceint(instr);
		}
		else {
			tracenop(instr);
		}
	}

	void AssemblyGeneratorX86::h_ISWAP_R(Instruction& instr, int i) {
		if (instr.src != instr.dst) {
			registerUsage[instr.dst] = i;
			registerUsage[instr.src] = i;
			asmCode << "\txchg " << regR[instr.dst] << ", " << regR[instr.src] << std::endl;
			traceint(instr);
		}
		else {
			tracenop(instr);
		}
	}

	void AssemblyGeneratorX86::h_FSWAP_R(Instruction& instr, int i) {
		asmCode << "\tshufpd " << regFE[instr.dst] << ", " << regFE[instr.dst] << ", 1" << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FADD_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		instr.src %= RegisterCountFlt;
		asmCode << "\taddpd " << regF[instr.dst] << ", " << regA[instr.src] << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FADD_M(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		genAddressReg(instr);
		asmCode << "\tcvtdq2pd " << tempRegx << ", qword ptr [" << regScratchpadAddr << "+rax]" << std::endl;
		asmCode << "\taddpd " << regF[instr.dst] << ", " << tempRegx << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FSUB_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		instr.src %= RegisterCountFlt;
		asmCode << "\tsubpd " << regF[instr.dst] << ", " << regA[instr.src] << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FSUB_M(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		genAddressReg(instr);
		asmCode << "\tcvtdq2pd " << tempRegx << ", qword ptr [" << regScratchpadAddr << "+rax]" << std::endl;
		asmCode << "\tsubpd " << regF[instr.dst] << ", " << tempRegx << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FSCAL_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		asmCode << "\txorps " << regF[instr.dst] << ", " << scaleMaskReg << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FMUL_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		instr.src %= RegisterCountFlt;
		asmCode << "\tmulpd " << regE[instr.dst] << ", " << regA[instr.src] << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FDIV_M(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		genAddressReg(instr);
		asmCode << "\tcvtdq2pd " << tempRegx << ", qword ptr [" << regScratchpadAddr << "+rax]" << std::endl;
		asmCode << "\tandps " << tempRegx << ", " << mantissaMaskReg << std::endl;
		asmCode << "\torps " << tempRegx << ", " << exponentMaskReg << std::endl;
		asmCode << "\tdivpd " << regE[instr.dst] << ", " << tempRegx << std::endl;
		traceflt(instr);
	}

	void AssemblyGeneratorX86::h_FSQRT_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		asmCode << "\tsqrtpd " << regE[instr.dst] << ", " << regE[instr.dst] << std::endl;
		traceflt(instr);
	}	

	void AssemblyGeneratorX86::h_CFROUND(Instruction& instr, int i) {
		asmCode << "\tmov rax, " << regR[instr.src] << std::endl;
		int rotate = (13 - (instr.getImm32() & 63)) & 63;
		if (rotate != 0)
			asmCode << "\trol rax, " << rotate << std::endl;
		asmCode << "\tand eax, 24576" << std::endl;
		asmCode << "\tor eax, 40896" << std::endl;
		asmCode << "\tpush rax" << std::endl;
		asmCode << "\tldmxcsr dword ptr [rsp]" << std::endl;
		asmCode << "\tpop rax" << std::endl;
		tracenop(instr);
	}

	void AssemblyGeneratorX86::h_CBRANCH(Instruction& instr, int i) {
		int reg = instr.dst;
		int target = registerUsage[reg] + 1;
		int shift = instr.getModCond() + ConditionOffset;
		int32_t imm = instr.getImm32() | (1L << shift);
		if (ConditionOffset > 0 || shift > 0)
			imm &= ~(1L << (shift - 1));
		asmCode << "\tadd " << regR[reg] << ", " << imm << std::endl;
		asmCode << "\ttest " << regR[reg] << ", " << (ConditionMask << shift) << std::endl;
		asmCode << "\tjz randomx_isn_" << target << std::endl;
		//mark all registers as used
		for (unsigned j = 0; j < RegistersCount; ++j) {
			registerUsage[j] = i;
		}
	}

	void AssemblyGeneratorX86::h_ISTORE(Instruction& instr, int i) {
		genAddressRegDst(instr);
		asmCode << "\tmov qword ptr [" << regScratchpadAddr << "+rax], " << regR[instr.src] << std::endl;
		tracenop(instr);
	}

	void AssemblyGeneratorX86::h_NOP(Instruction& instr, int i) {
		asmCode << "\tnop" << std::endl;
		tracenop(instr);
	}

#include "instruction_weights.hpp"
#define INST_HANDLE(x) REPN(&AssemblyGeneratorX86::h_##x, WT(x))

	InstructionGenerator AssemblyGeneratorX86::engine[256] = {
		INST_HANDLE(IADD_RS)
		INST_HANDLE(IADD_M)
		INST_HANDLE(ISUB_R)
		INST_HANDLE(ISUB_M)
		INST_HANDLE(IMUL_R)
		INST_HANDLE(IMUL_M)
		INST_HANDLE(IMULH_R)
		INST_HANDLE(IMULH_M)
		INST_HANDLE(ISMULH_R)
		INST_HANDLE(ISMULH_M)
		INST_HANDLE(IMUL_RCP)
		INST_HANDLE(INEG_R)
		INST_HANDLE(IXOR_R)
		INST_HANDLE(IXOR_M)
		INST_HANDLE(IROR_R)
		INST_HANDLE(IROL_R)
		INST_HANDLE(ISWAP_R)
		INST_HANDLE(FSWAP_R)
		INST_HANDLE(FADD_R)
		INST_HANDLE(FADD_M)
		INST_HANDLE(FSUB_R)
		INST_HANDLE(FSUB_M)
		INST_HANDLE(FSCAL_R)
		INST_HANDLE(FMUL_R)
		INST_HANDLE(FDIV_M)
		INST_HANDLE(FSQRT_R)
		INST_HANDLE(CBRANCH)
		INST_HANDLE(CFROUND)
		INST_HANDLE(ISTORE)
		INST_HANDLE(NOP)
	};
}