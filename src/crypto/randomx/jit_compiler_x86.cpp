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

#include <stdexcept>
#include <cstring>
#include <climits>
#include "jit_compiler_x86.hpp"
#include "jit_compiler_x86_static.hpp"
#include "superscalar.hpp"
#include "program.hpp"
#include "reciprocal.h"
#include "virtual_memory.hpp"

namespace randomx {
	/*

	REGISTER ALLOCATION:

	; rax -> temporary
	; rbx -> iteration counter "ic"
	; rcx -> temporary
	; rdx -> temporary
	; rsi -> scratchpad pointer
	; rdi -> dataset pointer
	; rbp -> memory registers "ma" (high 32 bits), "mx" (low 32 bits)
	; rsp -> stack pointer
	; r8  -> "r0"
	; r9  -> "r1"
	; r10 -> "r2"
	; r11 -> "r3"
	; r12 -> "r4"
	; r13 -> "r5"
	; r14 -> "r6"
	; r15 -> "r7"
	; xmm0 -> "f0"
	; xmm1 -> "f1"
	; xmm2 -> "f2"
	; xmm3 -> "f3"
	; xmm4 -> "e0"
	; xmm5 -> "e1"
	; xmm6 -> "e2"
	; xmm7 -> "e3"
	; xmm8 -> "a0"
	; xmm9 -> "a1"
	; xmm10 -> "a2"
	; xmm11 -> "a3"
	; xmm12 -> temporary
	; xmm13 -> E 'and' mask = 0x00ffffffffffffff00ffffffffffffff
	; xmm14 -> E 'or' mask  = 0x3*00000000******3*00000000******
	; xmm15 -> scale mask   = 0x81f000000000000081f0000000000000

	*/

	const uint8_t* codePrologue = (uint8_t*)&randomx_program_prologue;
	const uint8_t* codeLoopBegin = (uint8_t*)&randomx_program_loop_begin;
	const uint8_t* codeLoopLoad = (uint8_t*)&randomx_program_loop_load;
	const uint8_t* codeProgamStart = (uint8_t*)&randomx_program_start;
	const uint8_t* codeReadDataset = (uint8_t*)&randomx_program_read_dataset;
	const uint8_t* codeReadDatasetLightSshInit = (uint8_t*)&randomx_program_read_dataset_sshash_init;
	const uint8_t* codeReadDatasetLightSshFin = (uint8_t*)&randomx_program_read_dataset_sshash_fin;
	const uint8_t* codeDatasetInit = (uint8_t*)&randomx_dataset_init;
	const uint8_t* codeLoopStore = (uint8_t*)&randomx_program_loop_store;
	const uint8_t* codeLoopEnd = (uint8_t*)&randomx_program_loop_end;
	const uint8_t* codeEpilogue = (uint8_t*)&randomx_program_epilogue;
	const uint8_t* codeProgramEnd = (uint8_t*)&randomx_program_end;
	const uint8_t* codeShhLoad = (uint8_t*)&randomx_sshash_load;
	const uint8_t* codeShhPrefetch = (uint8_t*)&randomx_sshash_prefetch;
	const uint8_t* codeShhEnd = (uint8_t*)&randomx_sshash_end;
	const uint8_t* codeShhInit = (uint8_t*)&randomx_sshash_init;

	const int32_t prologueSize = codeLoopBegin - codePrologue;
	const int32_t loopLoadSize = codeProgamStart - codeLoopLoad;
	const int32_t readDatasetSize = codeReadDatasetLightSshInit - codeReadDataset;
	const int32_t readDatasetLightInitSize = codeReadDatasetLightSshFin - codeReadDatasetLightSshInit;
	const int32_t readDatasetLightFinSize = codeLoopStore - codeReadDatasetLightSshFin;
	const int32_t loopStoreSize = codeLoopEnd - codeLoopStore;
	const int32_t datasetInitSize = codeEpilogue - codeDatasetInit;
	const int32_t epilogueSize = codeShhLoad - codeEpilogue;
	const int32_t codeSshLoadSize = codeShhPrefetch - codeShhLoad;
	const int32_t codeSshPrefetchSize = codeShhEnd - codeShhPrefetch;
	const int32_t codeSshInitSize = codeProgramEnd - codeShhInit;

	const int32_t epilogueOffset = CodeSize - epilogueSize;
	constexpr int32_t superScalarHashOffset = 32768;

	static const uint8_t REX_ADD_RR[] = { 0x4d, 0x03 };
	static const uint8_t REX_ADD_RM[] = { 0x4c, 0x03 };
	static const uint8_t REX_SUB_RR[] = { 0x4d, 0x2b };
	static const uint8_t REX_SUB_RM[] = { 0x4c, 0x2b };
	static const uint8_t REX_MOV_RR[] = { 0x41, 0x8b };
	static const uint8_t REX_MOV_RR64[] = { 0x49, 0x8b };
	static const uint8_t REX_MOV_R64R[] = { 0x4c, 0x8b };
	static const uint8_t REX_IMUL_RR[] = { 0x4d, 0x0f, 0xaf };
	static const uint8_t REX_IMUL_RRI[] = { 0x4d, 0x69 };
	static const uint8_t REX_IMUL_RM[] = { 0x4c, 0x0f, 0xaf };
	static const uint8_t REX_MUL_R[] = { 0x49, 0xf7 };
	static const uint8_t REX_MUL_M[] = { 0x48, 0xf7 };
	static const uint8_t REX_81[] = { 0x49, 0x81 };
	static const uint8_t AND_EAX_I = 0x25;
	static const uint8_t MOV_EAX_I = 0xb8;
	static const uint8_t MOV_RAX_I[] = { 0x48, 0xb8 };
	static const uint8_t MOV_RCX_I[] = { 0x48, 0xb9 };
	static const uint8_t REX_LEA[] = { 0x4f, 0x8d };
	static const uint8_t REX_MUL_MEM[] = { 0x48, 0xf7, 0x24, 0x0e };
	static const uint8_t REX_IMUL_MEM[] = { 0x48, 0xf7, 0x2c, 0x0e };
	static const uint8_t REX_SHR_RAX[] = { 0x48, 0xc1, 0xe8 };
	static const uint8_t RAX_ADD_SBB_1[] = { 0x48, 0x83, 0xC0, 0x01, 0x48, 0x83, 0xD8, 0x00 };
	static const uint8_t MUL_RCX[] = { 0x48, 0xf7, 0xe1 };
	static const uint8_t REX_SHR_RDX[] = { 0x48, 0xc1, 0xea };
	static const uint8_t REX_SH[] = { 0x49, 0xc1 };
	static const uint8_t MOV_RCX_RAX_SAR_RCX_63[] = { 0x48, 0x89, 0xc1, 0x48, 0xc1, 0xf9, 0x3f };
	static const uint8_t AND_ECX_I[] = { 0x81, 0xe1 };
	static const uint8_t ADD_RAX_RCX[] = { 0x48, 0x01, 0xC8 };
	static const uint8_t SAR_RAX_I8[] = { 0x48, 0xC1, 0xF8 };
	static const uint8_t NEG_RAX[] = { 0x48, 0xF7, 0xD8 };
	static const uint8_t ADD_R_RAX[] = { 0x4C, 0x03 };
	static const uint8_t XOR_EAX_EAX[] = { 0x33, 0xC0 };
	static const uint8_t ADD_RDX_R[] = { 0x4c, 0x01 };
	static const uint8_t SUB_RDX_R[] = { 0x4c, 0x29 };
	static const uint8_t SAR_RDX_I8[] = { 0x48, 0xC1, 0xFA };
	static const uint8_t TEST_RDX_RDX[] = { 0x48, 0x85, 0xD2 };
	static const uint8_t SETS_AL_ADD_RDX_RAX[] = { 0x0F, 0x98, 0xC0, 0x48, 0x03, 0xD0 };
	static const uint8_t REX_NEG[] = { 0x49, 0xF7 };
	static const uint8_t REX_XOR_RR[] = { 0x4D, 0x33 };
	static const uint8_t REX_XOR_RI[] = { 0x49, 0x81 };
	static const uint8_t REX_XOR_RM[] = { 0x4c, 0x33 };
	static const uint8_t REX_ROT_CL[] = { 0x49, 0xd3 };
	static const uint8_t REX_ROT_I8[] = { 0x49, 0xc1 };
	static const uint8_t SHUFPD[] = { 0x66, 0x0f, 0xc6 };
	static const uint8_t REX_ADDPD[] = { 0x66, 0x41, 0x0f, 0x58 };
	static const uint8_t REX_CVTDQ2PD_XMM12[] = { 0xf3, 0x44, 0x0f, 0xe6, 0x24, 0x06 };
	static const uint8_t REX_SUBPD[] = { 0x66, 0x41, 0x0f, 0x5c };
	static const uint8_t REX_XORPS[] = { 0x41, 0x0f, 0x57 };
	static const uint8_t REX_MULPD[] = { 0x66, 0x41, 0x0f, 0x59 };
	static const uint8_t REX_MAXPD[] = { 0x66, 0x41, 0x0f, 0x5f };
	static const uint8_t REX_DIVPD[] = { 0x66, 0x41, 0x0f, 0x5e };
	static const uint8_t SQRTPD[] = { 0x66, 0x0f, 0x51 };
	static const uint8_t AND_OR_MOV_LDMXCSR[] = { 0x25, 0x00, 0x60, 0x00, 0x00, 0x0D, 0xC0, 0x9F, 0x00, 0x00, 0x50, 0x0F, 0xAE, 0x14, 0x24, 0x58 };
	static const uint8_t ROL_RAX[] = { 0x48, 0xc1, 0xc0 };
	static const uint8_t XOR_ECX_ECX[] = { 0x33, 0xC9 };
	static const uint8_t REX_CMP_R32I[] = { 0x41, 0x81 };
	static const uint8_t REX_CMP_M32I[] = { 0x81, 0x3c, 0x06 };
	static const uint8_t MOVAPD[] = { 0x66, 0x0f, 0x29 };
	static const uint8_t REX_MOV_MR[] = { 0x4c, 0x89 };
	static const uint8_t REX_XOR_EAX[] = { 0x41, 0x33 };
	static const uint8_t SUB_EBX[] = { 0x83, 0xEB, 0x01 };
	static const uint8_t JNZ[] = { 0x0f, 0x85 };
	static const uint8_t JMP = 0xe9;
	static const uint8_t REX_XOR_RAX_R64[] = { 0x49, 0x33 };
	static const uint8_t REX_XCHG[] = { 0x4d, 0x87 };
	static const uint8_t REX_ANDPS_XMM12[] = { 0x45, 0x0F, 0x54, 0xE5, 0x45, 0x0F, 0x56, 0xE6 };
	static const uint8_t REX_PADD[] = { 0x66, 0x44, 0x0f };
	static const uint8_t PADD_OPCODES[] = { 0xfc, 0xfd, 0xfe, 0xd4 };
	static const uint8_t CALL = 0xe8;
	static const uint8_t REX_ADD_I[] = { 0x49, 0x81 };
	static const uint8_t REX_TEST[] = { 0x49, 0xF7 };
	static const uint8_t JZ[] = { 0x0f, 0x84 };
	static const uint8_t RET = 0xc3;
	static const uint8_t LEA_32[] = { 0x67, 0x41, 0x8d };
	static const uint8_t MOVNTI[] = { 0x4c, 0x0f, 0xc3 };
	static const uint8_t ADD_EBX_I[] = { 0x81, 0xc3 };

	static const uint8_t NOP1[] = { 0x90 };
	static const uint8_t NOP2[] = { 0x66, 0x90 };
	static const uint8_t NOP3[] = { 0x66, 0x66, 0x90 };
	static const uint8_t NOP4[] = { 0x0F, 0x1F, 0x40, 0x00 };
	static const uint8_t NOP5[] = { 0x0F, 0x1F, 0x44, 0x00, 0x00 };
	static const uint8_t NOP6[] = { 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00 };
	static const uint8_t NOP7[] = { 0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00 };
	static const uint8_t NOP8[] = { 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 };

//	static const uint8_t* NOPX[] = { NOP1, NOP2, NOP3, NOP4, NOP5, NOP6, NOP7, NOP8 };

	size_t JitCompilerX86::getCodeSize() {
		return codePos - prologueSize;
	}

	JitCompilerX86::JitCompilerX86() {
		code = (uint8_t*)allocExecutableMemory(CodeSize);
		memcpy(code, codePrologue, prologueSize);
		memcpy(code + epilogueOffset, codeEpilogue, epilogueSize);
	}

	JitCompilerX86::~JitCompilerX86() {
		freePagedMemory(code, CodeSize);
	}

	void JitCompilerX86::generateProgram(Program& prog, ProgramConfiguration& pcfg) {
		generateProgramPrologue(prog, pcfg);
		memcpy(code + codePos, RandomX_CurrentConfig.codeReadDatasetTweaked, readDatasetSize);
		codePos += readDatasetSize;
		generateProgramEpilogue(prog);
	}

	void JitCompilerX86::generateProgramLight(Program& prog, ProgramConfiguration& pcfg, uint32_t datasetOffset) {
		generateProgramPrologue(prog, pcfg);
		emit(RandomX_CurrentConfig.codeReadDatasetLightSshInitTweaked, readDatasetLightInitSize);
		emit(ADD_EBX_I);
		emit32(datasetOffset / CacheLineSize);
		emitByte(CALL);
		emit32(superScalarHashOffset - (codePos + 4));
		emit(codeReadDatasetLightSshFin, readDatasetLightFinSize);
		generateProgramEpilogue(prog);
	}

	template<size_t N>
	void JitCompilerX86::generateSuperscalarHash(SuperscalarProgram(&programs)[N], std::vector<uint64_t> &reciprocalCache) {
		memcpy(code + superScalarHashOffset, codeShhInit, codeSshInitSize);
		codePos = superScalarHashOffset + codeSshInitSize;
		for (unsigned j = 0; j < RandomX_CurrentConfig.CacheAccesses; ++j) {
			SuperscalarProgram& prog = programs[j];
			for (unsigned i = 0; i < prog.getSize(); ++i) {
				Instruction& instr = prog(i);
				generateSuperscalarCode(instr, reciprocalCache);
			}
			emit(codeShhLoad, codeSshLoadSize);
			if (j < RandomX_CurrentConfig.CacheAccesses - 1) {
				emit(REX_MOV_RR64);
				emitByte(0xd8 + prog.getAddressRegister());
				emit(RandomX_CurrentConfig.codeShhPrefetchTweaked, codeSshPrefetchSize);
#ifdef RANDOMX_ALIGN
				int align = (codePos % 16);
				while (align != 0) {
					int nopSize = 16 - align;
					if (nopSize > 8) nopSize = 8;
					emit(NOPX[nopSize - 1], nopSize);
					align = (codePos % 16);
				}
#endif
			}
		}
		emitByte(RET);
	}

	template
	void JitCompilerX86::generateSuperscalarHash(SuperscalarProgram(&programs)[RANDOMX_CACHE_MAX_ACCESSES], std::vector<uint64_t> &reciprocalCache);

	void JitCompilerX86::generateDatasetInitCode() {
		memcpy(code, codeDatasetInit, datasetInitSize);
	}

	void JitCompilerX86::generateProgramPrologue(Program& prog, ProgramConfiguration& pcfg) {
		instructionOffsets.clear();
		for (unsigned i = 0; i < 8; ++i) {
			registerUsage[i] = -1;
		}
		codePos = prologueSize;
		memcpy(code + codePos - 48, &pcfg.eMask, sizeof(pcfg.eMask));
		emit(REX_XOR_RAX_R64);
		emitByte(0xc0 + pcfg.readReg0);
		emit(REX_XOR_RAX_R64);
		emitByte(0xc0 + pcfg.readReg1);
		memcpy(code + codePos, RandomX_CurrentConfig.codeLoopLoadTweaked, loopLoadSize);
		codePos += loopLoadSize;
		for (unsigned i = 0; i < prog.getSize(); ++i) {
			Instruction& instr = prog(i);
			instr.src %= RegistersCount;
			instr.dst %= RegistersCount;
			generateCode(instr, i);
		}
		emit(REX_MOV_RR);
		emitByte(0xc0 + pcfg.readReg2);
		emit(REX_XOR_EAX);
		emitByte(0xc0 + pcfg.readReg3);
	}

	void JitCompilerX86::generateProgramEpilogue(Program& prog) {
		memcpy(code + codePos, codeLoopStore, loopStoreSize);
		codePos += loopStoreSize;
		emit(SUB_EBX);
		emit(JNZ);
		emit32(prologueSize - codePos - 4);
		emitByte(JMP);
		emit32(epilogueOffset - codePos - 4);
	}

	void JitCompilerX86::generateCode(Instruction& instr, int i) {
		instructionOffsets.push_back(codePos);
		auto generator = engine[instr.opcode];
		(this->*generator)(instr, i);
	}

	void JitCompilerX86::generateSuperscalarCode(Instruction& instr, std::vector<uint64_t> &reciprocalCache) {
		switch ((SuperscalarInstructionType)instr.opcode)
		{
		case randomx::SuperscalarInstructionType::ISUB_R:
			emit(REX_SUB_RR);
			emitByte(0xc0 + 8 * instr.dst + instr.src);
			break;
		case randomx::SuperscalarInstructionType::IXOR_R:
			emit(REX_XOR_RR);
			emitByte(0xc0 + 8 * instr.dst + instr.src);
			break;
		case randomx::SuperscalarInstructionType::IADD_RS:
			emit(REX_LEA);
			emitByte(0x04 + 8 * instr.dst);
			genSIB(instr.getModShift(), instr.src, instr.dst);
			break;
		case randomx::SuperscalarInstructionType::IMUL_R:
			emit(REX_IMUL_RR);
			emitByte(0xc0 + 8 * instr.dst + instr.src);
			break;
		case randomx::SuperscalarInstructionType::IROR_C:
			emit(REX_ROT_I8);
			emitByte(0xc8 + instr.dst);
			emitByte(instr.getImm32() & 63);
			break;
		case randomx::SuperscalarInstructionType::IADD_C7:
			emit(REX_81);
			emitByte(0xc0 + instr.dst);
			emit32(instr.getImm32());
			break;
		case randomx::SuperscalarInstructionType::IXOR_C7:
			emit(REX_XOR_RI);
			emitByte(0xf0 + instr.dst);
			emit32(instr.getImm32());
			break;
		case randomx::SuperscalarInstructionType::IADD_C8:
			emit(REX_81);
			emitByte(0xc0 + instr.dst);
			emit32(instr.getImm32());
#ifdef RANDOMX_ALIGN
			emit(NOP1);
#endif
			break;
		case randomx::SuperscalarInstructionType::IXOR_C8:
			emit(REX_XOR_RI);
			emitByte(0xf0 + instr.dst);
			emit32(instr.getImm32());
#ifdef RANDOMX_ALIGN
			emit(NOP1);
#endif
			break;
		case randomx::SuperscalarInstructionType::IADD_C9:
			emit(REX_81);
			emitByte(0xc0 + instr.dst);
			emit32(instr.getImm32());
#ifdef RANDOMX_ALIGN
			emit(NOP2);
#endif
			break;
		case randomx::SuperscalarInstructionType::IXOR_C9:
			emit(REX_XOR_RI);
			emitByte(0xf0 + instr.dst);
			emit32(instr.getImm32());
#ifdef RANDOMX_ALIGN
			emit(NOP2);
#endif
			break;
		case randomx::SuperscalarInstructionType::IMULH_R:
			emit(REX_MOV_RR64);
			emitByte(0xc0 + instr.dst);
			emit(REX_MUL_R);
			emitByte(0xe0 + instr.src);
			emit(REX_MOV_R64R);
			emitByte(0xc2 + 8 * instr.dst);
			break;
		case randomx::SuperscalarInstructionType::ISMULH_R:
			emit(REX_MOV_RR64);
			emitByte(0xc0 + instr.dst);
			emit(REX_MUL_R);
			emitByte(0xe8 + instr.src);
			emit(REX_MOV_R64R);
			emitByte(0xc2 + 8 * instr.dst);
			break;
		case randomx::SuperscalarInstructionType::IMUL_RCP:
			emit(MOV_RAX_I);
			emit64(reciprocalCache[instr.getImm32()]);
			emit(REX_IMUL_RM);
			emitByte(0xc0 + 8 * instr.dst);
			break;
		default:
			UNREACHABLE;
		}
	}

	void JitCompilerX86::genAddressReg(Instruction& instr, bool rax = true) {
		emit(LEA_32);
		emitByte(0x80 + instr.src + (rax ? 0 : 8));
		if (instr.src == RegisterNeedsSib) {
			emitByte(0x24);
		}
		emit32(instr.getImm32());
		if (rax)
			emitByte(AND_EAX_I);
		else
			emit(AND_ECX_I);
		emit32(instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
	}

	void JitCompilerX86::genAddressRegDst(Instruction& instr) {
		emit(LEA_32);
		emitByte(0x80 + instr.dst);
		if (instr.dst == RegisterNeedsSib) {
			emitByte(0x24);
		}
		emit32(instr.getImm32());
		emitByte(AND_EAX_I);
		if (instr.getModCond() < StoreL3Condition) {
			emit32(instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
		}
		else {
			emit32(ScratchpadL3Mask);
		}
	}

	void JitCompilerX86::genAddressImm(Instruction& instr) {
		emit32(instr.getImm32() & ScratchpadL3Mask);
	}

	void JitCompilerX86::h_IADD_RS(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		emit(REX_LEA);
		if (instr.dst == RegisterNeedsDisplacement)
			emitByte(0xac);
		else
			emitByte(0x04 + 8 * instr.dst);
		genSIB(instr.getModShift(), instr.src, instr.dst);
		if (instr.dst == RegisterNeedsDisplacement)
			emit32(instr.getImm32());
	}

	void JitCompilerX86::h_IADD_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			emit(REX_ADD_RM);
			emitByte(0x04 + 8 * instr.dst);
			emitByte(0x06);
		}
		else {
			emit(REX_ADD_RM);
			emitByte(0x86 + 8 * instr.dst);
			genAddressImm(instr);
		}
	}

	void JitCompilerX86::genSIB(int scale, int index, int base) {
		emitByte((scale << 6) | (index << 3) | base);
	}

	void JitCompilerX86::h_ISUB_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			emit(REX_SUB_RR);
			emitByte(0xc0 + 8 * instr.dst + instr.src);
		}
		else {
			emit(REX_81);
			emitByte(0xe8 + instr.dst);
			emit32(instr.getImm32());
		}
	}

	void JitCompilerX86::h_ISUB_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			emit(REX_SUB_RM);
			emitByte(0x04 + 8 * instr.dst);
			emitByte(0x06);
		}
		else {
			emit(REX_SUB_RM);
			emitByte(0x86 + 8 * instr.dst);
			genAddressImm(instr);
		}
	}

	void JitCompilerX86::h_IMUL_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			emit(REX_IMUL_RR);
			emitByte(0xc0 + 8 * instr.dst + instr.src);
		}
		else {
			emit(REX_IMUL_RRI);
			emitByte(0xc0 + 9 * instr.dst);
			emit32(instr.getImm32());
		}
	}

	void JitCompilerX86::h_IMUL_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			emit(REX_IMUL_RM);
			emitByte(0x04 + 8 * instr.dst);
			emitByte(0x06);
		}
		else {
			emit(REX_IMUL_RM);
			emitByte(0x86 + 8 * instr.dst);
			genAddressImm(instr);
		}
	}

	void JitCompilerX86::h_IMULH_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		emit(REX_MOV_RR64);
		emitByte(0xc0 + instr.dst);
		emit(REX_MUL_R);
		emitByte(0xe0 + instr.src);
		emit(REX_MOV_R64R);
		emitByte(0xc2 + 8 * instr.dst);
	}

	void JitCompilerX86::h_IMULH_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr, false);
			emit(REX_MOV_RR64);
			emitByte(0xc0 + instr.dst);
			emit(REX_MUL_MEM);
		}
		else {
			emit(REX_MOV_RR64);
			emitByte(0xc0 + instr.dst);
			emit(REX_MUL_M);
			emitByte(0xa6);
			genAddressImm(instr);
		}
		emit(REX_MOV_R64R);
		emitByte(0xc2 + 8 * instr.dst);
	}

	void JitCompilerX86::h_ISMULH_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		emit(REX_MOV_RR64);
		emitByte(0xc0 + instr.dst);
		emit(REX_MUL_R);
		emitByte(0xe8 + instr.src);
		emit(REX_MOV_R64R);
		emitByte(0xc2 + 8 * instr.dst);
	}

	void JitCompilerX86::h_ISMULH_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr, false);
			emit(REX_MOV_RR64);
			emitByte(0xc0 + instr.dst);
			emit(REX_IMUL_MEM);
		}
		else {
			emit(REX_MOV_RR64);
			emitByte(0xc0 + instr.dst);
			emit(REX_MUL_M);
			emitByte(0xae);
			genAddressImm(instr);
		}
		emit(REX_MOV_R64R);
		emitByte(0xc2 + 8 * instr.dst);
	}

	void JitCompilerX86::h_IMUL_RCP(Instruction& instr, int i) {
		uint64_t divisor = instr.getImm32();
		if (!isPowerOf2(divisor)) {
			registerUsage[instr.dst] = i;
			emit(MOV_RAX_I);
			emit64(randomx_reciprocal_fast(divisor));
			emit(REX_IMUL_RM);
			emitByte(0xc0 + 8 * instr.dst);
		}
	}

	void JitCompilerX86::h_INEG_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		emit(REX_NEG);
		emitByte(0xd8 + instr.dst);
	}

	void JitCompilerX86::h_IXOR_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			emit(REX_XOR_RR);
			emitByte(0xc0 + 8 * instr.dst + instr.src);
		}
		else {
			emit(REX_XOR_RI);
			emitByte(0xf0 + instr.dst);
			emit32(instr.getImm32());
		}
	}

	void JitCompilerX86::h_IXOR_M(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			genAddressReg(instr);
			emit(REX_XOR_RM);
			emitByte(0x04 + 8 * instr.dst);
			emitByte(0x06);
		}
		else {
			emit(REX_XOR_RM);
			emitByte(0x86 + 8 * instr.dst);
			genAddressImm(instr);
		}
	}

	void JitCompilerX86::h_IROR_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			emit(REX_MOV_RR);
			emitByte(0xc8 + instr.src);
			emit(REX_ROT_CL);
			emitByte(0xc8 + instr.dst);
		}
		else {
			emit(REX_ROT_I8);
			emitByte(0xc8 + instr.dst);
			emitByte(instr.getImm32() & 63);
		}
	}

	void JitCompilerX86::h_IROL_R(Instruction& instr, int i) {
		registerUsage[instr.dst] = i;
		if (instr.src != instr.dst) {
			emit(REX_MOV_RR);
			emitByte(0xc8 + instr.src);
			emit(REX_ROT_CL);
			emitByte(0xc0 + instr.dst);
		}
		else {
			emit(REX_ROT_I8);
			emitByte(0xc0 + instr.dst);
			emitByte(instr.getImm32() & 63);
		}
	}

	void JitCompilerX86::h_ISWAP_R(Instruction& instr, int i) {
		if (instr.src != instr.dst) {
			registerUsage[instr.dst] = i;
			registerUsage[instr.src] = i;
			emit(REX_XCHG);
			emitByte(0xc0 + instr.src + 8 * instr.dst);
		}
	}

	void JitCompilerX86::h_FSWAP_R(Instruction& instr, int i) {
		emit(SHUFPD);
		emitByte(0xc0 + 9 * instr.dst);
		emitByte(1);
	}

	void JitCompilerX86::h_FADD_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		instr.src %= RegisterCountFlt;
		emit(REX_ADDPD);
		emitByte(0xc0 + instr.src + 8 * instr.dst);
	}

	void JitCompilerX86::h_FADD_M(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		genAddressReg(instr);
		emit(REX_CVTDQ2PD_XMM12);
		emit(REX_ADDPD);
		emitByte(0xc4 + 8 * instr.dst);
	}

	void JitCompilerX86::h_FSUB_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		instr.src %= RegisterCountFlt;
		emit(REX_SUBPD);
		emitByte(0xc0 + instr.src + 8 * instr.dst);
	}

	void JitCompilerX86::h_FSUB_M(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		genAddressReg(instr);
		emit(REX_CVTDQ2PD_XMM12);
		emit(REX_SUBPD);
		emitByte(0xc4 + 8 * instr.dst);
	}

	void JitCompilerX86::h_FSCAL_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		emit(REX_XORPS);
		emitByte(0xc7 + 8 * instr.dst);
	}

	void JitCompilerX86::h_FMUL_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		instr.src %= RegisterCountFlt;
		emit(REX_MULPD);
		emitByte(0xe0 + instr.src + 8 * instr.dst);
	}

	void JitCompilerX86::h_FDIV_M(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		genAddressReg(instr);
		emit(REX_CVTDQ2PD_XMM12);
		emit(REX_ANDPS_XMM12);
		emit(REX_DIVPD);
		emitByte(0xe4 + 8 * instr.dst);
	}

	void JitCompilerX86::h_FSQRT_R(Instruction& instr, int i) {
		instr.dst %= RegisterCountFlt;
		emit(SQRTPD);
		emitByte(0xe4 + 9 * instr.dst);
	}

	void JitCompilerX86::h_CFROUND(Instruction& instr, int i) {
		emit(REX_MOV_RR64);
		emitByte(0xc0 + instr.src);
		int rotate = (13 - (instr.getImm32() & 63)) & 63;
		if (rotate != 0) {
			emit(ROL_RAX);
			emitByte(rotate);
		}
		emit(AND_OR_MOV_LDMXCSR);
	}

	void JitCompilerX86::h_CBRANCH(Instruction& instr, int i) {
		int reg = instr.dst;
		int target = registerUsage[reg] + 1;
		emit(REX_ADD_I);
		emitByte(0xc0 + reg);
		int shift = instr.getModCond() + RandomX_CurrentConfig.JumpOffset;
		uint32_t imm = instr.getImm32() | (1UL << shift);
		if (RandomX_CurrentConfig.JumpOffset > 0 || shift > 0)
			imm &= ~(1UL << (shift - 1));
		emit32(imm);
		emit(REX_TEST);
		emitByte(0xc0 + reg);
		emit32(RandomX_CurrentConfig.ConditionMask_Calculated << shift);
		emit(JZ);
		emit32(instructionOffsets[target] - (codePos + 4));
		//mark all registers as used
		for (unsigned j = 0; j < RegistersCount; ++j) {
			registerUsage[j] = i;
		}
	}

	void JitCompilerX86::h_ISTORE(Instruction& instr, int i) {
		genAddressRegDst(instr);
		emit(REX_MOV_MR);
		emitByte(0x04 + 8 * instr.src);
		emitByte(0x06);
	}

	void JitCompilerX86::h_NOP(Instruction& instr, int i) {
		emit(NOP1);
	}

	InstructionGeneratorX86 JitCompilerX86::engine[256] = {};

}
