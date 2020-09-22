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
#include <atomic>
#include "crypto/randomx/jit_compiler_x86.hpp"
#include "crypto/randomx/jit_compiler_x86_static.hpp"
#include "crypto/randomx/superscalar.hpp"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/reciprocal.h"
#include "crypto/randomx/virtual_memory.hpp"
#include "base/tools/Profiler.h"
#include "backend/cpu/Cpu.h"

#ifdef XMRIG_FIX_RYZEN
#   include "crypto/rx/Rx.h"
#endif

#ifdef _MSC_VER
#   include <intrin.h>
#else
#   include <cpuid.h>
#endif

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

	#define codePrefetchScratchpad ((uint8_t*)&randomx_prefetch_scratchpad)
	#define codePrefetchScratchpadEnd ((uint8_t*)&randomx_prefetch_scratchpad_end)
	#define codePrologue ((uint8_t*)&randomx_program_prologue)
	#define codeLoopBegin ((uint8_t*)&randomx_program_loop_begin)
	#define codeLoopLoad ((uint8_t*)&randomx_program_loop_load)
	#define codeLoopLoadXOP ((uint8_t*)&randomx_program_loop_load_xop)
	#define codeProgamStart ((uint8_t*)&randomx_program_start)
	#define codeReadDatasetLightSshInit ((uint8_t*)&randomx_program_read_dataset_sshash_init)
	#define codeReadDatasetLightSshFin ((uint8_t*)&randomx_program_read_dataset_sshash_fin)
	#define codeDatasetInit ((uint8_t*)&randomx_dataset_init)
	#define codeLoopStore ((uint8_t*)&randomx_program_loop_store)
	#define codeLoopEnd ((uint8_t*)&randomx_program_loop_end)
	#define codeEpilogue ((uint8_t*)&randomx_program_epilogue)
	#define codeProgramEnd ((uint8_t*)&randomx_program_end)
	#define codeShhLoad ((uint8_t*)&randomx_sshash_load)
	#define codeShhPrefetch ((uint8_t*)&randomx_sshash_prefetch)
	#define codeShhEnd ((uint8_t*)&randomx_sshash_end)
	#define codeShhInit ((uint8_t*)&randomx_sshash_init)

	#define prefetchScratchpadSize (codePrefetchScratchpadEnd - codePrefetchScratchpad)
	#define prologueSize (codeLoopBegin - codePrologue)
	#define loopLoadSize (codeLoopLoadXOP - codeLoopLoad)
	#define loopLoadXOPSize (codeProgamStart - codeLoopLoadXOP)
	#define readDatasetLightInitSize (codeReadDatasetLightSshFin - codeReadDatasetLightSshInit)
	#define readDatasetLightFinSize (codeLoopStore - codeReadDatasetLightSshFin)
	#define loopStoreSize (codeLoopEnd - codeLoopStore)
	#define datasetInitSize (codeEpilogue - codeDatasetInit)
	#define epilogueSize (codeShhLoad - codeEpilogue)
	#define codeSshLoadSize (codeShhPrefetch - codeShhLoad)
	#define codeSshPrefetchSize (codeShhEnd - codeShhPrefetch)
	#define codeSshInitSize (codeProgramEnd - codeShhInit)

	#define epilogueOffset ((CodeSize - epilogueSize) & ~63)

	constexpr int32_t superScalarHashOffset = 32768;

	static const uint8_t NOP1[] = { 0x90 };
	static const uint8_t NOP2[] = { 0x66, 0x90 };
	static const uint8_t NOP3[] = { 0x66, 0x66, 0x90 };
	static const uint8_t NOP4[] = { 0x0F, 0x1F, 0x40, 0x00 };
	static const uint8_t NOP5[] = { 0x0F, 0x1F, 0x44, 0x00, 0x00 };
	static const uint8_t NOP6[] = { 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00 };
	static const uint8_t NOP7[] = { 0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00 };
	static const uint8_t NOP8[] = { 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 };

	static const uint8_t* NOPX[] = { NOP1, NOP2, NOP3, NOP4, NOP5, NOP6, NOP7, NOP8 };

	static const uint8_t JMP_ALIGN_PREFIX[14][16] = {
		{},
		{0x2E},
		{0x2E, 0x2E},
		{0x2E, 0x2E, 0x2E},
		{0x2E, 0x2E, 0x2E, 0x2E},
		{0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x90, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x66, 0x90, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x66, 0x66, 0x90, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x0F, 0x1F, 0x40, 0x00, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
		{0x0F, 0x1F, 0x44, 0x00, 0x00, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E},
	};

	size_t JitCompilerX86::getCodeSize() {
		return codePos < prologueSize ? 0 : codePos - prologueSize;
	}

    static inline void cpuid(uint32_t level, int32_t output[4])
    {
        memset(output, 0, sizeof(int32_t) * 4);

#   ifdef _MSC_VER
        __cpuid(output, static_cast<int>(level));
#   else
        __cpuid_count(level, 0, output[0], output[1], output[2], output[3]);
#   endif
    }

#	ifdef _MSC_VER
	static FORCE_INLINE uint32_t rotl32(uint32_t a, int shift) { return _rotl(a, shift); }
#	else
	static FORCE_INLINE uint32_t rotl32(uint32_t a, int shift) { return (a << shift) | (a >> (-shift & 31)); }
#	endif

	static std::atomic<size_t> codeOffset;

	JitCompilerX86::JitCompilerX86() {
		BranchesWithin32B = xmrig::Cpu::info()->jccErratum();

		int32_t info[4];
		cpuid(1, info);
		hasAVX = ((info[2] & (1 << 27)) != 0) && ((info[2] & (1 << 28)) != 0);

		cpuid(0x80000001, info);
		hasXOP = ((info[2] & (1 << 11)) != 0);

		allocatedCode = (uint8_t*)allocExecutableMemory(CodeSize * 2);
		// Shift code base address to improve caching - all threads will use different L2/L3 cache sets
		code = allocatedCode + (codeOffset.fetch_add(59 * 64) % CodeSize);
		memcpy(code, codePrologue, prologueSize);
		if (hasXOP) {
			memcpy(code + prologueSize, codeLoopLoadXOP, loopLoadXOPSize);
		}
		else {
			memcpy(code + prologueSize, codeLoopLoad, loopLoadSize);
		}
		memcpy(code + epilogueOffset, codeEpilogue, epilogueSize);

		codePosFirst = prologueSize + (hasXOP ? loopLoadXOPSize : loopLoadSize);

#		ifdef XMRIG_FIX_RYZEN
		mainLoopBounds.first = code + prologueSize;
		mainLoopBounds.second = code + epilogueOffset;
#		endif
	}

	JitCompilerX86::~JitCompilerX86() {
		freePagedMemory(allocatedCode, CodeSize);
	}

	void JitCompilerX86::prepare() {
		for (size_t i = 0; i < sizeof(engine); i += 64)
			rx_prefetch_nta((const char*)(&engine) + i);
		for (size_t i = 0; i < sizeof(RandomX_CurrentConfig); i += 64)
			rx_prefetch_nta((const char*)(&RandomX_CurrentConfig) + i);
	}

	void JitCompilerX86::generateProgram(Program& prog, ProgramConfiguration& pcfg, uint32_t flags) {
		PROFILE_SCOPE(RandomX_JIT_compile);

		vm_flags = flags;

		generateProgramPrologue(prog, pcfg);

		uint8_t* p;
		uint32_t n;
		if (flags & RANDOMX_FLAG_AMD) {
			p = RandomX_CurrentConfig.codeReadDatasetRyzenTweaked;
			n = RandomX_CurrentConfig.codeReadDatasetRyzenTweakedSize;
		}
		else {
			p = RandomX_CurrentConfig.codeReadDatasetTweaked;
			n = RandomX_CurrentConfig.codeReadDatasetTweakedSize;
		}
		memcpy(code + codePos, p, n);
		codePos += n;

		generateProgramEpilogue(prog, pcfg);
	}

	void JitCompilerX86::generateProgramLight(Program& prog, ProgramConfiguration& pcfg, uint32_t datasetOffset) {
		generateProgramPrologue(prog, pcfg);
		emit(RandomX_CurrentConfig.codeReadDatasetLightSshInitTweaked, readDatasetLightInitSize, code, codePos);
		*(uint32_t*)(code + codePos) = 0xc381;
		codePos += 2;
		emit32(datasetOffset / CacheLineSize, code, codePos);
		emitByte(0xe8, code, codePos);
		emit32(superScalarHashOffset - (codePos + 4), code, codePos);
		emit(codeReadDatasetLightSshFin, readDatasetLightFinSize, code, codePos);
		generateProgramEpilogue(prog, pcfg);
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
			emit(codeShhLoad, codeSshLoadSize, code, codePos);
			if (j < RandomX_CurrentConfig.CacheAccesses - 1) {
				*(uint32_t*)(code + codePos) = 0xd88b49 + (static_cast<uint32_t>(prog.getAddressRegister()) << 16);
				codePos += 3;
				emit(RandomX_CurrentConfig.codeShhPrefetchTweaked, codeSshPrefetchSize, code, codePos);
			}
		}
		emitByte(0xc3, code, codePos);
	}

	template
	void JitCompilerX86::generateSuperscalarHash(SuperscalarProgram(&programs)[RANDOMX_CACHE_MAX_ACCESSES], std::vector<uint64_t> &reciprocalCache);

	void JitCompilerX86::generateDatasetInitCode() {
		memcpy(code, codeDatasetInit, datasetInitSize);
	}

	void JitCompilerX86::generateProgramPrologue(Program& prog, ProgramConfiguration& pcfg) {
		codePos = ((uint8_t*)randomx_program_prologue_first_load) - ((uint8_t*)randomx_program_prologue);
		code[codePos + 2] = 0xc0 + pcfg.readReg0;
		code[codePos + 5] = 0xc0 + pcfg.readReg1;
		*(uint32_t*)(code + codePos + 10) = RandomX_CurrentConfig.ScratchpadL3Mask64_Calculated;
		*(uint32_t*)(code + codePos + 20) = RandomX_CurrentConfig.ScratchpadL3Mask64_Calculated;
		if (hasAVX) {
			uint32_t* p = (uint32_t*)(code + codePos + 67);
			*p = (*p & 0xFF000000U) | 0x0077F8C5U;
		}

#		ifdef XMRIG_FIX_RYZEN
		xmrig::Rx::setMainLoopBounds(mainLoopBounds);
#		endif

		memcpy(code + prologueSize - 48, &pcfg.eMask, sizeof(pcfg.eMask));
		codePos = codePosFirst;

		//mark all registers as used
		uint64_t* r = (uint64_t*)registerUsage;
		uint64_t k = codePos;
		k |= k << 32;
		for (unsigned j = 0; j < RegistersCount / 2; ++j) {
			r[j] = k;
		}

		for (int i = 0, n = static_cast<int>(RandomX_CurrentConfig.ProgramSize); i < n; i += 4) {
			Instruction& instr1 = prog(i);
			Instruction& instr2 = prog(i + 1);
			Instruction& instr3 = prog(i + 2);
			Instruction& instr4 = prog(i + 3);

			InstructionGeneratorX86 gen1 = engine[instr1.opcode];
			InstructionGeneratorX86 gen2 = engine[instr2.opcode];
			InstructionGeneratorX86 gen3 = engine[instr3.opcode];
			InstructionGeneratorX86 gen4 = engine[instr4.opcode];

			(*gen1)(this, instr1);
			(*gen2)(this, instr2);
			(*gen3)(this, instr3);
			(*gen4)(this, instr4);
		}

		*(uint64_t*)(code + codePos) = 0xc03341c08b41ull + (static_cast<uint64_t>(pcfg.readReg2) << 16) + (static_cast<uint64_t>(pcfg.readReg3) << 40);
		codePos += 6;
	}

	void JitCompilerX86::generateProgramEpilogue(Program& prog, ProgramConfiguration& pcfg) {
		*(uint64_t*)(code + codePos) = 0xc03349c08b49ull + (static_cast<uint64_t>(pcfg.readReg0) << 16) + (static_cast<uint64_t>(pcfg.readReg1) << 40);
		codePos += 6;
		emit(RandomX_CurrentConfig.codePrefetchScratchpadTweaked, prefetchScratchpadSize, code, codePos);
		memcpy(code + codePos, codeLoopStore, loopStoreSize);
		codePos += loopStoreSize;

		if (BranchesWithin32B) {
			const uint32_t branch_begin = static_cast<uint32_t>(codePos);
			const uint32_t branch_end = static_cast<uint32_t>(branch_begin + 9);

			// If the jump crosses or touches 32-byte boundary, align it
			if ((branch_begin ^ branch_end) >= 32) {
				uint32_t alignment_size = 32 - (branch_begin & 31);
				if (alignment_size > 8) {
					emit(NOPX[alignment_size - 9], alignment_size - 8, code, codePos);
					alignment_size = 8;
				}
				emit(NOPX[alignment_size - 1], alignment_size, code, codePos);
			}
		}

		*(uint64_t*)(code + codePos) = 0x850f01eb83ull;
		codePos += 5;
		emit32(prologueSize - codePos - 4, code, codePos);
		emitByte(0xe9, code, codePos);
		emit32(epilogueOffset - codePos - 4, code, codePos);
	}

	void JitCompilerX86::generateSuperscalarCode(Instruction& instr, std::vector<uint64_t> &reciprocalCache) {
		static constexpr uint8_t REX_SUB_RR[] = { 0x4d, 0x2b };
		static constexpr uint8_t REX_MOV_RR64[] = { 0x49, 0x8b };
		static constexpr uint8_t REX_MOV_R64R[] = { 0x4c, 0x8b };
		static constexpr uint8_t REX_IMUL_RR[] = { 0x4d, 0x0f, 0xaf };
		static constexpr uint8_t REX_IMUL_RM[] = { 0x4c, 0x0f, 0xaf };
		static constexpr uint8_t REX_MUL_R[] = { 0x49, 0xf7 };
		static constexpr uint8_t REX_81[] = { 0x49, 0x81 };
		static constexpr uint8_t MOV_RAX_I[] = { 0x48, 0xb8 };
		static constexpr uint8_t REX_LEA[] = { 0x4f, 0x8d };
		static constexpr uint8_t REX_XOR_RR[] = { 0x4D, 0x33 };
		static constexpr uint8_t REX_XOR_RI[] = { 0x49, 0x81 };
		static constexpr uint8_t REX_ROT_I8[] = { 0x49, 0xc1 };

		switch ((SuperscalarInstructionType)instr.opcode)
		{
		case randomx::SuperscalarInstructionType::ISUB_R:
			emit(REX_SUB_RR, code, codePos);
			emitByte(0xc0 + 8 * instr.dst + instr.src, code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IXOR_R:
			emit(REX_XOR_RR, code, codePos);
			emitByte(0xc0 + 8 * instr.dst + instr.src, code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IADD_RS:
			emit(REX_LEA, code, codePos);
			emitByte(0x04 + 8 * instr.dst, code, codePos);
			genSIB(instr.getModShift(), instr.src, instr.dst, code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IMUL_R:
			emit(REX_IMUL_RR, code, codePos);
			emitByte(0xc0 + 8 * instr.dst + instr.src, code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IROR_C:
			emit(REX_ROT_I8, code, codePos);
			emitByte(0xc8 + instr.dst, code, codePos);
			emitByte(instr.getImm32() & 63, code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IADD_C7:
			emit(REX_81, code, codePos);
			emitByte(0xc0 + instr.dst, code, codePos);
			emit32(instr.getImm32(), code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IXOR_C7:
			emit(REX_XOR_RI, code, codePos);
			emitByte(0xf0 + instr.dst, code, codePos);
			emit32(instr.getImm32(), code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IADD_C8:
			emit(REX_81, code, codePos);
			emitByte(0xc0 + instr.dst, code, codePos);
			emit32(instr.getImm32(), code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IXOR_C8:
			emit(REX_XOR_RI, code, codePos);
			emitByte(0xf0 + instr.dst, code, codePos);
			emit32(instr.getImm32(), code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IADD_C9:
			emit(REX_81, code, codePos);
			emitByte(0xc0 + instr.dst, code, codePos);
			emit32(instr.getImm32(), code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IXOR_C9:
			emit(REX_XOR_RI, code, codePos);
			emitByte(0xf0 + instr.dst, code, codePos);
			emit32(instr.getImm32(), code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IMULH_R:
			emit(REX_MOV_RR64, code, codePos);
			emitByte(0xc0 + instr.dst, code, codePos);
			emit(REX_MUL_R, code, codePos);
			emitByte(0xe0 + instr.src, code, codePos);
			emit(REX_MOV_R64R, code, codePos);
			emitByte(0xc2 + 8 * instr.dst, code, codePos);
			break;
		case randomx::SuperscalarInstructionType::ISMULH_R:
			emit(REX_MOV_RR64, code, codePos);
			emitByte(0xc0 + instr.dst, code, codePos);
			emit(REX_MUL_R, code, codePos);
			emitByte(0xe8 + instr.src, code, codePos);
			emit(REX_MOV_R64R, code, codePos);
			emitByte(0xc2 + 8 * instr.dst, code, codePos);
			break;
		case randomx::SuperscalarInstructionType::IMUL_RCP:
			emit(MOV_RAX_I, code, codePos);
			emit64(reciprocalCache[instr.getImm32()], code, codePos);
			emit(REX_IMUL_RM, code, codePos);
			emitByte(0xc0 + 8 * instr.dst, code, codePos);
			break;
		default:
			UNREACHABLE;
		}
	}

	template<bool rax>
	FORCE_INLINE void JitCompilerX86::genAddressReg(const Instruction& instr, const uint32_t src, uint8_t* code, uint32_t& codePos) {
		*(uint32_t*)(code + codePos) = (rax ? 0x24808d41 : 0x24888d41) + (src << 16);

		constexpr uint32_t add_table = 0x33333333u + (1u << (RegisterNeedsSib * 4));
		codePos += (add_table >> (src * 4)) & 0xf;

		emit32(instr.getImm32(), code, codePos);
		if (rax) {
			emitByte(0x25, code, codePos);
		}
		else {
			*(uint32_t*)(code + codePos) = 0xe181;
			codePos += 2;
		}
		emit32(AddressMask[instr.getModMem()], code, codePos);
	}

	template void JitCompilerX86::genAddressReg<false>(const Instruction& instr, const uint32_t src, uint8_t* code, uint32_t& codePos);
	template void JitCompilerX86::genAddressReg<true>(const Instruction& instr, const uint32_t src, uint8_t* code, uint32_t& codePos);

	FORCE_INLINE void JitCompilerX86::genAddressRegDst(const Instruction& instr, uint8_t* code, uint32_t& codePos) {
		const uint32_t dst = static_cast<uint32_t>(instr.dst % RegistersCount) << 16;
		*(uint32_t*)(code + codePos) = 0x24808d41 + dst;
		codePos += (dst == (RegisterNeedsSib << 16)) ? 4 : 3;

		emit32(instr.getImm32(), code, codePos);
		emitByte(0x25, code, codePos);
		if (instr.getModCond() < StoreL3Condition) {
			emit32(AddressMask[instr.getModMem()], code, codePos);
		}
		else {
			emit32(ScratchpadL3Mask, code, codePos);
		}
	}

	FORCE_INLINE void JitCompilerX86::genAddressImm(const Instruction& instr, uint8_t* code, uint32_t& codePos) {
		emit32(instr.getImm32() & ScratchpadL3Mask, code, codePos);
	}

	void JitCompilerX86::h_IADD_RS(const Instruction& instr) {
		uint32_t pos = codePos;
		uint8_t* const p = code + pos;

		const uint32_t dst = instr.dst % RegistersCount;
		const uint32_t sib = (instr.getModShift() << 6) | ((instr.src % RegistersCount) << 3) | dst;

		uint32_t k = 0x048d4f + (dst << 19);
		if (dst == RegisterNeedsDisplacement)
			k = 0xac8d4f;

		*(uint32_t*)(p) = k | (sib << 24);
		*(uint32_t*)(p + 4) = instr.getImm32();

		pos += ((dst == RegisterNeedsDisplacement) ? 8 : 4);

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IADD_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			genAddressReg<true>(instr, src, p, pos);
			emit32(0x0604034c + (dst << 19), p, pos);
		}
		else {
			*(uint32_t*)(p + pos) = 0x86034c + (dst << 19);
			pos += 3;
			genAddressImm(instr, p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::genSIB(int scale, int index, int base, uint8_t* code, uint32_t& codePos) {
		emitByte((scale << 6) | (index << 3) | base, code, codePos);
	}

	void JitCompilerX86::h_ISUB_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;
		
		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			*(uint32_t*)(p + pos) = 0xc02b4d + (dst << 19) + (src << 16);
			pos += 3;
		}
		else {
			*(uint32_t*)(p + pos) = 0xe88149 + (dst << 16);
			pos += 3;
			emit32(instr.getImm32(), p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_ISUB_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			genAddressReg<true>(instr, src, p, pos);
			emit32(0x06042b4c + (dst << 19), p, pos);
		}
		else {
			*(uint32_t*)(p + pos) = 0x862b4c + (dst << 19);
			pos += 3;
			genAddressImm(instr, p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IMUL_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			emit32(0xc0af0f4d + ((dst * 8 + src) << 24), p, pos);
		}
		else {
			*(uint32_t*)(p + pos) = 0xc0694d + (((dst << 3) + dst) << 16);
			pos += 3;
			emit32(instr.getImm32(), p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IMUL_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			genAddressReg<true>(instr, src, p, pos);
			*(uint64_t*)(p + pos) = 0x0604af0f4cull + (dst << 27);
			pos += 5;
		}
		else {
			emit32(0x86af0f4c + (dst << 27), p, pos);
			genAddressImm(instr, p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IMULH_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegistersCount;

		*(uint32_t*)(p + pos) = 0xc08b49 + (dst << 16);
		*(uint32_t*)(p + pos + 3) = 0xe0f749 + (src << 16);
		*(uint32_t*)(p + pos + 6) = 0xc28b4c + (dst << 19);
		pos += 9;

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IMULH_R_BMI2(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegistersCount;

		*(uint32_t*)(p + pos) = 0xC4D08B49 + (dst << 16);
		*(uint32_t*)(p + pos + 4) = 0xC0F6FB42 + (dst << 27) + (src << 24);
		pos += 8;

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IMULH_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			genAddressReg<false>(instr, src, p, pos);
			*(uint64_t*)(p + pos) = 0x0e24f748c08b49ull + (dst << 16);
			pos += 7;
		}
		else {
			*(uint64_t*)(p + pos) = 0xa6f748c08b49ull + (dst << 16);
			pos += 6;
			genAddressImm(instr, p, pos);
		}
		*(uint32_t*)(p + pos) = 0xc28b4c + (dst << 19);
		pos += 3;

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IMULH_M_BMI2(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			genAddressReg<false>(instr, src, p, pos);
			*(uint32_t*)(p + pos) = static_cast<uint32_t>(0xC4D08B49 + (dst << 16));
			*(uint64_t*)(p + pos + 4) = 0x0E04F6FB62ULL + (dst << 27);
			pos += 9;
		}
		else {
			*(uint64_t*)(p + pos) = 0x86F6FB62C4D08B49ULL + (dst << 16) + (dst << 59);
			*(uint32_t*)(p + pos + 8) = instr.getImm32() & ScratchpadL3Mask;
			pos += 12;
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_ISMULH_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		*(uint64_t*)(p + pos) = 0x8b4ce8f749c08b49ull + (dst << 16) + (src << 40);
		pos += 8;
		emitByte(0xc2 + 8 * dst, p, pos);

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_ISMULH_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			genAddressReg<false>(instr, src, p, pos);
			*(uint64_t*)(p + pos) = 0x0e2cf748c08b49ull + (dst << 16);
			pos += 7;
		}
		else {
			*(uint64_t*)(p + pos) = 0xaef748c08b49ull + (dst << 16);
			pos += 6;
			genAddressImm(instr, p, pos);
		}
		*(uint32_t*)(p + pos) = 0xc28b4c + (dst << 19);
		pos += 3;

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IMUL_RCP(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;
		
		uint64_t divisor = instr.getImm32();
		if (!isZeroOrPowerOf2(divisor)) {
			*(uint32_t*)(p + pos) = 0xb848;
			pos += 2;

			emit64(randomx_reciprocal_fast(divisor), p, pos);

			const uint32_t dst = instr.dst % RegistersCount;
			emit32(0xc0af0f4c + (dst << 27), p, pos);

			registerUsage[dst] = pos;
		}

		codePos = pos;
	}

	void JitCompilerX86::h_INEG_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t dst = instr.dst % RegistersCount;
		*(uint32_t*)(p + pos) = 0xd8f749 + (dst << 16);
		pos += 3;

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IXOR_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			*(uint32_t*)(p + pos) = 0xc0334d + (((dst << 3) + src) << 16);
			pos += 3;
		}
		else {
			const uint64_t imm = instr.getImm32();
			*(uint64_t*)(p + pos) = (imm << 24) + 0xf08149 + (dst << 16);
			pos += 7;
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IXOR_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			genAddressReg<true>(instr, src, p, pos);
			emit32(0x0604334c + (dst << 19), p, pos);
		}
		else {
			*(uint32_t*)(p + pos) = 0x86334c + (dst << 19);
			pos += 3;
			genAddressImm(instr, p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IROR_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			*(uint64_t*)(p + pos) = 0xc8d349c88b41ull + (src << 16) + (dst << 40);
			pos += 6;
		}
		else {
			*(uint32_t*)(p + pos) = 0xc8c149 + (dst << 16);
			pos += 3;
			emitByte(instr.getImm32() & 63, p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_IROL_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			*(uint64_t*)(p + pos) = 0xc0d349c88b41ull + (src << 16) + (dst << 40);
			pos += 6;
		}
		else {
			*(uint32_t*)(p + pos) = 0xc0c149 + (dst << 16);
			pos += 3;
			emitByte(instr.getImm32() & 63, p, pos);
		}

		registerUsage[dst] = pos;
		codePos = pos;
	}

	void JitCompilerX86::h_ISWAP_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegistersCount;

		if (src != dst) {
			*(uint32_t*)(p + pos) = 0xc0874d + (((dst << 3) + src) << 16);
			pos += 3;
			registerUsage[dst] = pos;
			registerUsage[src] = pos;
		}

		codePos = pos;
	}

	void JitCompilerX86::h_FSWAP_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t dst = instr.dst % RegistersCount;

		*(uint64_t*)(p + pos) = 0x01c0c60f66ull + (((dst << 3) + dst) << 24);
		pos += 5;

		codePos = pos;
	}

	void JitCompilerX86::h_FADD_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t dst = instr.dst % RegisterCountFlt;
		const uint64_t src = instr.src % RegisterCountFlt;

		*(uint64_t*)(p + pos) = 0xc0580f4166ull + (((dst << 3) + src) << 32);
		pos += 5;

		codePos = pos;
	}

	void JitCompilerX86::h_FADD_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegisterCountFlt;

		genAddressReg<true>(instr, src, p, pos);
		*(uint64_t*)(p + pos) = 0x41660624e60f44f3ull;
		*(uint32_t*)(p + pos + 8) = 0xc4580f + (dst << 19);
		pos += 11;

		codePos = pos;
	}

	void JitCompilerX86::h_FSUB_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t dst = instr.dst % RegisterCountFlt;
		const uint64_t src = instr.src % RegisterCountFlt;

		*(uint64_t*)(p + pos) = 0xc05c0f4166ull + (((dst << 3) + src) << 32);
		pos += 5;

		codePos = pos;
	}

	void JitCompilerX86::h_FSUB_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint32_t dst = instr.dst % RegisterCountFlt;

		genAddressReg<true>(instr, src, p, pos);
		*(uint64_t*)(p + pos) = 0x41660624e60f44f3ull;
		*(uint32_t*)(p + pos + 8) = 0xc45c0f + (dst << 19);
		pos += 11;

		codePos = pos;
	}

	void JitCompilerX86::h_FSCAL_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t dst = instr.dst % RegisterCountFlt;

		emit32(0xc7570f41 + (dst << 27), p, pos);

		codePos = pos;
	}

	void JitCompilerX86::h_FMUL_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;
		
		const uint64_t dst = instr.dst % RegisterCountFlt;
		const uint64_t src = instr.src % RegisterCountFlt;

		*(uint64_t*)(p + pos) = 0xe0590f4166ull + (((dst << 3) + src) << 32);
		pos += 5;

		codePos = pos;
	}

	void JitCompilerX86::h_FDIV_M(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;
		const uint64_t dst = instr.dst % RegisterCountFlt;

		genAddressReg<true>(instr, src, p, pos);

		*(uint64_t*)(p + pos) = 0x0624e60f44f3ull;
		pos += 6;
		if (hasXOP) {
			*(uint64_t*)(p + pos) = 0xd0e6a218488full;
			pos += 6;
		}
		else {
			*(uint64_t*)(p + pos) = 0xe6560f45e5540f45ull;
			pos += 8;
		}
		*(uint64_t*)(p + pos) = 0xe45e0f4166ull + (dst << 35);
		pos += 5;

		codePos = pos;
	}

	void JitCompilerX86::h_FSQRT_R(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t dst = instr.dst % RegisterCountFlt;

		emit32(0xe4510f66 + (((dst << 3) + dst) << 24), p, pos);

		codePos = pos;
	}

	void JitCompilerX86::h_CFROUND(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint32_t src = instr.src % RegistersCount;

		*(uint32_t*)(p + pos) = 0x00C08B49 + (src << 16);
		const int rotate = (static_cast<int>(instr.getImm32() & 63) - 2) & 63;
		*(uint32_t*)(p + pos + 3) = 0x00C8C148 + (rotate << 24);

		if (vm_flags & RANDOMX_FLAG_AMD) {
			*(uint64_t*)(p + pos + 7) = 0x742024443B0CE083ULL;
			*(uint64_t*)(p + pos + 15) = 0x8900EB0414AE0F0AULL;
			*(uint32_t*)(p + pos + 23) = 0x202444;
			pos += 26;
		}
		else {
			*(uint64_t*)(p + pos + 7) = 0x0414AE0F0CE083ULL;
			pos += 14;
		}

		codePos = pos;
	}

	void JitCompilerX86::h_CFROUND_BMI2(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		const uint64_t src = instr.src % RegistersCount;

		const uint64_t rotate = (static_cast<int>(instr.getImm32() & 63) - 2) & 63;
		*(uint64_t*)(p + pos) = 0xC0F0FBC3C4ULL | (src << 32) | (rotate << 40);

		if (vm_flags & RANDOMX_FLAG_AMD) {
			*(uint64_t*)(p + pos + 6) = 0x742024443B0CE083ULL;
			*(uint64_t*)(p + pos + 14) = 0x8900EB0414AE0F0AULL;
			*(uint32_t*)(p + pos + 22) = 0x202444;
			pos += 25;
		}
		else {
			*(uint64_t*)(p + pos + 6) = 0x0414AE0F0CE083ULL;
			pos += 13;
		}

		codePos = pos;
	}

	template<bool jccErratum>
	void JitCompilerX86::h_CBRANCH(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;
		
		const int reg = instr.dst % RegistersCount;
		int32_t jmp_offset = registerUsage[reg] - (pos + 16);

		if (jccErratum) {
			const uint32_t branch_begin = static_cast<uint32_t>(pos + 7);
			const uint32_t branch_end = static_cast<uint32_t>(branch_begin + ((jmp_offset >= -128) ? 9 : 13));

			// If the jump crosses or touches 32-byte boundary, align it
			if ((branch_begin ^ branch_end) >= 32) {
				const uint32_t alignment_size = 32 - (branch_begin & 31);
				jmp_offset -= alignment_size;
				emit(JMP_ALIGN_PREFIX[alignment_size], alignment_size, p, pos);
			}
		}

		*(uint32_t*)(p + pos) = 0x00c08149 + (reg << 16);
		const int shift = instr.getModCond();
		const uint32_t or_mask = (1UL << RandomX_ConfigurationBase::JumpOffset) << shift;
		const uint32_t and_mask = rotl32(~static_cast<uint32_t>(1UL << (RandomX_ConfigurationBase::JumpOffset - 1)), shift);
		*(uint32_t*)(p + pos + 3) = (instr.getImm32() | or_mask) & and_mask;
		*(uint32_t*)(p + pos + 7) = 0x00c0f749 + (reg << 16);
		*(uint32_t*)(p + pos + 10) = RandomX_ConfigurationBase::ConditionMask_Calculated << shift;
		pos += 14;

		if (jmp_offset >= -128) {
			*(uint32_t*)(p + pos) = 0x74 + (jmp_offset << 8);
			pos += 2;
		}
		else {
			*(uint64_t*)(p + pos) = 0x840f + ((static_cast<int64_t>(jmp_offset) - 4) << 16);
			pos += 6;
		}

		//mark all registers as used
		uint64_t* r = (uint64_t*) registerUsage;
		uint64_t k = pos;
		k |= k << 32;
		for (unsigned j = 0; j < RegistersCount / 2; ++j) {
			r[j] = k;
		}

		codePos = pos;
	}

	template void JitCompilerX86::h_CBRANCH<false>(const Instruction&);
	template void JitCompilerX86::h_CBRANCH<true>(const Instruction&);

	void JitCompilerX86::h_ISTORE(const Instruction& instr) {
		uint8_t* const p = code;
		uint32_t pos = codePos;

		genAddressRegDst(instr, p, pos);
		emit32(0x0604894c + (static_cast<uint32_t>(instr.src % RegistersCount) << 19), p, pos);

		codePos = pos;
	}

	void JitCompilerX86::h_NOP(const Instruction& instr) {
		emitByte(0x90, code, codePos);
	}

	alignas(64) InstructionGeneratorX86 JitCompilerX86::engine[256] = {};

}
