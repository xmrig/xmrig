/*
Copyright (c) 2023 tevador <tevador@gmail.com>

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
#include <cassert>
#include "crypto/randomx/jit_compiler_rv64.hpp"
#include "crypto/randomx/jit_compiler_rv64_static.hpp"
#include "crypto/randomx/jit_compiler_rv64_vector.h"
#include "crypto/randomx/jit_compiler_rv64_vector_static.h"
#include "crypto/randomx/superscalar.hpp"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/reciprocal.h"
#include "crypto/randomx/virtual_memory.hpp"
#include "crypto/common/VirtualMemory.h"


static bool hugePagesJIT = false;
static int optimizedDatasetInit = -1;

void randomx_set_huge_pages_jit(bool hugePages)
{
	hugePagesJIT = hugePages;
}

void randomx_set_optimized_dataset_init(int value)
{
	optimizedDatasetInit = value;
}

#define alignSize(pos, align) (((pos - 1) / align + 1) * align)


namespace rv64 {
	constexpr uint16_t C_LUI    =     0x6001;
	constexpr uint32_t LUI      = 0x00000037;
	constexpr uint16_t C_ADDI   =     0x0001;
	constexpr uint32_t ADDI     = 0x00000013;
	constexpr uint32_t ADDIW    = 0x0000001b;
	constexpr uint16_t C_ADD    =     0x9002;
	constexpr uint32_t ADD      = 0x00000033;
	constexpr uint32_t SHXADD   = 0x20000033; //Zba
	constexpr uint32_t SLL      = 0x00001033;
	constexpr uint32_t SRL      = 0x00005033;
	constexpr uint32_t SLLI     = 0x00001013;
	constexpr uint32_t C_SLLI   =     0x0002;
	constexpr uint32_t SRLI     = 0x00005013;
	constexpr uint32_t AND      = 0x00007033;
	constexpr uint32_t ANDI     = 0x00007013;
	constexpr uint16_t C_AND    =     0x8c61;
	constexpr uint16_t C_ANDI   =     0x8801;
	constexpr uint32_t OR       = 0x00006033;
	constexpr uint16_t C_OR     =     0x8c41;
	constexpr uint32_t XOR      = 0x00004033;
	constexpr uint16_t C_XOR    =     0x8c21;
	constexpr uint32_t LD       = 0x00003003;
	constexpr uint16_t C_LD     =     0x6000;
	constexpr uint16_t C_LW     =     0x4000;
	constexpr uint32_t SD       = 0x00003023;
	constexpr uint32_t SUB      = 0x40000033;
	constexpr uint16_t C_SUB    =     0x8c01;
	constexpr uint32_t MUL      = 0x02000033;
	constexpr uint32_t MULHU    = 0x02003033;
	constexpr uint32_t MULH     = 0x02001033;
	constexpr uint16_t C_MV     =     0x8002;
	constexpr uint32_t ROR      = 0x60005033; //Zbb
	constexpr uint32_t RORI     = 0x60005013; //Zbb
	constexpr uint32_t ROL      = 0x60001033; //Zbb
	constexpr uint32_t FMV_X_D  = 0xe2000053;
	constexpr uint32_t FMV_D_X  = 0xf2000053;
	constexpr uint32_t FMV_D    = 0x22000053;
	constexpr uint32_t FADD_D   = 0x02007053;
	constexpr uint32_t FSUB_D   = 0x0a007053;
	constexpr uint32_t FMUL_D   = 0x12007053;
	constexpr uint32_t FDIV_D   = 0x1a007053;
	constexpr uint32_t FSQRT_D  = 0x5a007053;
	constexpr uint32_t FCVT_D_W = 0xd2000053;
	constexpr uint32_t FSRM     = 0x00201073;
	constexpr uint16_t C_BEQZ   =     0xc001;
	constexpr uint32_t BEQ      = 0x00000063;
	constexpr uint16_t C_BNEZ   =     0xe001;
	constexpr uint32_t JAL      = 0x0000006f;
	constexpr uint16_t C_RET    =     0x8082;
}

namespace randomx {

	constexpr size_t MaxRandomXInstrCodeSize = 56;     //FDIV_M requires 56 bytes of rv64 code
	constexpr size_t MaxSuperscalarInstrSize = 12;     //IXOR_C requires 12 bytes of rv64 code
	constexpr size_t SuperscalarProgramHeader = 136;   //overhead per superscalar program
	constexpr size_t CodeAlign = 4096;                 //align code size to a multiple of 4 KiB
	constexpr size_t LiteralPoolSize = CodeAlign;
	constexpr size_t SuperscalarLiteraPoolSize = RANDOMX_CACHE_MAX_ACCESSES * CodeAlign;
	constexpr size_t ReserveCodeSize = CodeAlign;  //prologue, epilogue + reserve

	constexpr size_t RandomXCodeSize = alignSize(LiteralPoolSize + ReserveCodeSize + MaxRandomXInstrCodeSize * RANDOMX_PROGRAM_MAX_SIZE, CodeAlign);
	constexpr size_t SuperscalarSize = alignSize(SuperscalarLiteraPoolSize + ReserveCodeSize + (SuperscalarProgramHeader + MaxSuperscalarInstrSize * SuperscalarMaxSize) * RANDOMX_CACHE_MAX_ACCESSES, CodeAlign);

	constexpr uint32_t CodeSize = RandomXCodeSize + SuperscalarSize;
	constexpr uint32_t ExecutableSize = CodeSize - LiteralPoolSize;

	constexpr int32_t LiteralPoolOffset = LiteralPoolSize / 2;
	constexpr int32_t SuperScalarLiteralPoolOffset = RandomXCodeSize;
	constexpr int32_t SuperScalarLiteralPoolRefOffset = RandomXCodeSize + (RANDOMX_CACHE_MAX_ACCESSES - 1) * LiteralPoolSize + LiteralPoolOffset;
	constexpr int32_t SuperScalarHashOffset = SuperScalarLiteralPoolOffset + SuperscalarLiteraPoolSize;

	constexpr int32_t unsigned32ToSigned2sCompl(uint32_t x) {
		return (-1 == ~0) ? (int32_t)x : (x > INT32_MAX ? (-(int32_t)(UINT32_MAX - x) - 1) : (int32_t)x);
	}

#define MaskL1Shift (32 - RandomX_CurrentConfig.Log2_ScratchpadL1)
#define MaskL2Shift (32 - RandomX_CurrentConfig.Log2_ScratchpadL2)
#define	MaskL3Shift (32 - RandomX_CurrentConfig.Log2_ScratchpadL3)

	constexpr int RcpLiteralsOffset = 144;

	constexpr int LiteralPoolReg = 3; //x3
	constexpr int SpadReg = 5;  //x5
	constexpr int DataReg = 6;  //x6
	constexpr int SuperscalarReg = 7; //x7
	constexpr int SshTmp1Reg = 28; //x28
	constexpr int SshTmp2Reg = 29; //x29
	constexpr int SshPoolReg = 30; //x30
	constexpr int SshRcpReg = 31; //x31
	constexpr int Tmp1Reg = 8;  //x8
	constexpr int Tmp2Reg = 9;  //x9
	constexpr int Tmp1RegF = 24;  //f24
	constexpr int Tmp2RegF = 25;  //f25
	constexpr int MaskL1Reg = 10; //x10
	constexpr int MaskL2Reg = 11; //x11
	constexpr int MaskFscalReg = 12; //x12
	constexpr int MaskEclear = 13; //x13
	constexpr int MaskEsetLo = 14; //x14
	constexpr int MaskEsetHi = 15; //x15
	constexpr int MaskL3Reg = 1; //x1
	constexpr int ReturnReg = 1; //x1
	constexpr int SpAddr0Reg = 26; //x26
	constexpr int OffsetXC = -8; //x8-x15
	constexpr int OffsetR = 16; //x16-x23
	constexpr int OffsetF = 0;  //f0-f7
	constexpr int OffsetE = 8; //f8-f15
	constexpr int OffsetA = 16;  //f16-f23
	constexpr int OffsetRcp = 28; //x28-x31
	constexpr int OffsetRcpF = 22; //f26-f31
	constexpr int OffsetSsh = 8; //x8-x15

	//destination register (bit 7+)
	constexpr int rvrd(int reg) {
		return reg << 7;
	}

	//first source register (bit 15+)
	constexpr int rvrs1(int reg) {
		return reg << 15;
	}

	//second source register (bit 20+)
	constexpr int rvrs2(int reg) {
		return reg << 20;
	}

	//compressed source register (bit 2+)
	constexpr int rvcrs(int reg) {
		return reg << 2;
	}

	//base instruction: {op} x{rd}, x{rs1}, x{rs2}
	constexpr uint32_t rvi(uint32_t op, int rd, int rs1, int rs2 = 0) {
		return op | rvrs2(rs2) | rvrs1(rs1) | rvrd(rd);
	}

	//compressed instruction: op x{rd}, x{rs}
	constexpr uint16_t rvc(uint16_t op, int rd, int rs) {
		return op | rvrd(rd) | rvcrs(rs);
	}

	//compressed instruction: op x{rd}, imm6
	constexpr uint16_t rvc(uint16_t op, int imm5, int rd, int imm40) {
		return op | (imm5 << 12) | rvrd(rd) | (imm40 << 2);
	}

	constexpr int regR(int reg) {
		return reg + OffsetR;
	}

	constexpr int regLoA(int reg) {
		return 2 * reg + OffsetA;
	}

	constexpr int regHiA(int reg) {
		return 2 * reg + OffsetA + 1;
	}

	constexpr int regLoF(int reg) {
		return 2 * reg + OffsetF;
	}

	constexpr int regHiF(int reg) {
		return 2 * reg + OffsetF + 1;
	}

	constexpr int regLoE(int reg) {
		return 2 * reg + OffsetE;
	}

	constexpr int regHiE(int reg) {
		return 2 * reg + OffsetE + 1;
	}

	constexpr int regRcp(int reg) {
		return reg + OffsetRcp;
	}

	constexpr int regRcpF(int reg) {
		return reg + OffsetRcpF;
	}

	constexpr int regSS(int reg) {
		return reg + OffsetSsh;
	}

	static const uint8_t* codeLiterals = (uint8_t*)&randomx_riscv64_literals;
	static const uint8_t* codeLiteralsEnd = (uint8_t*)&randomx_riscv64_literals_end;
	static const uint8_t* codeDataInit = (uint8_t*)&randomx_riscv64_data_init;
	static const uint8_t* codeFixDataCall = (uint8_t*)&randomx_riscv64_fix_data_call;
	static const uint8_t* codePrologue = (uint8_t*)&randomx_riscv64_prologue;
	static const uint8_t* codeLoopBegin = (uint8_t*)&randomx_riscv64_loop_begin;
	static const uint8_t* codeDataRead = (uint8_t*)&randomx_riscv64_data_read;
	static const uint8_t* codeDataReadLight = (uint8_t*)&randomx_riscv64_data_read_light;
	static const uint8_t* codeFixLoopCall = (uint8_t*)&randomx_riscv64_fix_loop_call;
	static const uint8_t* codeSpadStore = (uint8_t*)&randomx_riscv64_spad_store;
	static const uint8_t* codeSpadStoreHardAes = (uint8_t*)&randomx_riscv64_spad_store_hardaes;
	static const uint8_t* codeSpadStoreSoftAes = (uint8_t*)&randomx_riscv64_spad_store_softaes;
	static const uint8_t* codeLoopEnd = (uint8_t*)&randomx_riscv64_loop_end;
	static const uint8_t* codeFixContinueLoop = (uint8_t*)&randomx_riscv64_fix_continue_loop;
	static const uint8_t* codeEpilogue = (uint8_t*)&randomx_riscv64_epilogue;
	static const uint8_t* codeSoftAes = (uint8_t*)&randomx_riscv64_softaes;
	static const uint8_t* codeProgramEnd = (uint8_t*)&randomx_riscv64_program_end;
	static const uint8_t* codeSshInit = (uint8_t*)&randomx_riscv64_ssh_init;
	static const uint8_t* codeSshLoad = (uint8_t*)&randomx_riscv64_ssh_load;
	static const uint8_t* codeSshPrefetch = (uint8_t*)&randomx_riscv64_ssh_prefetch;
	static const uint8_t* codeSshEnd = (uint8_t*)&randomx_riscv64_ssh_end;

	static const int32_t sizeLiterals = codeLiteralsEnd - codeLiterals;
	static const int32_t sizeDataInit = codePrologue - codeDataInit;
	static const int32_t sizePrologue = codeLoopBegin - codePrologue;
	static const int32_t sizeLoopBegin = codeDataRead - codeLoopBegin;
	static const int32_t sizeDataRead = codeDataReadLight - codeDataRead;
	static const int32_t sizeDataReadLight = codeSpadStore - codeDataReadLight;
	static const int32_t sizeSpadStore = codeSpadStoreHardAes - codeSpadStore;
	static const int32_t sizeSpadStoreSoftAes = codeLoopEnd - codeSpadStoreSoftAes;
	static const int32_t sizeLoopEnd = codeEpilogue - codeLoopEnd;
	static const int32_t sizeEpilogue = codeSoftAes - codeEpilogue;
	static const int32_t sizeSoftAes = codeProgramEnd - codeSoftAes;
	static const int32_t sizeSshInit = codeSshLoad - codeSshInit;
	static const int32_t sizeSshLoad = codeSshPrefetch - codeSshLoad;
	static const int32_t sizeSshPrefetch = codeSshEnd - codeSshPrefetch;

	static const int32_t offsetFixDataCall = codeFixDataCall - codeDataInit;
	static const int32_t offsetFixLoopCall = codeFixLoopCall - codeDataReadLight;
	static const int32_t offsetFixContinueLoop = codeFixContinueLoop - codeLoopEnd;

	static const int32_t LoopTopPos = LiteralPoolSize + sizeDataInit + sizePrologue;
	static const int32_t RandomXCodePos = LoopTopPos + sizeLoopBegin;

	static void clearCache(CodeBuffer& buf) {
#ifdef __GNUC__
		__builtin___clear_cache((char*)buf.code, (char*)(buf.code + CodeSize));
#endif
	}

	//emits code to calculate: x{dst} = x{src} + {imm32}
	//takes 1-3 isns, 2-10 bytes
	static void emitImm32(CodeBuffer& buf, int32_t imm, int dst, int src = 0, int tmp = 0) {

		//lower 12 bits
		int32_t limm = (imm << 20) >> 20;
		//upper 20 bits
		int32_t uimm = (imm >> 12) + (limm < 0);

		//If there are no upper bits, the whole thing
		//can be done with a single instruction.
		if (uimm == 0) {
			//addi x{dst}, x{src}, {limm}
			buf.emit(rvi(rv64::ADDI, dst, src, limm));
			return;
		}

		//dst1 is the register where imm will be materialized
		int dst1 = src != dst ? dst : tmp;
		assert(dst1 != 0);
		//src1 is the register that will be added to the result
		int src1 = src != dst ? src : dst1;

		//load upper bits
		if (uimm >= -32 && uimm <= 31) {
			//c.lui x{dst1}, {uimm}
			buf.emit(rvc(rv64::C_LUI, (uimm < 0), dst1, (uimm & 31)));
		}
		else {
			//lui x{dst1}, {uimm}
			buf.emit(rv64::LUI | (uimm << 12) | rvrd(dst1));
		}
		//load lower bits
		if (limm != 0) {
			//Note: this must be addiw NOT addi, otherwise the upper 32 bits
			//of the 64-bit register will be incorrect.
			//addiw x{dst1}, x{dst1}, {limm}
			buf.emit(rvi(rv64::ADDIW, dst1, dst1, limm));
		}
		//add src
		if (src1 != 0) {
			//c.add x{dst}, x{src1}
			buf.emit(rvc(rv64::C_ADD, dst, src1));
		}
	}

	//x9 = &Scratchpad[isn.imm]
	//takes 3 isns, 10 bytes
	static void genAddressRegImm(CodeBuffer& buf, const Instruction& isn) {
		//signed offset 8-byte aligned
		int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32()) & ScratchpadL3Mask;
		//x9 = x5 + {imm}
		emitImm32(buf, imm, Tmp2Reg, SpadReg, Tmp1Reg);
	}

	//x9 = &Scratchpad[isn.src + isn.imm] (for reading)
	//takes 5 isns, 12 bytes
	static void genAddressReg(CodeBuffer& buf, const Instruction& isn) {
		int shift, maskReg;
		if (isn.getModMem()) {
			shift = MaskL1Shift;
			maskReg = MaskL1Reg;
		}
		else {
			shift = MaskL2Shift;
			maskReg = MaskL2Reg;
		}
		int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
		imm = (imm << shift) >> shift;
		//x9 = x{src} + {imm}
		emitImm32(buf, imm, Tmp2Reg, regR(isn.src), Tmp1Reg);
		//c.and x9, x{maskReg}
		buf.emit(rvc(rv64::C_AND, (Tmp2Reg + OffsetXC), (maskReg + OffsetXC)));
		//c.add x9, x{spadReg}
		buf.emit(rvc(rv64::C_ADD, Tmp2Reg, SpadReg));
	}

	//x8 = Scratchpad[isn]
	static void loadFromScratchpad(CodeBuffer& buf, const Instruction& isn) {
		if (isn.src != isn.dst) {
			//x9 = &Scratchpad[isn.src + isn.imm]
			genAddressReg(buf, isn);
		}
		else {
			///x9 = &Scratchpad[isn.imm]
			genAddressRegImm(buf, isn);
		}
		//c.ld x8, 0(x9)
		buf.emit(rvc(rv64::C_LD, Tmp2Reg + OffsetXC, Tmp1Reg + OffsetXC));
	}

	//x9 = &Scratchpad[isn.dst + isn.imm32] (for writing)
	//takes 5 isns, 12-16 bytes
	static void genAddressRegDst(CodeBuffer& buf, const Instruction& isn) {
		if (isn.getModCond() < StoreL3Condition) {
			int shift, maskReg;
			if (isn.getModMem()) {
				shift = MaskL1Shift;
				maskReg = MaskL1Reg;
			}
			else {
				shift = MaskL2Shift;
				maskReg = MaskL2Reg;
			}
			int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
			imm = (imm << shift) >> shift;
			//x9 = x{dst} + {imm}
			emitImm32(buf, imm, Tmp2Reg, regR(isn.dst), Tmp1Reg);
			//c.and x9, x{maskReg}
			buf.emit(rvc(rv64::C_AND, Tmp2Reg + OffsetXC, maskReg + OffsetXC));
			//c.add x9, x5
			buf.emit(rvc(rv64::C_ADD, Tmp2Reg, SpadReg));
		}
		else {
			int shift = MaskL3Shift;
			int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
			imm = (imm << shift) >> shift;
			//x9 = x{dst} + {imm}
			emitImm32(buf, imm, Tmp2Reg, regR(isn.dst), Tmp1Reg);
			//and x9, x9, x1
			buf.emit(rvi(rv64::AND, Tmp2Reg, Tmp2Reg, MaskL3Reg));
			//c.add x9, x5
			buf.emit(rvc(rv64::C_ADD, Tmp2Reg, SpadReg));
		}
	}

	static void emitRcpLiteral1(CodeBuffer& buf, uint64_t literal) {
		//first 238 at positive offsets
		if (buf.rcpCount < 238) {
			buf.emitAt(LiteralPoolOffset + RcpLiteralsOffset + buf.rcpCount * 8, literal);
			buf.rcpCount++;
		}
		//next 256 at negative offsets
		else if (buf.rcpCount < 494) {
			buf.emitAt(buf.rcpCount * 8 - (2048 - RcpLiteralsOffset), literal);
			buf.rcpCount++;
		}
		else {
			//checked at compile time, but double-check here
			throw std::runtime_error("Literal pool overflow");
		}
	}

	static void emitRcpLiteral2(CodeBuffer& buf, uint64_t literal, bool lastLiteral) {
		//store the current literal in the pool
		int32_t offset = 2040 - buf.rcpCount * 8;
		buf.emitAt(SuperScalarLiteralPoolRefOffset + offset, literal);
		buf.rcpCount++;
		if (lastLiteral) {
			return;
		}
		//load the next literal
		offset -= 8;
		int32_t imm = offset & 0xfff;
		//ld x31, {offset}(x30)
		buf.emit(rvi(rv64::LD, SshRcpReg, SshPoolReg, imm));
		if (imm == 0x800) {
			//move pool pointer back 4KB
			//c.lui x29, 0xfffff
			buf.emit(rvc(rv64::C_LUI, 1, SshTmp2Reg, 31));
			//c.add x30, x29
			buf.emit(rvc(rv64::C_ADD, SshPoolReg, SshTmp2Reg));
		}
	}

	static void emitJump(CodeBuffer& buf, int dst, int32_t codePos, int32_t targetPos) {
		int32_t imm = targetPos - codePos;
		int32_t imm20 = (imm < 0) << 11;
		int32_t imm1912 = (imm >> 7) & 8160;
		int32_t imm11 = (imm >> 11) & 1;
		int32_t imm101 = imm & 2046;
		//jal x{dst}, {imm}
		buf.emitAt(codePos, rvi(rv64::JAL, dst + imm1912, 0, imm20 + imm101 + imm11));
	}

	static void emitInstruction(CompilerState& state, Instruction isn, int i) {
		state.instructionOffsets[i] = state.codePos;
		(*JitCompilerRV64::engine[isn.opcode])(state, isn, i);
	}

	static void emitProgramPrefix(CompilerState& state, Program& prog, ProgramConfiguration& pcfg) {
		state.codePos = RandomXCodePos;
		state.rcpCount = 0;
		state.emitAt(LiteralPoolOffset + sizeLiterals, pcfg.eMask[0]);
		state.emitAt(LiteralPoolOffset + sizeLiterals + 8, pcfg.eMask[1]);
		for (unsigned i = 0; i < RegistersCount; ++i) {
			state.registerUsage[i] = -1;
		}
		for (unsigned i = 0; i < prog.getSize(); ++i) {
			Instruction instr = prog(i);
			instr.src %= RegistersCount;
			instr.dst %= RegistersCount;
			emitInstruction(state, instr, i);
		}
	}

	static void emitProgramSuffix(CompilerState& state, ProgramConfiguration& pcfg) {
		state.emit(codeSpadStore, sizeSpadStore);
		int32_t fixPos = state.codePos;
		state.emit(codeLoopEnd, sizeLoopEnd);
		//xor x26, x{readReg0}, x{readReg1}
		state.emitAt(fixPos, rvi(rv64::XOR, SpAddr0Reg, regR(pcfg.readReg0), regR(pcfg.readReg1)));
		fixPos += offsetFixContinueLoop;
		//j LoopTop
		emitJump(state, 0, fixPos, LoopTopPos);
		state.emit(codeEpilogue, sizeEpilogue);
	}

	static void generateSuperscalarCode(CodeBuffer& buf, Instruction isn, bool lastLiteral) {
		switch ((SuperscalarInstructionType)isn.opcode)
		{
		case randomx::SuperscalarInstructionType::ISUB_R:
			//c.sub x{dst}, x{src}
			buf.emit(rvc(rv64::C_SUB, regSS(isn.dst) + OffsetXC, regSS(isn.src) + OffsetXC));
			break;
		case randomx::SuperscalarInstructionType::IXOR_R:
			//c.xor x{dst}, x{src}
			buf.emit(rvc(rv64::C_XOR, regSS(isn.dst) + OffsetXC, regSS(isn.src) + OffsetXC));
			break;
		case randomx::SuperscalarInstructionType::IADD_RS:
			{
				int shift = isn.getModShift();
				if (shift == 0) {
					//c.add x{dst}, x{src}
					buf.emit(rvc(rv64::C_ADD, regSS(isn.dst), regSS(isn.src)));
				}
				else {
#ifdef __riscv_zba
				//sh{1,2,3}add x{dst}, x{src}, x{dst}
				buf.emit(rv64::SHXADD | rvrs2(regSS(isn.dst)) | rvrs1(regSS(isn.src)) | (shift << 13) | rvrd(regSS(isn.dst)));
#else
				//slli x28, x{src}, {shift}
				buf.emit(rvi(rv64::SLLI, SshTmp1Reg, regSS(isn.src), shift));
				//c.add x{dst}, x28
				buf.emit(rvc(rv64::C_ADD, regSS(isn.dst), SshTmp1Reg));
#endif
				}
			}
			break;
		case randomx::SuperscalarInstructionType::IMUL_R:
			//mul x{dst}, x{dst}, x{src}
			buf.emit(rvi(rv64::MUL, regSS(isn.dst), regSS(isn.dst), regSS(isn.src)));
			break;
		case randomx::SuperscalarInstructionType::IROR_C:
			{
#ifdef __riscv_zbb
				int32_t imm = isn.getImm32() & 63;
				//rori x{dst}, x{dst}, {imm}
				buf.emit(rvi(rv64::RORI, regSS(isn.dst), regSS(isn.dst), imm));
#else
				int32_t immr = isn.getImm32() & 63;
				int32_t imml = -immr & 63;
				int32_t imml5 = imml >> 5;
				int32_t imml40 = imml & 31;
				//srli x28, x{dst}, {immr}
				buf.emit(rvi(rv64::SRLI, SshTmp1Reg, regSS(isn.dst), immr));
				//c.slli x{dst}, {imml}
				buf.emit(rvc(rv64::C_SLLI, imml5, regSS(isn.dst), imml40));
				//or x{dst}, x{dst}, x28
				buf.emit(rvi(rv64::OR, regSS(isn.dst), regSS(isn.dst), SshTmp1Reg));
#endif
			}
			break;
		case randomx::SuperscalarInstructionType::IADD_C7:
		case randomx::SuperscalarInstructionType::IADD_C8:
		case randomx::SuperscalarInstructionType::IADD_C9:
			{
				int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
				//x{dst} = x{dst} + {imm}
				emitImm32(buf, imm, regSS(isn.dst), regSS(isn.dst), SshTmp1Reg);
			}
			break;
		case randomx::SuperscalarInstructionType::IXOR_C7:
		case randomx::SuperscalarInstructionType::IXOR_C8:
		case randomx::SuperscalarInstructionType::IXOR_C9:
			{
				int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
				//x28 = {imm}
				emitImm32(buf, imm, SshTmp1Reg);
				//xor x{dst}, x{dst}, x28
				buf.emit(rvi(rv64::XOR, regSS(isn.dst), regSS(isn.dst), SshTmp1Reg));
			}
			break;
		case randomx::SuperscalarInstructionType::IMULH_R:
			//mulhu x{dst}, x{dst}, x{src}
			buf.emit(rvi(rv64::MULHU, regSS(isn.dst), regSS(isn.dst), regSS(isn.src)));
			break;
		case randomx::SuperscalarInstructionType::ISMULH_R:
			//mulh x{dst}, x{dst}, x{src}
			buf.emit(rvi(rv64::MULH, regSS(isn.dst), regSS(isn.dst), regSS(isn.src)));
			break;
		case randomx::SuperscalarInstructionType::IMUL_RCP:
			//mul x{dst}, x{dst}, x31
			buf.emit(rvi(rv64::MUL, regSS(isn.dst), regSS(isn.dst), SshRcpReg));
			//load the next literal into x31
			emitRcpLiteral2(buf, randomx_reciprocal(isn.getImm32()), lastLiteral);
			break;
		default:
			UNREACHABLE;
		}
	}

	size_t JitCompilerRV64::getCodeSize() {
		return CodeSize;
	}

	JitCompilerRV64::JitCompilerRV64(bool hugePagesEnable, bool) {
		state.code = static_cast<uint8_t*>(allocExecutableMemory(CodeSize, hugePagesJIT && hugePagesEnable));
		state.emitAt(LiteralPoolOffset, codeLiterals, sizeLiterals);

		const uint32_t L1_Mask = RandomX_CurrentConfig.ScratchpadL1_Size - 8;
		const uint32_t L2_Mask = RandomX_CurrentConfig.ScratchpadL2_Size - 8;
		const uint32_t L3_Mask = RandomX_CurrentConfig.ScratchpadL3_Size - 64;
		const uint32_t DatasetBaseSize_Mask = RandomX_CurrentConfig.DatasetBaseSize - 64;

		state.emitAt(LiteralPoolOffset + 80, reinterpret_cast<const uint8_t*>(&L1_Mask), sizeof(L1_Mask));
		state.emitAt(LiteralPoolOffset + 84, reinterpret_cast<const uint8_t*>(&L2_Mask), sizeof(L2_Mask));
		state.emitAt(LiteralPoolOffset + 88, reinterpret_cast<const uint8_t*>(&L3_Mask), sizeof(L3_Mask));
		state.emitAt(LiteralPoolOffset + 92, reinterpret_cast<const uint8_t*>(&DatasetBaseSize_Mask), sizeof(DatasetBaseSize_Mask));

		state.emitAt(LiteralPoolSize, codeDataInit, sizeDataInit + sizePrologue + sizeLoopBegin);
		entryDataInit = state.code + LiteralPoolSize;
		entryProgram = state.code + LiteralPoolSize + sizeDataInit;
		//jal x1, SuperscalarHash
		emitJump(state, ReturnReg, LiteralPoolSize + offsetFixDataCall, SuperScalarHashOffset);

		vectorCodeSize = ((uint8_t*)randomx_riscv64_vector_sshash_end) - ((uint8_t*)randomx_riscv64_vector_sshash_begin);
		vectorCode = static_cast<uint8_t*>(allocExecutableMemory(vectorCodeSize, hugePagesJIT && hugePagesEnable));
	}

	JitCompilerRV64::~JitCompilerRV64() {
		freePagedMemory(state.code, CodeSize);
		freePagedMemory(vectorCode, vectorCodeSize);
	}

	void JitCompilerRV64::enableWriting() const
	{
		xmrig::VirtualMemory::protectRW(entryDataInit, ExecutableSize);

		if (vectorCode) {
			xmrig::VirtualMemory::protectRW(vectorCode, vectorCodeSize);
		}
	}

	void JitCompilerRV64::enableExecution() const
	{
		xmrig::VirtualMemory::protectRX(entryDataInit, ExecutableSize);

		if (vectorCode) {
			xmrig::VirtualMemory::protectRX(vectorCode, vectorCodeSize);
		}
	}

	void JitCompilerRV64::generateProgram(Program& prog, ProgramConfiguration& pcfg, uint32_t) {
		emitProgramPrefix(state, prog, pcfg);
		int32_t fixPos = state.codePos;
		state.emit(codeDataRead, sizeDataRead);
		//xor x8, x{readReg2}, x{readReg3}
		state.emitAt(fixPos, rvi(rv64::XOR, Tmp1Reg, regR(pcfg.readReg2), regR(pcfg.readReg3)));
		emitProgramSuffix(state, pcfg);
		clearCache(state);
	}

	void JitCompilerRV64::generateProgramLight(Program& prog, ProgramConfiguration& pcfg, uint32_t datasetOffset) {
		emitProgramPrefix(state, prog, pcfg);
		int32_t fixPos = state.codePos;
		state.emit(codeDataReadLight, sizeDataReadLight);
		//xor x8, x{readReg2}, x{readReg3}
		state.emitAt(fixPos, rvi(rv64::XOR, Tmp1Reg, regR(pcfg.readReg2), regR(pcfg.readReg3)));
		int32_t imm = datasetOffset / CacheLineSize;
		int32_t limm = (imm << 20) >> 20;
		int32_t uimm = (imm >> 12) + (limm < 0);
		//lui x9, {uimm}
		state.emitAt(fixPos + 4, rv64::LUI | (uimm << 12) | rvrd(Tmp2Reg));
		//addi x9, x9, {limm}
		state.emitAt(fixPos + 8, rvi(rv64::ADDI, Tmp2Reg, Tmp2Reg, limm));
		fixPos += offsetFixLoopCall;
		//jal x1, SuperscalarHash
		emitJump(state, ReturnReg, fixPos, SuperScalarHashOffset);
		emitProgramSuffix(state, pcfg);
		clearCache(state);
	}

	template<size_t N>
	void JitCompilerRV64::generateSuperscalarHash(SuperscalarProgram(&programs)[N]) {
		if (optimizedDatasetInit > 0) {
			entryDataInitOptimized = generateDatasetInitVectorRV64(vectorCode, vectorCodeSize, programs, RandomX_ConfigurationBase::CacheAccesses);
			return;
		}

		state.codePos = SuperScalarHashOffset;
		state.rcpCount = 0;
		state.emit(codeSshInit, sizeSshInit);

		std::pair<uint32_t, uint32_t> lastLiteral{ 0xFFFFFFFFUL, 0xFFFFFFFFUL };

		for (int j = RandomX_ConfigurationBase::CacheAccesses - 1; (j >= 0) && (lastLiteral.first == 0xFFFFFFFFUL); --j) {
			SuperscalarProgram& prog = programs[j];
			for (int i = prog.getSize() - 1; i >= 0; --i) {
				if (prog(i).opcode == static_cast<uint8_t>(SuperscalarInstructionType::IMUL_RCP)) {
					lastLiteral.first = j;
					lastLiteral.second = i;
					break;
				}
			}
		}

		for (unsigned j = 0; j < RandomX_ConfigurationBase::CacheAccesses; ++j) {
			SuperscalarProgram& prog = programs[j];
			for (unsigned i = 0; i < prog.getSize(); ++i) {
				Instruction instr = prog(i);
				generateSuperscalarCode(state, instr, (j == lastLiteral.first) && (i == lastLiteral.second));
			}
			state.emit(codeSshLoad, sizeSshLoad);
			if (j < RandomX_ConfigurationBase::CacheAccesses - 1) {
				int32_t fixPos = state.codePos;
				state.emit(codeSshPrefetch, sizeSshPrefetch);
				//and x7, x{addrReg}, x7
				state.emitAt(fixPos, rvi(rv64::AND, SuperscalarReg, regSS(prog.getAddressRegister()), SuperscalarReg));
			}
		}
		state.emit(rvc(rv64::C_RET, 0, 0));
		clearCache(state);
	}

	template void JitCompilerRV64::generateSuperscalarHash(SuperscalarProgram(&)[RANDOMX_CACHE_MAX_ACCESSES]);

	DatasetInitFunc* JitCompilerRV64::getDatasetInitFunc() {
		return (DatasetInitFunc*)((optimizedDatasetInit > 0) ? entryDataInitOptimized : entryDataInit);
	}

	void JitCompilerRV64::v1_IADD_RS(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		int shift = isn.getModShift();
		if (shift == 0) {
			//c.add x{dst}, x{src}
			state.emit(rvc(rv64::C_ADD, regR(isn.dst), regR(isn.src)));
		}
		else {
#ifdef __riscv_zba
			//sh{1,2,3}add x{dst}, x{src}, x{dst}
			state.emit(rv64::SHXADD | rvrs2(regR(isn.dst)) | rvrs1(regR(isn.src)) | (shift << 13) | rvrd(regR(isn.dst)));
#else
			//slli x8, x{src}, {shift}
			state.emit(rvi(rv64::SLLI, Tmp1Reg, regR(isn.src), shift));
			//c.add x{dst}, x8
			state.emit(rvc(rv64::C_ADD, regR(isn.dst), Tmp1Reg));
#endif
		}
		if (isn.dst == RegisterNeedsDisplacement) {
			int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
			//x{dst} = x{dst} + {imm}
			emitImm32(state, imm, regR(isn.dst), regR(isn.dst), Tmp1Reg);
		}
	}

	void JitCompilerRV64::v1_IADD_M(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		loadFromScratchpad(state, isn);
		//c.add x{dst}, x8
		state.emit(rvc(rv64::C_ADD, regR(isn.dst), Tmp1Reg));
	}

	void JitCompilerRV64::v1_ISUB_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		if (isn.src != isn.dst) {
			//sub x{dst}, x{dst}, x{src}
			state.emit(rvi(rv64::SUB, regR(isn.dst), regR(isn.dst), regR(isn.src)));
		}
		else {
			int32_t imm = unsigned32ToSigned2sCompl(-isn.getImm32()); //convert to add
			//x{dst} = x{dst} + {-imm}
			emitImm32(state, imm, regR(isn.dst), regR(isn.dst), Tmp1Reg);
		}
	}

	void JitCompilerRV64::v1_ISUB_M(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		loadFromScratchpad(state, isn);
		//sub x{dst}, x{dst}, x8
		state.emit(rvi(rv64::SUB, regR(isn.dst), regR(isn.dst), Tmp1Reg));
	}

	void JitCompilerRV64::v1_IMUL_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		if (isn.src != isn.dst) {
			//mul x{dst}, x{dst}, x{src}
			state.emit(rvi(rv64::MUL, regR(isn.dst), regR(isn.dst), regR(isn.src)));
		}
		else {
			int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
			//x8 = {imm}
			emitImm32(state, imm, Tmp1Reg);
			//mul x{dst}, x{dst}, x8
			state.emit(rvi(rv64::MUL, regR(isn.dst), regR(isn.dst), Tmp1Reg));
		}
	}

	void JitCompilerRV64::v1_IMUL_M(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		loadFromScratchpad(state, isn);
		//mul x{dst}, x{dst}, x8
		state.emit(rvi(rv64::MUL, regR(isn.dst), regR(isn.dst), Tmp1Reg));
	}

	void JitCompilerRV64::v1_IMULH_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		//mulhu x{dst}, x{dst}, x{src}
		state.emit(rvi(rv64::MULHU, regR(isn.dst), regR(isn.dst), regR(isn.src)));
	}

	void JitCompilerRV64::v1_IMULH_M(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		loadFromScratchpad(state, isn);
		//mulhu x{dst}, x{dst}, x8
		state.emit(rvi(rv64::MULHU, regR(isn.dst), regR(isn.dst), Tmp1Reg));
	}

	void JitCompilerRV64::v1_ISMULH_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		//mulh x{dst}, x{dst}, x{src}
		state.emit(rvi(rv64::MULH, regR(isn.dst), regR(isn.dst), regR(isn.src)));
	}

	void JitCompilerRV64::v1_ISMULH_M(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		loadFromScratchpad(state, isn);
		//mulh x{dst}, x{dst}, x8
		state.emit(rvi(rv64::MULH, regR(isn.dst), regR(isn.dst), Tmp1Reg));
	}

	void JitCompilerRV64::v1_IMUL_RCP(HANDLER_ARGS) {
		const uint32_t divisor = isn.getImm32();
		if (!isZeroOrPowerOf2(divisor)) {
			state.registerUsage[isn.dst] = i;
			if (state.rcpCount < 4) {
				//mul x{dst}, x{dst}, x{rcp}
				state.emit(rvi(rv64::MUL, regR(isn.dst), regR(isn.dst), regRcp(state.rcpCount)));
			}
			else if (state.rcpCount < 10) {
				//fmv.x.d x8, f{rcp}
				state.emit(rvi(rv64::FMV_X_D, Tmp1Reg, regRcpF(state.rcpCount)));
				//mul x{dst}, x{dst}, x8
				state.emit(rvi(rv64::MUL, regR(isn.dst), regR(isn.dst), Tmp1Reg));
			}
			else {
				int32_t offset = RcpLiteralsOffset + state.rcpCount * 8;
				//ld x8, {offset}(x3)
				state.emit(rvi(rv64::LD, Tmp1Reg, LiteralPoolReg, offset));
				//mul x{dst}, x{dst}, x8
				state.emit(rvi(rv64::MUL, regR(isn.dst), regR(isn.dst), Tmp1Reg));
			}
			emitRcpLiteral1(state, randomx_reciprocal_fast(divisor));
		}
	}

	void JitCompilerRV64::v1_INEG_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		//sub x{dst}, x0, x{dst}
		state.emit(rvi(rv64::SUB, regR(isn.dst), 0, regR(isn.dst)));
	}

	void JitCompilerRV64::v1_IXOR_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		if (isn.src != isn.dst) {
			//xor x{dst}, x{dst}, x{src}
			state.emit(rvi(rv64::XOR, regR(isn.dst), regR(isn.dst), regR(isn.src)));
		}
		else {
			int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
			//x8 = {imm}
			emitImm32(state, imm, Tmp1Reg);
			//xor x{dst}, x{dst}, x8
			state.emit(rvi(rv64::XOR, regR(isn.dst), regR(isn.dst), Tmp1Reg));
		}
	}

	void JitCompilerRV64::v1_IXOR_M(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
		loadFromScratchpad(state, isn);
		//xor x{dst}, x{dst}, x8
		state.emit(rvi(rv64::XOR, regR(isn.dst), regR(isn.dst), Tmp1Reg));
	}

	void JitCompilerRV64::v1_IROR_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
#ifdef __riscv_zbb
		if (isn.src != isn.dst) {
			//ror x{dst}, x{dst}, x{src}
			state.emit(rvi(rv64::ROR, regR(isn.dst), regR(isn.dst), regR(isn.src)));
		}
		else {
			int32_t imm = isn.getImm32() & 63;
			//rori x{dst}, x{dst}, {imm}
			state.emit(rvi(rv64::RORI, regR(isn.dst), regR(isn.dst), imm));
		}
#else
		if (isn.src != isn.dst) {
			//sub x8, x0, x{src}
			state.emit(rvi(rv64::SUB, Tmp1Reg, 0, regR(isn.src)));
			//srl x9, x{dst}, x{src}
			state.emit(rvi(rv64::SRL, Tmp2Reg, regR(isn.dst), regR(isn.src)));
			//sll x{dst}, x{dst}, x8
			state.emit(rvi(rv64::SLL, regR(isn.dst), regR(isn.dst), Tmp1Reg));
			//or x{dst}, x{dst}, x9
			state.emit(rvi(rv64::OR, regR(isn.dst), regR(isn.dst), Tmp2Reg));
		}
		else {
			int32_t immr = isn.getImm32() & 63;
			int32_t imml = -immr & 63;
			int32_t imml5 = imml >> 5;
			int32_t imml40 = imml & 31;
			//srli x8, x{dst}, {immr}
			state.emit(rvi(rv64::SRLI, Tmp1Reg, regR(isn.dst), immr));
			//c.slli x{dst}, {imml}
			state.emit(rvc(rv64::C_SLLI, imml5, regR(isn.dst), imml40));
			//or x{dst}, x{dst}, x8
			state.emit(rvi(rv64::OR, regR(isn.dst), regR(isn.dst), Tmp1Reg));
		}
#endif
	}

	void JitCompilerRV64::v1_IROL_R(HANDLER_ARGS) {
		state.registerUsage[isn.dst] = i;
#ifdef __riscv_zbb
		if (isn.src != isn.dst) {
			//rol x{dst}, x{dst}, x{src}
			state.emit(rvi(rv64::ROL, regR(isn.dst), regR(isn.dst), regR(isn.src)));
		}
		else {
			int32_t imm = -isn.getImm32() & 63;
			//rori x{dst}, x{dst}, {imm}
			state.emit(rvi(rv64::RORI, regR(isn.dst), regR(isn.dst), imm));
		}
#else
		if (isn.src != isn.dst) {
			//sub x8, x0, x{src}
			state.emit(rvi(rv64::SUB, Tmp1Reg, 0, regR(isn.src)));
			//sll x9, x{dst}, x{src}
			state.emit(rvi(rv64::SLL, Tmp2Reg, regR(isn.dst), regR(isn.src)));
			//srl x{dst}, x{dst}, x8
			state.emit(rvi(rv64::SRL, regR(isn.dst), regR(isn.dst), Tmp1Reg));
			//or x{dst}, x{dst}, x9
			state.emit(rvi(rv64::OR, regR(isn.dst), regR(isn.dst), Tmp2Reg));
		}
		else {
			int32_t imml = isn.getImm32() & 63;
			int32_t immr = -imml & 63;
			int32_t imml5 = imml >> 5;
			int32_t imml40 = imml & 31;
			//srli x8, x{dst}, {immr}
			state.emit(rvi(rv64::SRLI, Tmp1Reg, regR(isn.dst), immr));
			//c.slli x{dst}, {imml}
			state.emit(rvc(rv64::C_SLLI, imml5, regR(isn.dst), imml40));
			//or x{dst}, x{dst}, x8
			state.emit(rvi(rv64::OR, regR(isn.dst), regR(isn.dst), Tmp1Reg));
		}
#endif
	}

	void JitCompilerRV64::v1_ISWAP_R(HANDLER_ARGS) {
		if (isn.src != isn.dst) {
			state.registerUsage[isn.dst] = i;
			state.registerUsage[isn.src] = i;
			//c.mv x8, x{dst}
			state.emit(rvc(rv64::C_MV, Tmp1Reg, regR(isn.dst)));
			//c.mv x{dst}, x{src}
			state.emit(rvc(rv64::C_MV, regR(isn.dst), regR(isn.src)));
			//c.mv x{src}, x8
			state.emit(rvc(rv64::C_MV, regR(isn.src), Tmp1Reg));
		}
	}

	void JitCompilerRV64::v1_FSWAP_R(HANDLER_ARGS) {
		//fmv.d f24, f{dst_lo}
		state.emit(rvi(rv64::FMV_D, Tmp1RegF, regLoF(isn.dst), regLoF(isn.dst)));
		//fmv.d f{dst_lo}, f{dst_hi}
		state.emit(rvi(rv64::FMV_D, regLoF(isn.dst), regHiF(isn.dst), regHiF(isn.dst)));
		//fmv.d f{dst_hi}, f24
		state.emit(rvi(rv64::FMV_D, regHiF(isn.dst), Tmp1RegF, Tmp1RegF));
	}

	void JitCompilerRV64::v1_FADD_R(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		isn.src %= RegisterCountFlt;
		//fadd.d f{dst_lo}, f{dst_lo}, f{src_lo}
		state.emit(rvi(rv64::FADD_D, regLoF(isn.dst), regLoF(isn.dst), regLoA(isn.src)));
		//fadd.d f{dst_hi}, f{dst_hi}, f{src_hi}
		state.emit(rvi(rv64::FADD_D, regHiF(isn.dst), regHiF(isn.dst), regHiA(isn.src)));
	}

	void JitCompilerRV64::v1_FADD_M(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		//x9 = mem
		genAddressReg(state, isn);
		//lw x8, 0(x9)
		state.emit(rvc(rv64::C_LW, Tmp2Reg + OffsetXC, Tmp1Reg + OffsetXC));
		//lw x9, 4(x9)
		state.emit(rvc(rv64::C_LW, Tmp2Reg + OffsetXC, 16 + Tmp2Reg + OffsetXC));
		//fcvt.d.w f24, x8
		state.emit(rvi(rv64::FCVT_D_W, Tmp1RegF, Tmp1Reg));
		//fcvt.d.w f25, x9
		state.emit(rvi(rv64::FCVT_D_W, Tmp2RegF, Tmp2Reg));
		//fadd.d f{dst_lo}, f{dst_lo}, f24
		state.emit(rvi(rv64::FADD_D, regLoF(isn.dst), regLoF(isn.dst), Tmp1RegF));
		//fadd.d f{dst_hi}, f{dst_hi}, f25
		state.emit(rvi(rv64::FADD_D, regHiF(isn.dst), regHiF(isn.dst), Tmp2RegF));
	}

	void JitCompilerRV64::v1_FSUB_R(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		isn.src %= RegisterCountFlt;
		//fsub.d f{dst_lo}, f{dst_lo}, f{src_lo}
		state.emit(rvi(rv64::FSUB_D, regLoF(isn.dst), regLoF(isn.dst), regLoA(isn.src)));
		//fsub.d f{dst_hi}, f{dst_hi}, f{src_hi}
		state.emit(rvi(rv64::FSUB_D, regHiF(isn.dst), regHiF(isn.dst), regHiA(isn.src)));
	}

	void JitCompilerRV64::v1_FSUB_M(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		//x9 = mem
		genAddressReg(state, isn);
		//c.lw x8, 0(x9)
		state.emit(rvc(rv64::C_LW, Tmp2Reg + OffsetXC, Tmp1Reg + OffsetXC));
		//c.lw x9, 4(x9)
		state.emit(rvc(rv64::C_LW, Tmp2Reg + OffsetXC, 16 + Tmp2Reg + OffsetXC));
		//fcvt.d.w f24, x8
		state.emit(rvi(rv64::FCVT_D_W, Tmp1RegF, Tmp1Reg));
		//fcvt.d.w f25, x9
		state.emit(rvi(rv64::FCVT_D_W, Tmp2RegF, Tmp2Reg));
		//fsub.d f{dst_lo}, f{dst_lo}, f24
		state.emit(rvi(rv64::FSUB_D, regLoF(isn.dst), regLoF(isn.dst), Tmp1RegF));
		//fsub.d f{dst_hi}, f{dst_hi}, f25
		state.emit(rvi(rv64::FSUB_D, regHiF(isn.dst), regHiF(isn.dst), Tmp2RegF));
	}

	void JitCompilerRV64::v1_FSCAL_R(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		//fmv.x.d x8, f{dst_lo}
		state.emit(rvi(rv64::FMV_X_D, Tmp1Reg, regLoF(isn.dst)));
		//fmv.x.d x9, f{dst_hi}
		state.emit(rvi(rv64::FMV_X_D, Tmp2Reg, regHiF(isn.dst)));
		//c.xor x8, x12
		state.emit(rvc(rv64::C_XOR, Tmp1Reg + OffsetXC, MaskFscalReg + OffsetXC));
		//c.xor x9, x12
		state.emit(rvc(rv64::C_XOR, Tmp2Reg + OffsetXC, MaskFscalReg + OffsetXC));
		//fmv.d.x f{dst_lo}, x8
		state.emit(rvi(rv64::FMV_D_X, regLoF(isn.dst), Tmp1Reg));
		//fmv.d.x f{dst_hi}, x9
		state.emit(rvi(rv64::FMV_D_X, regHiF(isn.dst), Tmp2Reg));
	}

	void JitCompilerRV64::v1_FMUL_R(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		isn.src %= RegisterCountFlt;
		//fmul.d f{dst_lo}, f{dst_lo}, f{src_lo}
		state.emit(rvi(rv64::FMUL_D, regLoE(isn.dst), regLoE(isn.dst), regLoA(isn.src)));
		//fmul.d f{dst_hi}, f{dst_hi}, f{src_hi}
		state.emit(rvi(rv64::FMUL_D, regHiE(isn.dst), regHiE(isn.dst), regHiA(isn.src)));
	}

	void JitCompilerRV64::v1_FDIV_M(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		//x9 = mem
		genAddressReg(state, isn);
		//lw x8, 0(x9)
		state.emit(rvc(rv64::C_LW, Tmp2Reg + OffsetXC, Tmp1Reg + OffsetXC));
		//lw x9, 4(x9)
		state.emit(rvc(rv64::C_LW, Tmp2Reg + OffsetXC, 16 + Tmp2Reg + OffsetXC));
		//fcvt.d.w f24, x8
		state.emit(rvi(rv64::FCVT_D_W, Tmp1RegF, Tmp1Reg));
		//fcvt.d.w f25, x9
		state.emit(rvi(rv64::FCVT_D_W, Tmp2RegF, Tmp2Reg));
		//fmv.x.d x8, f24
		state.emit(rvi(rv64::FMV_X_D, Tmp1Reg, Tmp1RegF));
		//fmv.x.d x9, f25
		state.emit(rvi(rv64::FMV_X_D, Tmp2Reg, Tmp2RegF));
		//c.and x8, x13
		state.emit(rvc(rv64::C_AND, Tmp1Reg + OffsetXC, MaskEclear + OffsetXC));
		//c.and x9, x13
		state.emit(rvc(rv64::C_AND, Tmp2Reg + OffsetXC, MaskEclear + OffsetXC));
		//c.or x8, x14
		state.emit(rvc(rv64::C_OR, Tmp1Reg + OffsetXC, MaskEsetLo + OffsetXC));
		//c.or x9, x15
		state.emit(rvc(rv64::C_OR, Tmp2Reg + OffsetXC, MaskEsetHi + OffsetXC));
		//fmv.d.x f24, x8
		state.emit(rvi(rv64::FMV_D_X, Tmp1RegF, Tmp1Reg));
		//fmv.d.x f25, x9
		state.emit(rvi(rv64::FMV_D_X, Tmp2RegF, Tmp2Reg));
		//fdiv.d f{dst_lo}, f{dst_lo}, f24
		state.emit(rvi(rv64::FDIV_D, regLoE(isn.dst), regLoE(isn.dst), Tmp1RegF));
		//fdiv.d f{dst_hi}, f{dst_hi}, f25
		state.emit(rvi(rv64::FDIV_D, regHiE(isn.dst), regHiE(isn.dst), Tmp2RegF));
	}

	void JitCompilerRV64::v1_FSQRT_R(HANDLER_ARGS) {
		isn.dst %= RegisterCountFlt;
		//fsqrt.d f{dst_lo}, f{dst_lo}
		state.emit(rvi(rv64::FSQRT_D, regLoE(isn.dst), regLoE(isn.dst)));
		//fsqrt.d f{dst_hi}, f{dst_hi}
		state.emit(rvi(rv64::FSQRT_D, regHiE(isn.dst), regHiE(isn.dst)));
	}

	void JitCompilerRV64::v1_CBRANCH(HANDLER_ARGS) {
		int reg = isn.dst;
		int target = state.registerUsage[reg] + 1;
		int shift = isn.getModCond() + RandomX_ConfigurationBase::JumpOffset;
		int32_t imm = unsigned32ToSigned2sCompl(isn.getImm32());
		imm |= (1UL << shift);
		if (RandomX_ConfigurationBase::JumpOffset > 0 || shift > 0)
			imm &= ~(1UL << (shift - 1));
		//x8 = branchMask
		emitImm32(state, (int32_t)((1 << RandomX_ConfigurationBase::JumpBits) - 1) << shift, Tmp1Reg);
		//x{dst} += {imm}
		emitImm32(state, imm, regR(isn.dst), regR(isn.dst), Tmp2Reg);
		//and x8, x8, x{dst}
		state.emit(rvi(rv64::AND, Tmp1Reg, Tmp1Reg, regR(isn.dst)));
		int32_t targetPos = state.instructionOffsets[target];
		int offset = targetPos - state.codePos;
		if (offset >= -256) { //C.BEQZ only has a range of 256B
			//c.beqz x8, {offset}
			int imm8 = 1; //sign bit is always 1
			int imm21 = offset & 6; //offset[2:1]
			int imm5 = (offset >> 5) & 1; //offset[5]
			int imm43 = offset & 24; //offset[4:3]
			int imm76 = (offset >> 3) & 24; //offset[7:6]
			state.emit(rvc(rv64::C_BEQZ, imm8, imm43 + (Tmp1Reg + OffsetXC), imm76 + imm21 + imm5));
		}
		else if (offset >= -4096) { //BEQ only has a range of 4KB
			//beq x8, x0, offset
			int imm12 = 1 << 11; //sign bit is always 1
			int imm105 = offset & 2016; //offset[10:5]
			int imm41 = offset & 30; //offset[4:1]
			int imm11 = (offset >> 11) & 1; //offset[11]
			state.emit(rvi(rv64::BEQ, imm41 + imm11, Tmp1Reg, imm12 + imm105));
		}
		else {
			//c.bnez x8, +6
			state.emit(rvc(rv64::C_BNEZ, Tmp1Reg + OffsetXC, 6));
			//j targetPos
			emitJump(state, 0, state.codePos, targetPos);
			state.codePos += 4;
		}
		//mark all registers as used
		for (unsigned j = 0; j < RegistersCount; ++j) {
			state.registerUsage[j] = i;
		}
	}

	void JitCompilerRV64::v1_CFROUND(HANDLER_ARGS) {
		int32_t imm = (isn.getImm32() - 2) & 63; //-2 to avoid a later left shift to multiply by 4
		if (imm != 0) {
#ifdef __riscv_zbb
			//rori x8, x{src}, {imm}
			state.emit(rvi(rv64::RORI, Tmp1Reg, regR(isn.src), imm));
#else
			int32_t imml = -imm & 63;
			//srli x8, x{src}, {imm}
			state.emit(rvi(rv64::SRLI, Tmp1Reg, regR(isn.src), imm));
			//slli x9, x{src}, {imml}
			state.emit(rvi(rv64::SLLI, Tmp2Reg, regR(isn.src), imml));
			//c.or x8, x9
			state.emit(rvc(rv64::C_OR, Tmp1Reg + OffsetXC, Tmp2Reg + OffsetXC));
#endif
			//c.andi x8, 12
			state.emit(rvc(rv64::C_ANDI, Tmp1Reg + OffsetXC, 12));
		}
		else {
			//and x8, x{src}, 12
			state.emit(rvi(rv64::ANDI, Tmp1Reg, regR(isn.src), 12));
		}
		//c.add x8, x3
		state.emit(rvc(rv64::C_ADD, Tmp1Reg, LiteralPoolReg));
		//c.lw x8, 64(x8)
		state.emit(rvc(rv64::C_LW, Tmp1Reg + OffsetXC, 8 + Tmp1Reg + OffsetXC));
		//fsrm x8
		state.emit(rvi(rv64::FSRM, 0, Tmp1Reg, 0));
	}

	void JitCompilerRV64::v1_ISTORE(HANDLER_ARGS) {
		genAddressRegDst(state, isn);
		//sd x{src}, 0(x9)
		state.emit(rvi(rv64::SD, 0, Tmp2Reg, regR(isn.src)));
	}

	void JitCompilerRV64::v1_NOP(HANDLER_ARGS) {
	}

InstructionGeneratorRV64 JitCompilerRV64::engine[256] = {};
}
