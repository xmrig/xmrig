/*
Copyright (c) 2018-2020, tevador    <tevador@gmail.com>
Copyright (c) 2019-2020, SChernykh  <https://github.com/SChernykh>
Copyright (c) 2019-2020, XMRig      <https://github.com/xmrig>, <support@xmrig.com>

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

#include "crypto/randomx/jit_compiler_a64.hpp"
#include "crypto/common/VirtualMemory.h"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/reciprocal.h"
#include "crypto/randomx/superscalar.hpp"
#include "crypto/randomx/virtual_memory.hpp"

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

namespace ARMV8A {

constexpr uint32_t B           = 0x14000000;
constexpr uint32_t EOR         = 0xCA000000;
constexpr uint32_t EOR32       = 0x4A000000;
constexpr uint32_t ADD         = 0x8B000000;
constexpr uint32_t SUB         = 0xCB000000;
constexpr uint32_t MUL         = 0x9B007C00;
constexpr uint32_t UMULH       = 0x9BC07C00;
constexpr uint32_t SMULH       = 0x9B407C00;
constexpr uint32_t MOVZ        = 0xD2800000;
constexpr uint32_t MOVN        = 0x92800000;
constexpr uint32_t MOVK        = 0xF2800000;
constexpr uint32_t ADD_IMM_LO  = 0x91000000;
constexpr uint32_t ADD_IMM_HI  = 0x91400000;
constexpr uint32_t LDR_LITERAL = 0x58000000;
constexpr uint32_t ROR         = 0x9AC02C00;
constexpr uint32_t ROR_IMM     = 0x93C00000;
constexpr uint32_t MOV_REG     = 0xAA0003E0;
constexpr uint32_t MOV_VREG_EL = 0x6E080400;
constexpr uint32_t FADD        = 0x4E60D400;
constexpr uint32_t FSUB        = 0x4EE0D400;
constexpr uint32_t FEOR        = 0x6E201C00;
constexpr uint32_t FMUL        = 0x6E60DC00;
constexpr uint32_t FDIV        = 0x6E60FC00;
constexpr uint32_t FSQRT       = 0x6EE1F800;

}

namespace randomx {

static const size_t CodeSize = ((uint8_t*)randomx_init_dataset_aarch64_end) - ((uint8_t*)randomx_program_aarch64);
static const size_t MainLoopBegin = ((uint8_t*)randomx_program_aarch64_main_loop) - ((uint8_t*)randomx_program_aarch64);
static const size_t PrologueSize = ((uint8_t*)randomx_program_aarch64_vm_instructions) - ((uint8_t*)randomx_program_aarch64);
static const size_t ImulRcpLiteralsEnd = ((uint8_t*)randomx_program_aarch64_imul_rcp_literals_end) - ((uint8_t*)randomx_program_aarch64);

static size_t CalcDatasetItemSize()
{
	return
	// Prologue
	((uint8_t*)randomx_calc_dataset_item_aarch64_prefetch - (uint8_t*)randomx_calc_dataset_item_aarch64) +
	// Main loop
	RandomX_CurrentConfig.CacheAccesses * (
		// Main loop prologue
		((uint8_t*)randomx_calc_dataset_item_aarch64_mix - ((uint8_t*)randomx_calc_dataset_item_aarch64_prefetch)) + 4 +
		// Inner main loop (instructions)
		((RandomX_ConfigurationBase::SuperscalarLatency * 3) + 2) * 16 +
		// Main loop epilogue
		((uint8_t*)randomx_calc_dataset_item_aarch64_store_result - (uint8_t*)randomx_calc_dataset_item_aarch64_mix) + 4
	) +
	// Epilogue
	((uint8_t*)randomx_calc_dataset_item_aarch64_end - (uint8_t*)randomx_calc_dataset_item_aarch64_store_result);
}

constexpr uint32_t IntRegMap[8] = { 4, 5, 6, 7, 12, 13, 14, 15 };

JitCompilerA64::JitCompilerA64(bool hugePagesEnable, bool) :
	hugePages(hugePagesJIT && hugePagesEnable),
	literalPos(ImulRcpLiteralsEnd)
{
}

JitCompilerA64::~JitCompilerA64()
{
	freePagedMemory(code, allocatedSize);
}

void JitCompilerA64::generateProgram(Program& program, ProgramConfiguration& config, uint32_t)
{
	if (!allocatedSize) {
		allocate(CodeSize);
	}
#ifdef XMRIG_SECURE_JIT
	else {
		enableWriting();
	}
#endif

	uint32_t codePos = MainLoopBegin + 4;

	// and w16, w10, ScratchpadL3Mask64
	emit32(0x121A0000 | 16 | (10 << 5) | ((RandomX_CurrentConfig.Log2_ScratchpadL3 - 7) << 10), code, codePos);

	// and w17, w20, ScratchpadL3Mask64
	emit32(0x121A0000 | 17 | (20 << 5) | ((RandomX_CurrentConfig.Log2_ScratchpadL3 - 7) << 10), code, codePos);

	codePos = PrologueSize;
	literalPos = ImulRcpLiteralsEnd;
	num32bitLiterals = 0;

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = codePos;

	for (uint32_t i = 0; i < program.getSize(); ++i)
	{
		Instruction& instr = program(i);
		(this->*engine[instr.opcode])(instr, codePos);
	}

	// Update spMix2
	// eor w20, config.readReg2, config.readReg3
	emit32(ARMV8A::EOR32 | 20 | (IntRegMap[config.readReg2] << 5) | (IntRegMap[config.readReg3] << 16), code, codePos);

	// Jump back to the main loop
	const uint32_t offset = (((uint8_t*)randomx_program_aarch64_vm_instructions_end) - ((uint8_t*)randomx_program_aarch64)) - codePos;
	emit32(ARMV8A::B | (offset / 4), code, codePos);

	// and w20, w20, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_aarch64_cacheline_align_mask1) - ((uint8_t*)randomx_program_aarch64));
	emit32(0x121A0000 | 20 | (20 << 5) | ((RandomX_CurrentConfig.Log2_DatasetBaseSize - 7) << 10), code, codePos);

	// and w10, w10, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_aarch64_cacheline_align_mask2) - ((uint8_t*)randomx_program_aarch64));
	emit32(0x121A0000 | 10 | (10 << 5) | ((RandomX_CurrentConfig.Log2_DatasetBaseSize - 7) << 10), code, codePos);

	// Update spMix1
	// eor x10, config.readReg0, config.readReg1
	codePos = ((uint8_t*)randomx_program_aarch64_update_spMix1) - ((uint8_t*)randomx_program_aarch64);
	emit32(ARMV8A::EOR | 10 | (IntRegMap[config.readReg0] << 5) | (IntRegMap[config.readReg1] << 16), code, codePos);

#	ifndef XMRIG_OS_APPLE
	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code + MainLoopBegin), codePos - MainLoopBegin);
#	endif
}

void JitCompilerA64::generateProgramLight(Program& program, ProgramConfiguration& config, uint32_t datasetOffset)
{
	if (!allocatedSize) {
		allocate(CodeSize);
	}
#ifdef XMRIG_SECURE_JIT
	else {
		enableWriting();
	}
#endif

	uint32_t codePos = MainLoopBegin + 4;

	// and w16, w10, ScratchpadL3Mask64
	emit32(0x121A0000 | 16 | (10 << 5) | ((RandomX_CurrentConfig.Log2_ScratchpadL3 - 7) << 10), code, codePos);

	// and w17, w20, ScratchpadL3Mask64
	emit32(0x121A0000 | 17 | (20 << 5) | ((RandomX_CurrentConfig.Log2_ScratchpadL3 - 7) << 10), code, codePos);

	codePos = PrologueSize;
	literalPos = ImulRcpLiteralsEnd;
	num32bitLiterals = 0;

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = codePos;

	for (uint32_t i = 0; i < program.getSize(); ++i)
	{
		Instruction& instr = program(i);
		(this->*engine[instr.opcode])(instr, codePos);
	}

	// Update spMix2
	// eor w20, config.readReg2, config.readReg3
	emit32(ARMV8A::EOR32 | 20 | (IntRegMap[config.readReg2] << 5) | (IntRegMap[config.readReg3] << 16), code, codePos);

	// Jump back to the main loop
	const uint32_t offset = (((uint8_t*)randomx_program_aarch64_vm_instructions_end_light) - ((uint8_t*)randomx_program_aarch64)) - codePos;
	emit32(ARMV8A::B | (offset / 4), code, codePos);

	// and w2, w9, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_aarch64_light_cacheline_align_mask) - ((uint8_t*)randomx_program_aarch64));
	emit32(0x121A0000 | 2 | (9 << 5) | ((RandomX_CurrentConfig.Log2_DatasetBaseSize - 7) << 10), code, codePos);

	// Update spMix1
	// eor x10, config.readReg0, config.readReg1
	codePos = ((uint8_t*)randomx_program_aarch64_update_spMix1) - ((uint8_t*)randomx_program_aarch64);
	emit32(ARMV8A::EOR | 10 | (IntRegMap[config.readReg0] << 5) | (IntRegMap[config.readReg1] << 16), code, codePos);

	// Apply dataset offset
	codePos = ((uint8_t*)randomx_program_aarch64_light_dataset_offset) - ((uint8_t*)randomx_program_aarch64);

	datasetOffset /= CacheLineSize;
	const uint32_t imm_lo = datasetOffset & ((1 << 12) - 1);
	const uint32_t imm_hi = datasetOffset >> 12;

	emit32(ARMV8A::ADD_IMM_LO | 2 | (2 << 5) | (imm_lo << 10), code, codePos);
	emit32(ARMV8A::ADD_IMM_HI | 2 | (2 << 5) | (imm_hi << 10), code, codePos);

#	ifndef XMRIG_OS_APPLE
	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code + MainLoopBegin), codePos - MainLoopBegin);
#	endif
}

template<size_t N>
void JitCompilerA64::generateSuperscalarHash(SuperscalarProgram(&programs)[N])
{
	if (!allocatedSize) {
		allocate(CodeSize + CalcDatasetItemSize());
	}
#ifdef XMRIG_SECURE_JIT
	else {
		enableWriting();
	}
#endif

	uint32_t codePos = CodeSize;

	uint8_t* p1 = (uint8_t*)randomx_calc_dataset_item_aarch64;
	uint8_t* p2 = (uint8_t*)randomx_calc_dataset_item_aarch64_prefetch;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

	num32bitLiterals = 64;
	constexpr uint32_t tmp_reg = 12;

	for (size_t i = 0; i < RandomX_CurrentConfig.CacheAccesses; ++i)
	{
		// and x11, x10, CacheSize / CacheLineSize - 1
		emit32(0x92400000 | 11 | (10 << 5) | ((RandomX_CurrentConfig.Log2_CacheSize - 1) << 10), code, codePos);

		p1 = ((uint8_t*)randomx_calc_dataset_item_aarch64_prefetch) + 4;
		p2 = (uint8_t*)randomx_calc_dataset_item_aarch64_mix;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		SuperscalarProgram& prog = programs[i];
		const size_t progSize = prog.getSize();

		uint32_t jmp_pos = codePos;
		codePos += 4;

		// Fill in literal pool
		for (size_t j = 0; j < progSize; ++j)
		{
			const Instruction& instr = prog(j);
			if (static_cast<SuperscalarInstructionType>(instr.opcode) == randomx::SuperscalarInstructionType::IMUL_RCP)
				emit64(randomx_reciprocal(instr.getImm32()), code, codePos);
		}

		// Jump over literal pool
		uint32_t literal_pos = jmp_pos;
		emit32(ARMV8A::B | ((codePos - jmp_pos) / 4), code, literal_pos);

		for (size_t j = 0; j < progSize; ++j)
		{
			const Instruction& instr = prog(j);
			const uint32_t src = instr.src;
			const uint32_t dst = instr.dst;

			switch (static_cast<SuperscalarInstructionType>(instr.opcode))
			{
			case randomx::SuperscalarInstructionType::ISUB_R:
				emit32(ARMV8A::SUB | dst | (dst << 5) | (src << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_R:
				emit32(ARMV8A::EOR | dst | (dst << 5) | (src << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IADD_RS:
				emit32(ARMV8A::ADD | dst | (dst << 5) | (instr.getModShift() << 10) | (src << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMUL_R:
				emit32(ARMV8A::MUL | dst | (dst << 5) | (src << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IROR_C:
				emit32(ARMV8A::ROR_IMM | dst | (dst << 5) | ((instr.getImm32() & 63) << 10) | (dst << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IADD_C7:
			case randomx::SuperscalarInstructionType::IADD_C8:
			case randomx::SuperscalarInstructionType::IADD_C9:
				emitAddImmediate(dst, dst, instr.getImm32(), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_C7:
			case randomx::SuperscalarInstructionType::IXOR_C8:
			case randomx::SuperscalarInstructionType::IXOR_C9:
				emitMovImmediate(tmp_reg, instr.getImm32(), code, codePos);
				emit32(ARMV8A::EOR | dst | (dst << 5) | (tmp_reg << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMULH_R:
				emit32(ARMV8A::UMULH | dst | (dst << 5) | (src << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::ISMULH_R:
				emit32(ARMV8A::SMULH | dst | (dst << 5) | (src << 16), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMUL_RCP:
				{
					int32_t offset = (literal_pos - codePos) / 4;
					offset &= (1 << 19) - 1;
					literal_pos += 8;

					// ldr tmp_reg, reciprocal
					emit32(ARMV8A::LDR_LITERAL | tmp_reg | (offset << 5), code, codePos);

					// mul dst, dst, tmp_reg
					emit32(ARMV8A::MUL | dst | (dst << 5) | (tmp_reg << 16), code, codePos);
				}
				break;
			default:
				break;
			}
		}

		p1 = (uint8_t*)randomx_calc_dataset_item_aarch64_mix;
		p2 = (uint8_t*)randomx_calc_dataset_item_aarch64_store_result;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		// Update registerValue
		emit32(ARMV8A::MOV_REG | 10 | (prog.getAddressRegister() << 16), code, codePos);
	}

	p1 = (uint8_t*)randomx_calc_dataset_item_aarch64_store_result;
	p2 = (uint8_t*)randomx_calc_dataset_item_aarch64_end;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

#	ifndef XMRIG_OS_APPLE
	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code + CodeSize), codePos - MainLoopBegin);
#	endif
}

template void JitCompilerA64::generateSuperscalarHash(SuperscalarProgram(&programs)[RANDOMX_CACHE_MAX_ACCESSES]);

DatasetInitFunc* JitCompilerA64::getDatasetInitFunc() const
{
#	ifdef XMRIG_SECURE_JIT
	enableExecution();
#	endif

	return (DatasetInitFunc*)(code + (((uint8_t*)randomx_init_dataset_aarch64) - ((uint8_t*)randomx_program_aarch64)));
}

size_t JitCompilerA64::getCodeSize()
{
	return CodeSize;
}

void JitCompilerA64::enableWriting() const
{
	xmrig::VirtualMemory::protectRW(code, allocatedSize);
}

void JitCompilerA64::enableExecution() const
{
	xmrig::VirtualMemory::protectRX(code, allocatedSize);
}


void JitCompilerA64::allocate(size_t size)
{
	allocatedSize = size;
	code = static_cast<uint8_t*>(allocExecutableMemory(allocatedSize, hugePages));

	memcpy(code, reinterpret_cast<const void *>(randomx_program_aarch64), CodeSize);

#	ifndef XMRIG_OS_APPLE
	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code), CodeSize);
#	endif
}


void JitCompilerA64::emitMovImmediate(uint32_t dst, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	if (imm < (1 << 16))
	{
		// movz tmp_reg, imm32 (16 low bits)
		emit32(ARMV8A::MOVZ | dst | (imm << 5), code, k);
	}
	else
	{
		if (num32bitLiterals < 64)
		{
			if (static_cast<int32_t>(imm) < 0)
			{
				// smov dst, vN.s[M]
				emit32(0x4E042C00 | dst | ((num32bitLiterals / 4) << 5) | ((num32bitLiterals % 4) << 19), code, k);
			}
			else
			{
				// umov dst, vN.s[M]
				emit32(0x0E043C00 | dst | ((num32bitLiterals / 4) << 5) | ((num32bitLiterals % 4) << 19), code, k);
			}

			((uint32_t*)(code + ImulRcpLiteralsEnd))[num32bitLiterals] = imm;
			++num32bitLiterals;
		}
		else
		{
			if (static_cast<int32_t>(imm) < 0)
			{
				// movn tmp_reg, ~imm32 (16 high bits)
				emit32(ARMV8A::MOVN | dst | (1 << 21) | ((~imm >> 16) << 5), code, k);
			}
			else
			{
				// movz tmp_reg, imm32 (16 high bits)
				emit32(ARMV8A::MOVZ | dst | (1 << 21) | ((imm >> 16) << 5), code, k);
			}

			// movk tmp_reg, imm32 (16 low bits)
			emit32(ARMV8A::MOVK | dst | ((imm & 0xFFFF) << 5), code, k);
		}
	}

	codePos = k;
}

void JitCompilerA64::emitAddImmediate(uint32_t dst, uint32_t src, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	if (imm < (1 << 24))
	{
		const uint32_t imm_lo = imm & ((1 << 12) - 1);
		const uint32_t imm_hi = imm >> 12;

		if (imm_lo && imm_hi)
		{
			emit32(ARMV8A::ADD_IMM_LO | dst | (src << 5) | (imm_lo << 10), code, k);
			emit32(ARMV8A::ADD_IMM_HI | dst | (dst << 5) | (imm_hi << 10), code, k);
		}
		else if (imm_lo)
		{
			emit32(ARMV8A::ADD_IMM_LO | dst | (src << 5) | (imm_lo << 10), code, k);
		}
		else
		{
			emit32(ARMV8A::ADD_IMM_HI | dst | (src << 5) | (imm_hi << 10), code, k);
		}
	}
	else
	{
		constexpr uint32_t tmp_reg = 20;
		emitMovImmediate(tmp_reg, imm, code, k);

		// add dst, src, tmp_reg
		emit32(ARMV8A::ADD | dst | (src << 5) | (tmp_reg << 16), code, k);
	}

	codePos = k;
}

template<uint32_t tmp_reg>
void JitCompilerA64::emitMemLoad(uint32_t dst, uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t imm = instr.getImm32();

	if (src != dst)
	{
		imm &= instr.getModMem() ? (RandomX_CurrentConfig.ScratchpadL1_Size - 1) : (RandomX_CurrentConfig.ScratchpadL2_Size - 1);
		emitAddImmediate(tmp_reg, src, imm, code, k);

		constexpr uint32_t t = 0x927d0000 | tmp_reg | (tmp_reg << 5);
		const uint32_t andInstrL1 = t | ((RandomX_CurrentConfig.Log2_ScratchpadL1 - 4) << 10);
		const uint32_t andInstrL2 = t | ((RandomX_CurrentConfig.Log2_ScratchpadL2 - 4) << 10);

		emit32(instr.getModMem() ? andInstrL1 : andInstrL2, code, k);

		// ldr tmp_reg, [x2, tmp_reg]
		emit32(0xf8606840 | tmp_reg | (tmp_reg << 16), code, k);
	}
	else
	{
		imm = (imm & ScratchpadL3Mask) >> 3;
		emitMovImmediate(tmp_reg, imm, code, k);

		// ldr tmp_reg, [x2, tmp_reg, lsl 3]
		emit32(0xf8607840 | tmp_reg | (tmp_reg << 16), code, k);
	}

	codePos = k;
}

template<uint32_t tmp_reg_fp>
void JitCompilerA64::emitMemLoadFP(uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t imm = instr.getImm32();
	constexpr uint32_t tmp_reg = 19;

	imm &= instr.getModMem() ? (RandomX_CurrentConfig.ScratchpadL1_Size - 1) : (RandomX_CurrentConfig.ScratchpadL2_Size - 1);
	emitAddImmediate(tmp_reg, src, imm, code, k);

	constexpr uint32_t t = 0x927d0000 | tmp_reg | (tmp_reg << 5);
	const uint32_t andInstrL1 = t | ((RandomX_CurrentConfig.Log2_ScratchpadL1 - 4) << 10);
	const uint32_t andInstrL2 = t | ((RandomX_CurrentConfig.Log2_ScratchpadL2 - 4) << 10);

	emit32(instr.getModMem() ? andInstrL1 : andInstrL2, code, k);

	// add tmp_reg, x2, tmp_reg
	emit32(ARMV8A::ADD | tmp_reg | (2 << 5) | (tmp_reg << 16), code, k);

	// ldpsw tmp_reg, tmp_reg + 1, [tmp_reg]
	emit32(0x69400000 | tmp_reg | (tmp_reg << 5) | ((tmp_reg + 1) << 10), code, k);

	// ins tmp_reg_fp.d[0], tmp_reg
	emit32(0x4E081C00 | tmp_reg_fp | (tmp_reg << 5), code, k);

	// ins tmp_reg_fp.d[1], tmp_reg + 1
	emit32(0x4E181C00 | tmp_reg_fp | ((tmp_reg + 1) << 5), code, k);

	// scvtf tmp_reg_fp.2d, tmp_reg_fp.2d
	emit32(0x4E61D800 | tmp_reg_fp | (tmp_reg_fp << 5), code, k);

	codePos = k;
}

void JitCompilerA64::h_IADD_RS(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	const uint32_t shift = instr.getModShift();

	// add dst, src << shift
	emit32(ARMV8A::ADD | dst | (dst << 5) | (shift << 10) | (src << 16), code, k);

	if (instr.dst == RegisterNeedsDisplacement)
		emitAddImmediate(dst, dst, instr.getImm32(), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IADD_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp_reg = 20;
	emitMemLoad<tmp_reg>(dst, src, instr, code, k);

	// add dst, dst, tmp_reg
	emit32(ARMV8A::ADD | dst | (dst << 5) | (tmp_reg << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_ISUB_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src != dst)
	{
		// sub dst, dst, src
		emit32(ARMV8A::SUB | dst | (dst << 5) | (src << 16), code, k);
	}
	else
	{
		emitAddImmediate(dst, dst, -instr.getImm32(), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_ISUB_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp_reg = 20;
	emitMemLoad<tmp_reg>(dst, src, instr, code, k);

	// sub dst, dst, tmp_reg
	emit32(ARMV8A::SUB | dst | (dst << 5) | (tmp_reg << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IMUL_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src == dst)
	{
		src = 20;
		emitMovImmediate(src, instr.getImm32(), code, k);
	}

	// mul dst, dst, src
	emit32(ARMV8A::MUL | dst | (dst << 5) | (src << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IMUL_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp_reg = 20;
	emitMemLoad<tmp_reg>(dst, src, instr, code, k);

	// sub dst, dst, tmp_reg
	emit32(ARMV8A::MUL | dst | (dst << 5) | (tmp_reg << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IMULH_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	// umulh dst, dst, src
	emit32(ARMV8A::UMULH | dst | (dst << 5) | (src << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IMULH_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp_reg = 20;
	emitMemLoad<tmp_reg>(dst, src, instr, code, k);

	// umulh dst, dst, tmp_reg
	emit32(ARMV8A::UMULH | dst | (dst << 5) | (tmp_reg << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_ISMULH_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	// smulh dst, dst, src
	emit32(ARMV8A::SMULH | dst | (dst << 5) | (src << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_ISMULH_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp_reg = 20;
	emitMemLoad<tmp_reg>(dst, src, instr, code, k);

	// smulh dst, dst, tmp_reg
	emit32(ARMV8A::SMULH | dst | (dst << 5) | (tmp_reg << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IMUL_RCP(Instruction& instr, uint32_t& codePos)
{
	const uint64_t divisor = instr.getImm32();
	if (isZeroOrPowerOf2(divisor))
		return;

	uint32_t k = codePos;

	constexpr uint32_t tmp_reg = 20;
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint64_t N = 1ULL << 63;
	const uint64_t q = N / divisor;
	const uint64_t r = N % divisor;
#ifdef __GNUC__
	const uint64_t shift = 64 - __builtin_clzll(divisor);
#else
	uint64_t shift = 32;
	for (uint64_t k = 1U << 31; (k & divisor) == 0; k >>= 1)
		--shift;
#endif

	const uint32_t literal_id = (ImulRcpLiteralsEnd - literalPos) / sizeof(uint64_t);

	literalPos -= sizeof(uint64_t);
	*(uint64_t*)(code + literalPos) = (q << shift) + ((r << shift) / divisor);

	if (literal_id < 12)
	{
		static constexpr uint32_t literal_regs[12] = { 30 << 16, 29 << 16, 28 << 16, 27 << 16, 26 << 16, 25 << 16, 24 << 16, 23 << 16, 22 << 16, 21 << 16, 11 << 16, 0 };

		// mul dst, dst, literal_reg
		emit32(ARMV8A::MUL | dst | (dst << 5) | literal_regs[literal_id], code, k);
	}
	else
	{
		// ldr tmp_reg, reciprocal
		const uint32_t offset = (literalPos - k) / 4;
		emit32(ARMV8A::LDR_LITERAL | tmp_reg | (offset << 5), code, k);

		// mul dst, dst, tmp_reg
		emit32(ARMV8A::MUL | dst | (dst << 5) | (tmp_reg << 16), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_INEG_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = IntRegMap[instr.dst];

	// sub dst, xzr, dst
	emit32(ARMV8A::SUB | dst | (31 << 5) | (dst << 16), code, codePos);

	reg_changed_offset[instr.dst] = codePos;
}

void JitCompilerA64::h_IXOR_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src == dst)
	{
		src = 20;
		emitMovImmediate(src, instr.getImm32(), code, k);
	}

	// eor dst, dst, src
	emit32(ARMV8A::EOR | dst | (dst << 5) | (src << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IXOR_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp_reg = 20;
	emitMemLoad<tmp_reg>(dst, src, instr, code, k);

	// eor dst, dst, tmp_reg
	emit32(ARMV8A::EOR | dst | (dst << 5) | (tmp_reg << 16), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_IROR_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src != dst)
	{
		// ror dst, dst, src
		emit32(ARMV8A::ROR | dst | (dst << 5) | (src << 16), code, codePos);
	}
	else
	{
		// ror dst, dst, imm
		emit32(ARMV8A::ROR_IMM | dst | (dst << 5) | ((instr.getImm32() & 63) << 10) | (dst << 16), code, codePos);
	}

	reg_changed_offset[instr.dst] = codePos;
}

void JitCompilerA64::h_IROL_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src != dst)
	{
		constexpr uint32_t tmp_reg = 20;

		// sub tmp_reg, xzr, src
		emit32(ARMV8A::SUB | tmp_reg | (31 << 5) | (src << 16), code, k);

		// ror dst, dst, tmp_reg
		emit32(ARMV8A::ROR | dst | (dst << 5) | (tmp_reg << 16), code, k);
	}
	else
	{
		// ror dst, dst, imm
		emit32(ARMV8A::ROR_IMM | dst | (dst << 5) | ((-instr.getImm32() & 63) << 10) | (dst << 16), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_ISWAP_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src == dst)
		return;

	uint32_t k = codePos;

	constexpr uint32_t tmp_reg = 20;
	emit32(ARMV8A::MOV_REG | tmp_reg | (dst << 16), code, k);
	emit32(ARMV8A::MOV_REG | dst | (src << 16), code, k);
	emit32(ARMV8A::MOV_REG | src | (tmp_reg << 16), code, k);

	reg_changed_offset[instr.src] = k;
	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerA64::h_FSWAP_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t dst = instr.dst + 16;

	constexpr uint32_t tmp_reg_fp = 28;
	constexpr uint32_t src_index1 = 1 << 14;
	constexpr uint32_t dst_index1 = 1 << 20;

	emit32(ARMV8A::MOV_VREG_EL | tmp_reg_fp | (dst << 5) | src_index1, code, k);
	emit32(ARMV8A::MOV_VREG_EL | dst | (dst << 5) | dst_index1, code, k);
	emit32(ARMV8A::MOV_VREG_EL | dst | (tmp_reg_fp << 5), code, k);

	codePos = k;
}

void JitCompilerA64::h_FADD_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = (instr.src % 4) + 24;
	const uint32_t dst = (instr.dst % 4) + 16;

	emit32(ARMV8A::FADD | dst | (dst << 5) | (src << 16), code, codePos);
}

void JitCompilerA64::h_FADD_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % 4) + 16;

	constexpr uint32_t tmp_reg_fp = 28;
	emitMemLoadFP<tmp_reg_fp>(src, instr, code, k);

	emit32(ARMV8A::FADD | dst | (dst << 5) | (tmp_reg_fp << 16), code, k);

	codePos = k;
}

void JitCompilerA64::h_FSUB_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = (instr.src % 4) + 24;
	const uint32_t dst = (instr.dst % 4) + 16;

	emit32(ARMV8A::FSUB | dst | (dst << 5) | (src << 16), code, codePos);
}

void JitCompilerA64::h_FSUB_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % 4) + 16;

	constexpr uint32_t tmp_reg_fp = 28;
	emitMemLoadFP<tmp_reg_fp>(src, instr, code, k);

	emit32(ARMV8A::FSUB | dst | (dst << 5) | (tmp_reg_fp << 16), code, k);

	codePos = k;
}

void JitCompilerA64::h_FSCAL_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = (instr.dst % 4) + 16;

	emit32(ARMV8A::FEOR | dst | (dst << 5) | (31 << 16), code, codePos);
}

void JitCompilerA64::h_FMUL_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = (instr.src % 4) + 24;
	const uint32_t dst = (instr.dst % 4) + 20;

	emit32(ARMV8A::FMUL | dst | (dst << 5) | (src << 16), code, codePos);
}

void JitCompilerA64::h_FDIV_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % 4) + 20;

	constexpr uint32_t tmp_reg_fp = 28;
	emitMemLoadFP<tmp_reg_fp>(src, instr, code, k);

	// and tmp_reg_fp, tmp_reg_fp, and_mask_reg
	emit32(0x4E201C00 | tmp_reg_fp | (tmp_reg_fp << 5) | (29 << 16), code, k);

	// orr tmp_reg_fp, tmp_reg_fp, or_mask_reg
	emit32(0x4EA01C00 | tmp_reg_fp | (tmp_reg_fp << 5) | (30 << 16), code, k);

	emit32(ARMV8A::FDIV | dst | (dst << 5) | (tmp_reg_fp << 16), code, k);

	codePos = k;
}

void JitCompilerA64::h_FSQRT_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = (instr.dst % 4) + 20;

	emit32(ARMV8A::FSQRT | dst | (dst << 5), code, codePos);
}

void JitCompilerA64::h_CBRANCH(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t dst = IntRegMap[instr.dst];
	const uint32_t modCond = instr.getModCond();
	const uint32_t shift = modCond + RandomX_ConfigurationBase::JumpOffset;
	const uint32_t imm = (instr.getImm32() | (1U << shift)) & ~(1U << (shift - 1));

	emitAddImmediate(dst, dst, imm, code, k);

	// tst dst, mask
	emit32((0xF2781C1F - (modCond << 16)) | (dst << 5), code, k);

	int32_t offset = reg_changed_offset[instr.dst];
	offset = ((offset - k) >> 2) & ((1 << 19) - 1);

	// beq target
	emit32(0x54000000 | (offset << 5), code, k);

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = k;

	codePos = k;
}

void JitCompilerA64::h_CFROUND(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];

	constexpr uint32_t tmp_reg = 20;
	constexpr uint32_t fpcr_tmp_reg = 8;

	// ror tmp_reg, src, imm
	emit32(ARMV8A::ROR_IMM | tmp_reg | (src << 5) | ((instr.getImm32() & 63) << 10) | (src << 16), code, k);

	// bfi fpcr_tmp_reg, tmp_reg, 40, 2
	emit32(0xB3580400 | fpcr_tmp_reg | (tmp_reg << 5), code, k);

	// rbit tmp_reg, fpcr_tmp_reg
	emit32(0xDAC00000 | tmp_reg | (fpcr_tmp_reg << 5), code, k);

	// msr fpcr, tmp_reg
	emit32(0xD51B4400 | tmp_reg, code, k);

	codePos = k;
}

void JitCompilerA64::h_ISTORE(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	constexpr uint32_t tmp_reg = 20;

	uint32_t imm = instr.getImm32();

	if (instr.getModCond() < StoreL3Condition)
		imm &= instr.getModMem() ? (RandomX_CurrentConfig.ScratchpadL1_Size - 1) : (RandomX_CurrentConfig.ScratchpadL2_Size - 1);
	else
		imm &= RandomX_CurrentConfig.ScratchpadL3_Size - 1;

	emitAddImmediate(tmp_reg, dst, imm, code, k);

	constexpr uint32_t t = 0x927d0000 | tmp_reg | (tmp_reg << 5);
	const uint32_t andInstrL1 = t | ((RandomX_CurrentConfig.Log2_ScratchpadL1 - 4) << 10);
	const uint32_t andInstrL2 = t | ((RandomX_CurrentConfig.Log2_ScratchpadL2 - 4) << 10);
	const uint32_t andInstrL3 = t | ((RandomX_CurrentConfig.Log2_ScratchpadL3 - 4) << 10);

	emit32((instr.getModCond() < StoreL3Condition) ? (instr.getModMem() ? andInstrL1 : andInstrL2) : andInstrL3, code, k);

	// str src, [x2, tmp_reg]
	emit32(0xF8206840 | src | (tmp_reg << 16), code, k);

	codePos = k;
}

void JitCompilerA64::h_NOP(Instruction& instr, uint32_t& codePos)
{
}

InstructionGeneratorA64 JitCompilerA64::engine[257] = {};

}
