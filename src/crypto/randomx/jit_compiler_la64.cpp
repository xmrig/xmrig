/*
Copyright (c) 2018-2020, tevador    <tevador@gmail.com>
Copyright (c) 2019-2020, SChernykh  <https://github.com/SChernykh>
Copyright (c) 2019-2020, XMRig      <https://github.com/xmrig>, <support@xmrig.com>
Copyright (c) 2024, RESBI <https://github.com/RESBI>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above
	  copyright notice, this list of conditions and the following disclaimer
	  in the documentation and/or other materials provided with the
	  distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * LoongArch 64 backend, LSX only.
 *
 * The register allocation of jit_compiler_la64_static.S is the contract:
 *
 *   $r4  -> pointer to the register file
 *   $r5  -> pointer to mem, then to the dataset
 *   $r6  -> pointer to the scratchpad
 *   $r7  -> program iteration counter
 *   $r8-$r15   -> "r0"-"r7", see IntRegMap
 *   $r16 -> spAddr0
 *   $r17 -> spAddr1
 *   $r18 -> shadow of $fcsr0, written by h_CFROUND
 *   $r19 -> mx, ma
 *   $r20 -> spMix1
 *   $r21, $r22, $r23 -> temporaries
 *
 *   $vr0-$vr3   -> "f0"-"f3"
 *   $vr4-$vr7   -> "e0"-"e3"
 *   $vr8-$vr11  -> "a0"-"a3"
 *   $vr12       -> temporary
 *   $vr13       -> E "and" mask
 *   $vr14       -> E "or" mask
 *   $vr15       -> scale mask
 *
 * The last instruction of both generated programs writes
 * $r22 = r[readReg2] ^ r[readReg3], which the static code
 * after randomx_program_la64_vm_instructions_end reads.
 *
 * Unlike AArch64 there is no literal pool: every constant is
 * materialized in place, therefore one superscalar instruction
 * takes 20 bytes at most and CalcDatasetItemSize() counts 20.
 *
 * All bases were measured with the ruler in
 * scripts/140-la64-jit-probe.S, see logs/141-jit-probe.txt.
 * Two things were measured before they entered this file:
 * scripts/144-la64-fcsr-probe.c (see logs/145-la64-fcsr.txt)
 * for the $fcsr0 rounding field, and
 * scripts/146-la64-swap-probe.c (see logs/147-la64-swap.txt)
 * for the LSX swap of group F, which is the same
 * shuffle that intrin_portable.h asks for.
 */

#include <cstring>
#include "crypto/randomx/jit_compiler_la64.hpp"
#include "crypto/common/VirtualMemory.h"
#include "crypto/randomx/common.hpp"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/randomx.h"
#include "crypto/randomx/reciprocal.h"
#include "crypto/randomx/soft_aes.h"
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

namespace LA64 {

/* The scalar shape is  base | rk << 10 | rj << 5 | rd. */
constexpr uint32_t ADD_D      = 0x00108000;
constexpr uint32_t SUB_D      = 0x00118000;
constexpr uint32_t MUL_D      = 0x001d8000;
constexpr uint32_t MULH_D     = 0x001e0000;
constexpr uint32_t MULH_DU    = 0x001e8000;
constexpr uint32_t AND        = 0x00148000;
constexpr uint32_t OR         = 0x00150000;
constexpr uint32_t XOR        = 0x00158000;
constexpr uint32_t ANDI       = 0x03400000;
constexpr uint32_t ORI        = 0x03800000;
constexpr uint32_t ADDI_D     = 0x02c00000;
constexpr uint32_t SLLI_D     = 0x00410000;
constexpr uint32_t ROTR_D     = 0x001b8000;
constexpr uint32_t ROTRI_D    = 0x004d0000;
constexpr uint32_t BSTRPICK_D = 0x00c00000;
constexpr uint32_t BSTRINS_D  = 0x00800000;
constexpr uint32_t LU12I_W    = 0x14000000;
constexpr uint32_t LU32I_D    = 0x16000000;
constexpr uint32_t LU52I_D    = 0x03000000;
constexpr uint32_t LDX_D     = 0x380c0000;
constexpr uint32_t STX_D     = 0x381c0000;
constexpr uint32_t VLD        = 0x2c000000;
constexpr uint32_t VST        = 0x2c400000;
constexpr uint32_t B          = 0x50000000;
constexpr uint32_t BL         = 0x54000000;
constexpr uint32_t BEQ        = 0x58000000;
constexpr uint32_t BNE        = 0x5c000000;
constexpr uint32_t MOVGR2FCSR = 0x0114c000;
constexpr uint32_t MOVFCSR2GR = 0x0114c800;

/*
 * All of the vector words below were produced by the
 * assembler of this machine, see logs/141-jit-probe.txt.
 * The one exception is called out where it appears.
 */
constexpr uint32_t VAND_V      = 0x71260000;
constexpr uint32_t VOR_V       = 0x71268000;
constexpr uint32_t VXOR_V      = 0x71270000;
constexpr uint32_t VFADD_D     = 0x71310000;
constexpr uint32_t VFSUB_D     = 0x71330000;
constexpr uint32_t VFMUL_D     = 0x71390000;
constexpr uint32_t VFDIV_D     = 0x713b0000;
constexpr uint32_t VFSQRT_D    = 0x729ce800;
constexpr uint32_t VFFINTL_D_W = 0x729e1000;
constexpr uint32_t VPICKVE2GR_DU = 0x72f3f000;
constexpr uint32_t VINSGR2VR_D = 0x72ebf000;

/*
 * The $fcsr0 field that holds the floating point rounding
 * mode. Measured with scripts/144-la64-fcsr-probe.c, so the
 * JIT writes exactly the register state that
 * instructions_portable.cpp asks for with fesetround()
 * (see logs/145-la64-fcsr.txt):
 *
 *   fesetround(FE_TONEAREST)   -> $fcsr0 = 0x000
 *   fesetround(FE_DOWNWARD)    -> $fcsr0 = 0x300
 *   fesetround(FE_UPWARD)      -> $fcsr0 = 0x200
 *   fesetround(FE_TOWARDZERO)  -> $fcsr0 = 0x100
 *
 * and the interpreted virtual machine maps the low
 * two bits of the rotated value through
 * rx_set_rounding_mode(), that is
 *
 *   mode 0 -> FE_TONEAREST  -> field 0
 *   mode 1 -> FE_DOWNWARD   -> field 3
 *   mode 2 -> FE_UPWARD     -> field 2
 *   mode 3 -> FE_TOWARDZERO  -> field 1
 *
 * so the field is (-mode) & 3. This is the whole
 * rounding contract of h_CFROUND(...) below.
 */
constexpr uint32_t FCSR_ROUNDING_MSB = 9;
constexpr uint32_t FCSR_ROUNDING_LSB = 8;
constexpr uint32_t FCSR_ROUNDING_MASK = 60;

/* $r8-$r15 are "r0"-"r7". */
constexpr uint8_t IntRegMap[8] = { 8, 9, 10, 11, 12, 13, 14, 15 };

constexpr uint32_t RegScratchpad = 6;
constexpr uint32_t RegTemp = 21;
constexpr uint32_t RegSpMix2 = 22;
constexpr uint32_t RegMask = 23;
constexpr uint32_t RegTempFp = 12;

/*
 * The three vector register groups:
 * $vr0-$vr3 = f, $vr4-$vr7 = e, $vr8-$vr11 = a
 * (see the contract in jit_compiler_la64_static.S,
 * which loads group A once in the prologue and
 * groups F and E from the scratchpad in the
 * main loop). A mistake here is silent: the
 * generated code runs and merely computes
 * with another group.
 */
constexpr uint32_t RegGroupF = 0;
constexpr uint32_t RegGroupE = randomx::RegisterCountFlt;
constexpr uint32_t RegGroupA = 2 * randomx::RegisterCountFlt;

/* The shadow of $fcsr0, filled by the static prologue. */
constexpr uint32_t RegFcsrShadow = 18;

}

namespace randomx {

static const size_t ProgramSize = ((uint8_t*)randomx_init_dataset_la64_end) - ((uint8_t*)randomx_program_la64);
static const size_t MainLoopBegin = ((uint8_t*)randomx_program_la64_main_loop) - ((uint8_t*)randomx_program_la64);
static const size_t PrologueSize = ((uint8_t*)randomx_program_la64_vm_instructions) - ((uint8_t*)randomx_program_la64);

/* The offset of the two main loop mask slots. */
static const size_t MainLoopMask0 = ((uint8_t*)randomx_program_la64_main_loop_mask0) - ((uint8_t*)randomx_program_la64);
static const size_t MainLoopMask1 = ((uint8_t*)randomx_program_la64_main_loop_mask1) - ((uint8_t*)randomx_program_la64);

/*
 * Upper bound of the bytes taken by the dataset item
 * function. No literal pool is needed here, so one
 * superscalar instruction takes 20 bytes at most.
 */
static size_t CalcDatasetItemSize()
{
	return
	// prologue
	((uint8_t*)randomx_calc_dataset_item_la64_prefetch - (uint8_t*)randomx_calc_dataset_item_la64) +
	// one block per cache access
	RandomX_ConfigurationBase::CacheAccesses * (
		// the cache line mask word
		4 +
		// the prefetch code
		((uint8_t*)randomx_calc_dataset_item_la64_mix - ((uint8_t*)randomx_calc_dataset_item_la64_prefetch)) + 4 +
		// the generated superscalar program
		((RandomX_ConfigurationBase::SuperscalarMaxLatency * 3) + 2) * 20 +
		// the mix code and the new register value
		((uint8_t*)randomx_calc_dataset_item_la64_store_result - (uint8_t*)randomx_calc_dataset_item_la64_mix) + 4
	) +
	// epilogue
	((uint8_t*)randomx_calc_dataset_item_la64_end - (uint8_t*)randomx_calc_dataset_item_la64_store_result);
}

/*
 * Log2 of the cache size in bytes, which is
 * ArgonMemory * ArgonBlockSize (common.hpp) divided by
 * CacheLineSize. RandomX_ConfigurationBase does not
 * publish this value outside the AArch64 and RV64 builds,
 * so the dataset item generator derives the number of
 * cache line index bits itself and keeps it a constant.
 */
static uint32_t GetCacheLineIndexMsbd()
{
	uint32_t lines = (RandomX_ConfigurationBase::ArgonMemory * ArgonBlockSize) / CacheLineSize;
	uint32_t msbd = 0;

	while (lines > 1) {
		lines >>= 1;
		msbd++;
	}

	return msbd - 1;
}

JitCompilerLa64::JitCompilerLa64(bool hugePagesEnable, bool optimizedInitDatasetEnable) :
	hugePages(hugePagesJIT && hugePagesEnable),
	optimizedInitDataset(optimizedDatasetInit && optimizedInitDatasetEnable)
{
}

JitCompilerLa64::~JitCompilerLa64()
{
	freePagedMemory(code, allocatedSize);
}

void JitCompilerLa64::enableWriting() const
{
	xmrig::VirtualMemory::protectRW(code, allocatedSize);
}

void JitCompilerLa64::enableExecution() const
{
	xmrig::VirtualMemory::protectRX(code, allocatedSize);
}

void JitCompilerLa64::allocate(size_t size)
{
	allocatedSize = size;
	code = static_cast<uint8_t*>(allocExecutableMemory(allocatedSize, hugePages));

	memcpy(code, reinterpret_cast<const void*>(randomx_program_la64), ProgramSize);

	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code), ProgramSize);
}

DatasetInitFunc* JitCompilerLa64::getDatasetInitFunc() const
{
#	ifdef XMRIG_SECURE_JIT
	enableExecution();
#	endif

	return (DatasetInitFunc*)(code + (((uint8_t*)randomx_init_dataset_la64) - ((uint8_t*)randomx_program_la64)));
}

size_t JitCompilerLa64::getCodeSize()
{
	return ProgramSize;
}

/*
 * Three words of the fixed shape:
 *   lu12i.w $r23, <hi20>
 *   ori     $r23, $r23, <lo12>
 *   and     <dst>, <src>, $r23
 */
void JitCompilerLa64::emitMaskRegister(uint32_t dst, uint32_t src, uint32_t mask, uint8_t* code, uint32_t& codePos)
{
	emit32(LA64::LU12I_W | LA64::RegMask | (((mask >> 12) & 0xFFFFF) << 5), code, codePos);
	emit32(LA64::ORI | LA64::RegMask | (LA64::RegMask << 5) | ((mask & 0xFFF) << 10), code, codePos);
	emit32(LA64::AND | dst | (src << 5) | (LA64::RegMask << 10), code, codePos);
}

void JitCompilerLa64::generateProgram(Program& program, ProgramConfiguration& config, uint32_t flags)
{
	if (!allocatedSize) {
		allocate(ProgramSize);
	}
#ifdef XMRIG_SECURE_JIT
	else {
		enableWriting();
	}
#endif

	vm_flags = flags;

	// lu12i.w/ori/and $r16, $r20, ScratchpadL3Mask64
	uint32_t codePos = MainLoopMask0;
	emitMaskRegister(16, 20, RandomX_CurrentConfig.ScratchpadL3Mask64_Calculated, code, codePos);

	// lu12i.w/ori/and $r17, $r21, ScratchpadL3Mask64
	codePos = MainLoopMask1;
	emitMaskRegister(17, 21, RandomX_CurrentConfig.ScratchpadL3Mask64_Calculated, code, codePos);

	codePos = PrologueSize;

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = codePos;

	for (uint32_t i = 0; i < program.getSize(); ++i)
	{
		Instruction& instr = program(i);
		(this->*engine[instr.opcode])(instr, codePos);
	}

	// Update spMix2: xor $r22, $r<readReg2>, $r<readReg3>
	emit32(LA64::XOR | LA64::RegSpMix2 |
		(LA64::IntRegMap[config.readReg2] << 5) |
		(LA64::IntRegMap[config.readReg3] << 10), code, codePos);

	// Jump back to the static main loop
	// The branch offset lives in bits 10-25 as a word distance
	const uint32_t offset =
		(((uint8_t*)randomx_program_la64_vm_instructions_end) - ((uint8_t*)randomx_program_la64)) - codePos;
	emit32(LA64::B | (((offset / 4) & 0xFFFF) << 10), code, codePos);

	const uint32_t cacheLineAlignMask = RandomX_ConfigurationBase::CacheLineAlignMask_Calculated;

	// and $r22, $r22, CacheLineAlignMask
	codePos = ((uint8_t*)randomx_program_la64_cacheline_align_mask1) - ((uint8_t*)randomx_program_la64);
	emitMaskRegister(LA64::RegSpMix2, LA64::RegSpMix2, cacheLineAlignMask, code, codePos);

	// and $r21, $r21, CacheLineAlignMask
	codePos = ((uint8_t*)randomx_program_la64_cacheline_align_mask2) - ((uint8_t*)randomx_program_la64);
	emitMaskRegister(LA64::RegTemp, LA64::RegTemp, cacheLineAlignMask, code, codePos);

	// Update spMix1: xor $r20, $r<readReg0>, $r<readReg1>
	codePos = ((uint8_t*)randomx_program_la64_update_spMix1) - ((uint8_t*)randomx_program_la64);
	emit32(LA64::XOR | 20 |
		(LA64::IntRegMap[config.readReg0] << 5) |
		(LA64::IntRegMap[config.readReg1] << 10), code, codePos);

	/*
	 * RandomX v1 mixes group F with group E by a plain
	 * XOR, RandomX v2 uses the (soft) AES routines.
	 * LoongArch has no AES, so v2 always takes the
	 * soft AES path and v1 only has to branch back
	 * to the plain XOR mix.
	 */
	codePos = ((uint8_t*)randomx_program_la64_v2_FE_mix) - ((uint8_t*)randomx_program_la64);

	if (!RandomX_CurrentConfig.Tweak_V2_AES) {
		const uint32_t v1Offset =
			((uint8_t*)randomx_program_la64_v1_FE_mix) - ((uint8_t*)randomx_program_la64_v2_FE_mix);
		emit32(LA64::B | (((v1Offset / 4) & 0xFFFF) << 10), code, codePos);
	}

	// The soft AES routines read these two tables
	{
		const uint32_t luts =
			((uint8_t*)randomx_program_la64_aes_lut_pointers) - ((uint8_t*)randomx_program_la64);
		*(uint64_t*)(code + luts + 0) = (uint64_t)&lutEnc[0][0];
		*(uint64_t*)(code + luts + 8) = (uint64_t)&lutDec[0][0];
	}

	// The v2 prefetch order
	{
		const uint32_t dst =
			((uint8_t*)randomx_program_la64_vm_instructions_end) - ((uint8_t*)randomx_program_la64);
		const uint32_t src = RandomX_CurrentConfig.Tweak_V2_PREFETCH ?
			((uint8_t*)randomx_program_la64_vm_instructions_end_v2) - ((uint8_t*)randomx_program_la64) :
			((uint8_t*)randomx_program_la64_vm_instructions_end_v1) - ((uint8_t*)randomx_program_la64);

		memcpy(code + dst, code + src, 16);
	}

	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code + MainLoopBegin), codePos - MainLoopBegin);
}

void JitCompilerLa64::generateProgramLight(Program& program, ProgramConfiguration& config, uint32_t datasetOffset)
{
	if (!allocatedSize) {
		allocate(ProgramSize);
	}
#ifdef XMRIG_SECURE_JIT
	else {
		enableWriting();
	}
#endif

	// lu12i.w/ori/and $r16, $r20, ScratchpadL3Mask64
	uint32_t codePos = MainLoopMask0;
	emitMaskRegister(16, 20, RandomX_CurrentConfig.ScratchpadL3Mask64_Calculated, code, codePos);

	// lu12i.w/ori/and $r17, $r21, ScratchpadL3Mask64
	codePos = MainLoopMask1;
	emitMaskRegister(17, 21, RandomX_CurrentConfig.ScratchpadL3Mask64_Calculated, code, codePos);

	codePos = PrologueSize;

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = codePos;

	for (uint32_t i = 0; i < program.getSize(); ++i)
	{
		Instruction& instr = program(i);
		(this->*engine[instr.opcode])(instr, codePos);
	}

	// Update spMix2: xor $r22, $r<readReg2>, $r<readReg3>
	emit32(LA64::XOR | LA64::RegSpMix2 |
		(LA64::IntRegMap[config.readReg2] << 5) |
		(LA64::IntRegMap[config.readReg3] << 10), code, codePos);

	// The v2 prefetch order, 8 bytes of the light tail
	{
		const uint32_t dst =
			((uint8_t*)randomx_program_la64_vm_instructions_end_light_tweak) - ((uint8_t*)randomx_program_la64);
		const uint32_t src = RandomX_CurrentConfig.Tweak_V2_PREFETCH ?
			((uint8_t*)randomx_program_la64_vm_instructions_end_light_v2) - ((uint8_t*)randomx_program_la64) :
			((uint8_t*)randomx_program_la64_vm_instructions_end_light_v1) - ((uint8_t*)randomx_program_la64);

		memcpy(code + dst, code + src, 8);
	}

	// Jump to the light dataset code
	const uint32_t lightOffset =
		(((uint8_t*)randomx_program_la64_vm_instructions_end_light) - ((uint8_t*)randomx_program_la64)) - codePos;
	emit32(LA64::B | (((lightOffset / 4) & 0xFFFF) << 10), code, codePos);

	// lu12i.w/ori/and $r6, $r6, CacheLineAlignMask
	codePos = ((uint8_t*)randomx_program_la64_light_cacheline_align_mask) - ((uint8_t*)randomx_program_la64);
	emitMaskRegister(LA64::RegScratchpad, LA64::RegScratchpad,
		RandomX_ConfigurationBase::CacheLineAlignMask_Calculated, code, codePos);

	// Update spMix1: xor $r20, $r<readReg0>, $r<readReg1>
	codePos = ((uint8_t*)randomx_program_la64_update_spMix1) - ((uint8_t*)randomx_program_la64);
	emit32(LA64::XOR | 20 |
		(LA64::IntRegMap[config.readReg0] << 5) |
		(LA64::IntRegMap[config.readReg1] << 10), code, codePos);

	/*
	 * The same AES decision as for the full
	 * memory program above.
	 */
	codePos = ((uint8_t*)randomx_program_la64_v2_FE_mix) - ((uint8_t*)randomx_program_la64);

	if (!RandomX_CurrentConfig.Tweak_V2_AES) {
		const uint32_t v1Offset =
			((uint8_t*)randomx_program_la64_v1_FE_mix) - ((uint8_t*)randomx_program_la64_v2_FE_mix);
		emit32(LA64::B | (((v1Offset / 4) & 0xFFFF) << 10), code, codePos);
	}

	// The soft AES routines read these two tables
	{
		const uint32_t luts =
			((uint8_t*)randomx_program_la64_aes_lut_pointers) - ((uint8_t*)randomx_program_la64);
		*(uint64_t*)(code + luts + 0) = (uint64_t)&lutEnc[0][0];
		*(uint64_t*)(code + luts + 8) = (uint64_t)&lutDec[0][0];
	}

	// The dataset offset, three words of the fixed shape
	codePos = ((uint8_t*)randomx_program_la64_light_dataset_offset) - ((uint8_t*)randomx_program_la64);

	const uint32_t lines = datasetOffset / CacheLineSize;

	emit32(LA64::LU12I_W | LA64::RegTemp | (((lines >> 12) & 0xFFFFF) << 5), code, codePos);
	emit32(LA64::ORI | LA64::RegTemp | (LA64::RegTemp << 5) | ((lines & 0xFFF) << 10), code, codePos);
	emit32(LA64::ADD_D | LA64::RegScratchpad | (LA64::RegScratchpad << 5) | (LA64::RegTemp << 10), code, codePos);

	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code + MainLoopBegin), codePos - MainLoopBegin);
}

template<size_t N>
void JitCompilerLa64::generateSuperscalarHash(SuperscalarProgram(&programs)[N])
{
	if (!allocatedSize) {
		allocate(ProgramSize + CalcDatasetItemSize());
	}
#ifdef XMRIG_SECURE_JIT
	else {
		enableWriting();
	}
#endif

	uint32_t codePos = ProgramSize;

	uint8_t* p1 = (uint8_t*)randomx_calc_dataset_item_la64;
	uint8_t* p2 = (uint8_t*)randomx_calc_dataset_item_la64_prefetch;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

	/*
	 * bstrpick.d $r15, $r14, Log2(CacheSize / CacheLineSize) - 1, 0
	 * keeps the cache line index of the item.
	 */
	const uint32_t msbd = GetCacheLineIndexMsbd();

	for (size_t i = 0; i < RandomX_ConfigurationBase::CacheAccesses; ++i)
	{
		emit32(LA64::BSTRPICK_D | 15 | (14 << 5) | (msbd << 16), code, codePos);

		p1 = ((uint8_t*)randomx_calc_dataset_item_la64_prefetch) + 4;
		p2 = (uint8_t*)randomx_calc_dataset_item_la64_mix;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		SuperscalarProgram& prog = programs[i];
		const size_t progSize = prog.getSize();

		for (size_t j = 0; j < progSize; ++j)
		{
			const Instruction& instr = prog(j);

			/*
			 * The register file of the dataset item
			 * function is $r4-$r11, see the
			 * contract in jit_compiler_la64_static.S.
			 */
			const uint32_t src = instr.src + 4;
			const uint32_t dst = instr.dst + 4;

			switch (static_cast<SuperscalarInstructionType>(instr.opcode))
			{
			case randomx::SuperscalarInstructionType::ISUB_R:
				emit32(LA64::SUB_D | dst | (dst << 5) | (src << 10), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_R:
				emit32(LA64::XOR | dst | (dst << 5) | (src << 10), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IADD_RS:
				if (instr.getModShift()) {
					emit32(LA64::SLLI_D | LA64::RegMask | (src << 5) | (instr.getModShift() << 10), code, codePos);
					emit32(LA64::ADD_D | dst | (dst << 5) | (LA64::RegMask << 10), code, codePos);
				}
				else {
					emit32(LA64::ADD_D | dst | (dst << 5) | (src << 10), code, codePos);
				}
				break;
			case randomx::SuperscalarInstructionType::IMUL_R:
				emit32(LA64::MUL_D | dst | (dst << 5) | (src << 10), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IROR_C:
				emit32(LA64::ROTRI_D | dst | (dst << 5) | ((instr.getImm32() & 63) << 10), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IADD_C7:
			case randomx::SuperscalarInstructionType::IADD_C8:
			case randomx::SuperscalarInstructionType::IADD_C9:
				emitAddImmediate(dst, dst, instr.getImm32(), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_C7:
			case randomx::SuperscalarInstructionType::IXOR_C8:
			case randomx::SuperscalarInstructionType::IXOR_C9:
				emitMovImmediate(LA64::RegMask, instr.getImm32(), code, codePos);
				emit32(LA64::XOR | dst | (dst << 5) | (LA64::RegMask << 10), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMULH_R:
				emit32(LA64::MULH_DU | dst | (dst << 5) | (src << 10), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::ISMULH_R:
				emit32(LA64::MULH_D | dst | (dst << 5) | (src << 10), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMUL_RCP:
				/*
				 * dst = dst * randomx_reciprocal(imm32).
				 *
				 * There is no literal pool, so the
				 * reciprocal is materialized in
				 * $r23 next to the multiply:
				 * four words plus one, which
				 * is the 20 byte budget
				 * of one superscalar
				 * instruction.
				 */
				emitMov64Immediate(LA64::RegMask, randomx_reciprocal(instr.getImm32()), code, codePos);
				emit32(LA64::MUL_D | dst | (dst << 5) | (LA64::RegMask << 10), code, codePos);
				break;
			default:
				break;
			}
		}

		p1 = (uint8_t*)randomx_calc_dataset_item_la64_mix;
		p2 = (uint8_t*)randomx_calc_dataset_item_la64_store_result;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		// Update registerValue: or $r14, $r<addressRegister>, $r0
		emit32(LA64::OR | 14 | ((prog.getAddressRegister() + 4) << 5), code, codePos);
	}

	p1 = (uint8_t*)randomx_calc_dataset_item_la64_store_result;
	p2 = (uint8_t*)randomx_calc_dataset_item_la64_end;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(code + ProgramSize), codePos - ProgramSize);
}

template void JitCompilerLa64::generateSuperscalarHash(SuperscalarProgram(&programs)[RANDOMX_CACHE_MAX_ACCESSES]);

/*
 * A 32-bit immediate. "lu12i.w" sign extends the
 * high 20 bits and "ori" adds the low 12 bits, so
 * the pair yields the sign extended value of the
 * 32-bit immediate, which is what the interpreted
 * virtual machine computes for IMUL_R/IXOR_R when
 * src == dst (see randomx::BytecodeMachine).
 */
void JitCompilerLa64::emitMovImmediate(uint32_t dst, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	emit32(LA64::LU12I_W | dst | (((imm >> 12) & 0xFFFFF) << 5), code, codePos);
	emit32(LA64::ORI | dst | (dst << 5) | ((imm & 0xFFF) << 10), code, codePos);
}

/* An arbitrary 64-bit immediate, four words. */
void JitCompilerLa64::emitMov64Immediate(uint32_t dst, uint64_t imm, uint8_t* code, uint32_t& codePos)
{
	emit32(LA64::LU12I_W | dst | ((uint32_t)(imm >> 12) & 0xFFFFF) << 5, code, codePos);
	emit32(LA64::ORI | dst | (dst << 5) | ((uint32_t)(imm & 0xFFF) << 10), code, codePos);
	emit32(LA64::LU32I_D | dst | ((uint32_t)(imm >> 32) & 0xFFFFF) << 5, code, codePos);
	emit32(LA64::LU52I_D | dst | (dst << 5) | (((uint32_t)(imm >> 52) & 0xFFF) << 10), code, codePos);
}

/*
 * dst = src + imm. The immediate is added as a
 * 64-bit value, exactly like the AArch64
 * implementation does (see jit_compiler_a64.cpp).
 */
void JitCompilerLa64::emitAddImmediate(uint32_t dst, uint32_t src, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	if (imm == 0)
	{
		if (dst != src)
		{
			// or dst, src, $r0
			emit32(LA64::OR | dst | (src << 5), code, k);
		}
	}
	else if (imm < 2048)
	{
		// addi.d dst, src, imm
		emit32(LA64::ADDI_D | dst | (src << 5) | (imm << 10), code, k);
	}
	else
	{
		// lu12i.w/ori sign extends the 32-bit value
		emit32(LA64::LU12I_W | LA64::RegMask | (((imm >> 12) & 0xFFFFF) << 5), code, k);
		emit32(LA64::ORI | LA64::RegMask | (LA64::RegMask << 5) | ((imm & 0xFFF) << 10), code, k);

		// add.d dst, src, $r23
		emit32(LA64::ADD_D | dst | (src << 5) | (LA64::RegMask << 10), code, k);
	}

	codePos = k;
}

template<uint32_t tmp_reg>
void JitCompilerLa64::emitMemLoad(uint32_t dst, uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	if (src != dst)
	{
		// (src + imm) & (Size - 8)
		emitScratchpadAddress(tmp_reg, src, instr, code, k);

		// add.d tmp_reg, $r6, tmp_reg
		emit32(LA64::ADD_D | tmp_reg | (LA64::RegScratchpad << 5) | (tmp_reg << 10), code, k);

		// ldx.d tmp_reg, tmp_reg, $r0
		emit32(LA64::LDX_D | tmp_reg | (tmp_reg << 5), code, k);
	}
	else
	{
		// (imm & ScratchpadL3Mask) + scratchpad
		const uint32_t imm = instr.getImm32() & ScratchpadL3Mask;

		emitMovImmediate(tmp_reg, imm, code, k);

		// add.d tmp_reg, $r6, tmp_reg
		emit32(LA64::ADD_D | tmp_reg | (LA64::RegScratchpad << 5) | (tmp_reg << 10), code, k);

		// ldx.d tmp_reg, tmp_reg, $r0
		emit32(LA64::LDX_D | tmp_reg | (tmp_reg << 5), code, k);
	}

	codePos = k;
}

/*
 * (src + imm) & AddressMask[modMem]
 *
 * The mask of the interpreted virtual machine is
 * AddressMask_Calculated (see randomx.cpp Apply()),
 * that is (ScratchpadL1_Size / 8 - 1) * 8 and only
 * the low three bits of the immediate can survive
 * the mask, so the sign of the immediate and the
 * zero extension of the JIT agree.
 */
void JitCompilerLa64::emitScratchpadAddress(uint32_t tmp_reg, uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	const uint32_t mask = RandomX_CurrentConfig.AddressMask_Calculated[instr.getModMem()];
	const uint32_t imm = instr.getImm32();

	if (imm) {
		emitAddImmediate(tmp_reg, src, imm, code, codePos);
	}
	else {
		// or tmp_reg, src, $r0
		emit32(LA64::OR | tmp_reg | (src << 5), code, codePos);
	}

	emitMaskRegister(tmp_reg, tmp_reg, mask, code, codePos);
}

template<uint32_t tmp_reg_fp>
void JitCompilerLa64::emitMemLoadFP(uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t mask = RandomX_CurrentConfig.AddressMask_Calculated[instr.getModMem()];
	const uint32_t imm = instr.getImm32();

	if (imm) {
		emitAddImmediate(LA64::RegTemp, src, imm, code, k);
	}
	else {
		// or $r21, src, $r0
		emit32(LA64::OR | LA64::RegTemp | (src << 5), code, k);
	}

	emitMaskRegister(LA64::RegTemp, LA64::RegTemp, mask, code, k);

	// add.d $r21, $r6, $r21
	emit32(LA64::ADD_D | LA64::RegTemp | (LA64::RegScratchpad << 5) | (LA64::RegTemp << 10), code, k);

	// vld $vr12, $r21, 0 -- two packed int32
	emit32(LA64::VLD | tmp_reg_fp | (LA64::RegTemp << 5), code, k);

	// vffintl.d.w $vr12, $vr12 -- the two int32 become two doubles
	emit32(LA64::VFFINTL_D_W | tmp_reg_fp | (tmp_reg_fp << 5), code, k);

	codePos = k;
}

void JitCompilerLa64::h_IADD_RS(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];
	const uint32_t shift = instr.getModShift();

	if (shift) {
		// slli.d $r22, src, shift; add.d dst, dst, $r22
		emit32(LA64::SLLI_D | LA64::RegSpMix2 | (src << 5) | (shift << 10), code, k);
		emit32(LA64::ADD_D | dst | (dst << 5) | (LA64::RegSpMix2 << 10), code, k);
	}
	else {
		// add.d dst, dst, src
		emit32(LA64::ADD_D | dst | (dst << 5) | (src << 10), code, k);
	}

	if (instr.dst == RegisterNeedsDisplacement) {
		// The displacement is added as a 64-bit value
		emitAddImmediate(dst, dst, instr.getImm32(), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IADD_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	emitMemLoad<LA64::RegTemp>(dst, src, instr, code, k);

	// add.d dst, dst, $r21
	emit32(LA64::ADD_D | dst | (dst << 5) | (LA64::RegTemp << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_ISUB_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	if (src != dst) {
		// sub.d dst, dst, src
		emit32(LA64::SUB_D | dst | (dst << 5) | (src << 10), code, k);
	}
	else {
		// dst + (uint32_t)(-(int32_t)imm) is dst - imm
		emitAddImmediate(dst, dst, (uint32_t)(-(int32_t)instr.getImm32()), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_ISUB_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	emitMemLoad<LA64::RegTemp>(dst, src, instr, code, k);

	// sub.d dst, dst, $r21
	emit32(LA64::SUB_D | dst | (dst << 5) | (LA64::RegTemp << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IMUL_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	if (src == dst) {
		// The interpreter multiplies by the immediate
		src = LA64::RegTemp;
		emitMovImmediate(src, instr.getImm32(), code, k);
	}

	// mul.d dst, dst, src
	emit32(LA64::MUL_D | dst | (dst << 5) | (src << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IMUL_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	emitMemLoad<LA64::RegTemp>(dst, src, instr, code, k);

	// mul.d dst, dst, $r21
	emit32(LA64::MUL_D | dst | (dst << 5) | (LA64::RegTemp << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IMULH_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	// mulh.du dst, dst, src
	emit32(LA64::MULH_DU | dst | (dst << 5) | (src << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IMULH_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	emitMemLoad<LA64::RegTemp>(dst, src, instr, code, k);

	// mulh.du dst, dst, $r21
	emit32(LA64::MULH_DU | dst | (dst << 5) | (LA64::RegTemp << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_ISMULH_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	// mulh.d dst, dst, src
	emit32(LA64::MULH_D | dst | (dst << 5) | (src << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_ISMULH_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	emitMemLoad<LA64::RegTemp>(dst, src, instr, code, k);

	// mulh.d dst, dst, $r21
	emit32(LA64::MULH_D | dst | (dst << 5) | (LA64::RegTemp << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IMUL_RCP(Instruction& instr, uint32_t& codePos)
{
	const uint64_t divisor = instr.getImm32();

	if (isZeroOrPowerOf2(divisor)) {
		// The interpreted virtual machine treats it as NOP
		return;
	}

	uint32_t k = codePos;

	const uint32_t dst = LA64::IntRegMap[instr.dst];

	// The interpreter multiplies by randomx_reciprocal(divisor)
	emitMov64Immediate(dst, randomx_reciprocal(divisor), code, k);

	// mul.d dst, dst, dst
	emit32(LA64::MUL_D | dst | (dst << 5) | (dst << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_INEG_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	// sub.d dst, $r0, dst
	emit32(LA64::SUB_D | dst | (dst << 10), code, codePos);

	reg_changed_offset[instr.dst] = codePos;
}

void JitCompilerLa64::h_IXOR_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	if (src == dst) {
		// The interpreter exclusive ors by the immediate
		src = LA64::RegTemp;
		emitMovImmediate(src, instr.getImm32(), code, k);
	}

	// xor dst, dst, src
	emit32(LA64::XOR | dst | (dst << 5) | (src << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IXOR_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	emitMemLoad<LA64::RegTemp>(dst, src, instr, code, k);

	// xor dst, dst, $r21
	emit32(LA64::XOR | dst | (dst << 5) | (LA64::RegTemp << 10), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_IROR_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	if (src != dst) {
		// rotr.d dst, dst, src
		emit32(LA64::ROTR_D | dst | (dst << 5) | (src << 10), code, codePos);
	}
	else if (instr.getImm32() & 63) {
		// rotri.d dst, dst, imm
		emit32(LA64::ROTRI_D | dst | (dst << 5) | ((instr.getImm32() & 63) << 10), code, codePos);
	}

	reg_changed_offset[instr.dst] = codePos;
}

void JitCompilerLa64::h_IROL_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	if (src != dst) {
		// sub.d $r21, $r0, src; rotr.d dst, dst, $r21
		emit32(LA64::SUB_D | LA64::RegTemp | (src << 10), code, k);
		emit32(LA64::ROTR_D | dst | (dst << 5) | (LA64::RegTemp << 10), code, k);
	}
	else if (instr.getImm32() & 63) {
		// rotri.d dst, dst, (64 - imm)
		emit32(LA64::ROTRI_D | dst | (dst << 5) | ((-instr.getImm32() & 63) << 10), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_ISWAP_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	if (src == dst) {
		// The interpreted virtual machine has nothing to do
		return;
	}

	uint32_t k = codePos;

	// or $r21, src, $r0 / or src, dst, $r0 / or dst, $r21, $r0
	emit32(LA64::OR | LA64::RegTemp | (src << 5), code, k);
	emit32(LA64::OR | src | (dst << 5), code, k);
	emit32(LA64::OR | dst | (LA64::RegTemp << 5), code, k);

	reg_changed_offset[instr.src] = k;
	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerLa64::h_FSWAP_R(Instruction& instr, uint32_t& codePos)
{
	/*
	 * The interpreted machine swaps the two doubles of
	 * $vr<instr.dst % RegistersCount>, that is of group
	 * F for dst < 4 and of group E for dst >= 4.
	 * vshuf4i.d is not a pair shuffle, see the
	 * measurement in logs/153-la64-vex.txt, therefore
	 * the two halves travel through $r21 and $r22,
	 * measured with scripts/154-la64-fswap-probe.S and
	 * recorded in logs/155-la64-fswap.txt.
	 */
	const uint32_t dst = instr.dst % RegistersCount;

	// vpickve2gr.du $r21, $vr<dst>, 0
	emit32(LA64::VPICKVE2GR_DU | LA64::RegTemp | (dst << 5), code, codePos);
	// vpickve2gr.du $r22, $vr<dst>, 1
	emit32(LA64::VPICKVE2GR_DU | LA64::RegSpMix2 | (dst << 5) | (1 << 10), code, codePos);
	// vinsgr2vr.d $vr<dst>, $r21, 1
	emit32(LA64::VINSGR2VR_D | dst | (LA64::RegTemp << 5) | (1 << 10), code, codePos);
	// vinsgr2vr.d $vr<dst>, $r22, 0
	emit32(LA64::VINSGR2VR_D | dst | (LA64::RegSpMix2 << 5), code, codePos);
}

void JitCompilerLa64::h_FADD_R(Instruction& instr, uint32_t& codePos)
{
	// the source is group A, the destination is group F
	const uint32_t src = (instr.src % RegisterCountFlt) + LA64::RegGroupA;
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupF;

	// vfadd.d dst, dst, src
	emit32(LA64::VFADD_D | dst | (dst << 5) | (src << 10), code, codePos);
}

void JitCompilerLa64::h_FADD_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupF;

	emitMemLoadFP<LA64::RegTempFp>(src, instr, code, k);

	// vfadd.d dst, dst, $vr12
	emit32(LA64::VFADD_D | dst | (dst << 5) | (LA64::RegTempFp << 10), code, k);

	codePos = k;
}

void JitCompilerLa64::h_FSUB_R(Instruction& instr, uint32_t& codePos)
{
	// the source is group A, the destination is group F
	const uint32_t src = (instr.src % RegisterCountFlt) + LA64::RegGroupA;
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupF;

	// vfsub.d dst, dst, src
	emit32(LA64::VFSUB_D | dst | (dst << 5) | (src << 10), code, codePos);
}

void JitCompilerLa64::h_FSUB_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupF;

	emitMemLoadFP<LA64::RegTempFp>(src, instr, code, k);

	// vfsub.d dst, dst, $vr12
	emit32(LA64::VFSUB_D | dst | (dst << 5) | (LA64::RegTempFp << 10), code, k);

	codePos = k;
}

void JitCompilerLa64::h_FSCAL_R(Instruction& instr, uint32_t& codePos)
{
	// group F, the scale mask is $vr15
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupF;

	// vxor.v dst, dst, $vr15
	emit32(LA64::VXOR_V | dst | (dst << 5) | (15 << 10), code, codePos);
}

void JitCompilerLa64::h_FMUL_R(Instruction& instr, uint32_t& codePos)
{
	// the source is group A, the destination is group E
	const uint32_t src = (instr.src % RegisterCountFlt) + LA64::RegGroupA;
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupE;

	// vfmul.d dst, dst, src
	emit32(LA64::VFMUL_D | dst | (dst << 5) | (src << 10), code, codePos);
}

void JitCompilerLa64::h_FDIV_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupE;

	emitMemLoadFP<LA64::RegTempFp>(src, instr, code, k);

	// and $vr12, $vr12, $vr13; or $vr12, $vr12, $vr14
	emit32(LA64::VAND_V | LA64::RegTempFp | (LA64::RegTempFp << 5) | (13 << 10), code, k);
	emit32(LA64::VOR_V | LA64::RegTempFp | (LA64::RegTempFp << 5) | (14 << 10), code, k);

	// vfdiv.d dst, dst, $vr12
	emit32(LA64::VFDIV_D | dst | (dst << 5) | (LA64::RegTempFp << 10), code, k);

	codePos = k;
}

void JitCompilerLa64::h_FSQRT_R(Instruction& instr, uint32_t& codePos)
{
	// group E
	const uint32_t dst = (instr.dst % RegisterCountFlt) + LA64::RegGroupE;

	// vfsqrt.d dst, dst
	emit32(LA64::VFSQRT_D | dst | (dst << 5), code, codePos);
}

void JitCompilerLa64::h_CBRANCH(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t dst = LA64::IntRegMap[instr.dst];
	const uint32_t modCond = instr.getModCond();
	const uint32_t shift = modCond + RandomX_ConfigurationBase::JumpOffset;
	const uint32_t imm = (instr.getImm32() | (1U << shift)) & ~(1U << (shift - 1));

	emitAddImmediate(dst, dst, imm, code, k);

	// bstrpick.d $r22, $r<dst>, 15 + modCond, 8 + modCond
	emit32(LA64::BSTRPICK_D | LA64::RegSpMix2 | (dst << 5) |
		((15 + modCond) << 16) | ((8 + modCond) << 10), code, k);

	// beq $r22, $r0, <the position where dst was last changed>
	// The offset of a beq is a word offset in bits
	// 10-25, measured in logs/149-la64-branch.txt.
	// The target lies behind k, so the difference
	// is negative: mask it down to the 16
	// immediate bits, exactly like the
	// "(offset - k) >> 2) & ((1 << 19) - 1)"
	// of jit_compiler_a64.cpp does. Without the
	// mask the sign bits would spill into the
	// opcode field and produce an illegal word.
	int32_t offset = reg_changed_offset[instr.dst];
	offset = ((offset - static_cast<int32_t>(k)) >> 2) & 0xFFFF;

	emit32(LA64::BEQ | LA64::RegSpMix2 | (static_cast<uint32_t>(offset) << 10), code, k);

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = k;

	codePos = k;
}

void JitCompilerLa64::h_CFROUND(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];

	// rotri.d $r23, $r<src>, imm
	emit32(LA64::ROTRI_D | LA64::RegMask | (src << 5) | ((instr.getImm32() & 63) << 10), code, k);

	if (RandomX_CurrentConfig.Tweak_V2_CFROUND) {
		/*
		 * In RandomX v2 the rounding mode is only
		 * written when (isrc & 60) == 0, exactly like
		 * exe_CFROUND() of bytecode_machine.hpp.
		 */
		// andi $r21, $r23, 60
		emit32(LA64::ANDI | LA64::RegTemp | (LA64::RegMask << 5) |
			(LA64::FCSR_ROUNDING_MASK << 10), code, k);

		// bne $r21, $r0, over the three
		// words below. The offset field of a
		// La64 branch counts words, so this is
		// 3: bstrins.d, movgr2fcsr and the
		// sub.d below. The A64 blueprint
		// (0x54000081) skips 4 of them, one
		// more than it emits, and would land
		// on the first word of the next
		// bytecode instruction.
		emit32(LA64::BNE | LA64::RegTemp | (3 << 10), code, k);
	}

	/*
	 * The two bits [9:8] of $fcsr0 hold the same
	 * rounding mode that fesetround() writes,
	 * measured with scripts/144-la64-fcsr-probe.c:
	 * FE_TONEAREST 0, FE_TOWARDZERO 1, FE_UPWARD 2,
	 * FE_DOWNWARD 3. rx_set_rounding_mode() is the
	 * interpreted machine's writer, so the field is
	 * exactly the two's complement of the two low
	 * bits of the rotated value.
	 */
	// sub.d $r21, $r0, $r23
	emit32(LA64::SUB_D | LA64::RegTemp | (LA64::RegMask << 10), code, k);

	// bstrins.d $r18, $r21, 9, 8
	emit32(LA64::BSTRINS_D | LA64::RegFcsrShadow | (LA64::RegTemp << 5) |
		(LA64::FCSR_ROUNDING_MSB << 16) | (LA64::FCSR_ROUNDING_LSB << 10), code, k);

	// movgr2fcsr $fcsr0, $r18
	emit32(LA64::MOVGR2FCSR | (LA64::RegFcsrShadow << 5), code, k);

	codePos = k;
}

void JitCompilerLa64::h_ISTORE(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = LA64::IntRegMap[instr.src];
	const uint32_t dst = LA64::IntRegMap[instr.dst];

	const uint32_t size = (instr.getModCond() < StoreL3Condition) ?
		(instr.getModMem() ? RandomX_CurrentConfig.ScratchpadL1_Size : RandomX_CurrentConfig.ScratchpadL2_Size) :
		RandomX_CurrentConfig.ScratchpadL3_Size;

	const uint32_t imm = instr.getImm32() & (size - 1);
	const uint32_t mask = size - 8;

	if (imm) {
		emitAddImmediate(LA64::RegTemp, dst, imm, code, k);
	}
	else {
		// or $r21, dst, $r0
		emit32(LA64::OR | LA64::RegTemp | (dst << 5), code, k);
	}

	// and $r21, $r21, $r23 (the mask word below)
	emit32(LA64::LU12I_W | LA64::RegMask | (((mask >> 12) & 0xFFFFF) << 5), code, k);
	emit32(LA64::ORI | LA64::RegMask | (LA64::RegMask << 5) | ((mask & 0xFFF) << 10), code, k);
	emit32(LA64::AND | LA64::RegTemp | (LA64::RegTemp << 5) | (LA64::RegMask << 10), code, k);

	// stx.d src, $r6, $r21  --  [$r6 + $r21], the scratchpad
	// is the base, exactly like the AArch64
	// "str src, [x2, tmp_reg]".
	emit32(LA64::STX_D | src | (LA64::RegScratchpad << 5) | (LA64::RegTemp << 10), code, k);

	codePos = k;
}

void JitCompilerLa64::h_NOP(Instruction& instr, uint32_t& codePos)
{
}

InstructionGeneratorLa64 JitCompilerLa64::engine[256] = {};

}
