/*
 * The 256 bit ( LASX ) half of the LoongArch dataset
 * item generator.
 *
 * The 128 bit half is the " la64 " code of
 * jit_compiler_la64.cpp; this file is only built
 * when cmake/cpu.cmake proved that the CPU
 * carries LASX, so a release of a machine
 * with plain LSX holds none of the
 * words below.
 *
 * reports/26-*.md holds the derivation of
 * the leading words, which came from
 * the assembler of this machine
 * ( scripts/298-la64-lasx-mnemonic-check.sh ).
 */

#include <cstring>

#include "backend/cpu/Cpu.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/randomx/jit_compiler_la64.hpp"
#include "crypto/randomx/jit_compiler_la64_lasx.hpp"
#include "crypto/randomx/jit_compiler_la64_lasx_static.hpp"
#include "crypto/randomx/common.hpp"
#include "crypto/randomx/reciprocal.h"
#include "crypto/randomx/superscalar.hpp"
#include "crypto/randomx/virtual_memory.hpp"

namespace randomx {
namespace la64_lasx {

static size_t GetLoopSize()
{
	return ((uint8_t*)randomx_init_dataset_la64_lasx_end) - ((uint8_t*)randomx_init_dataset_la64_lasx);
}

static void emit32(uint32_t val, uint8_t* code, uint32_t& codePos)
{
	*(uint32_t*)(code + codePos) = val;
	codePos += sizeof(val);
}

/*
 * The mask of the cache line index, exactly the
 * number of the 128 bit half.
 */
uint64_t getCacheLineMask()
{
	return ((uint64_t)RandomX_ConfigurationBase::ArgonMemory * ArgonBlockSize) / CacheLineSize - 1;
}

size_t getBlobSize(size_t accesses)
{
	return
		GetLoopSize() +
		((uint8_t*)randomx_calc_dataset_item_la64_lasx_prefetch - (uint8_t*)randomx_calc_dataset_item_la64_lasx) +
		/* the three words of the cache line index mask */
		12 +
		accesses * (
			((uint8_t*)randomx_calc_dataset_item_la64_lasx_mix - (uint8_t*)randomx_calc_dataset_item_la64_lasx_prefetch) +
			((RandomX_ConfigurationBase::SuperscalarMaxLatency * 3) + 2) * 24 +
			((uint8_t*)randomx_calc_dataset_item_la64_lasx_store_result - (uint8_t*)randomx_calc_dataset_item_la64_lasx_mix) + 4
		) +
		((uint8_t*)randomx_calc_dataset_item_la64_lasx_end - (uint8_t*)randomx_calc_dataset_item_la64_lasx_store_result);
}

size_t copyInitLoop(uint8_t* code)
{
	memcpy(code, (uint8_t*)randomx_init_dataset_la64_lasx, GetLoopSize());

	return GetLoopSize();
}

}
}
namespace randomx {
namespace la64_lasx {
constexpr uint32_t XVSUB_D      = 0x740D8000;
constexpr uint32_t XVADD_D      = 0x740B8000;
constexpr uint32_t XVMUL_D      = 0x74858000;
constexpr uint32_t XVMUH_D      = 0x74878000;
constexpr uint32_t XVMUH_DU     = 0x74898000;
constexpr uint32_t XVXOR_V      = 0x75270000;
constexpr uint32_t XVOR_V       = 0x75268000;
constexpr uint32_t XVSLLI_D     = 0x772D0000;
constexpr uint32_t XVROTRI_D    = 0x76A10000;
constexpr uint32_t XVREPLGR2VR_D = 0x769F0C00;
constexpr uint32_t LU12I_W      = 0x14000000;
constexpr uint32_t ORI          = 0x03800000;
constexpr uint32_t LU32I_D      = 0x16000000;
constexpr uint32_t LU52I_D      = 0x03000000;

/*
 * The two scratch registers of the emitted
 * program, see jit_compiler_la64_lasx_static.S:
 * $xr21 for the shifted addend,
 * $xr23 for a 64 bit immediate.
 */
constexpr uint32_t RegShift = 21;
constexpr uint32_t RegImm = 23;

/* lu12i.w / ori / lu32i.d / lu52i.d, one 64 bit immediate. */
static void emitScalarImmediate(uint32_t reg, uint64_t imm, uint8_t* code, uint32_t& codePos)
{
	emit32(LU12I_W | reg | ((((uint32_t)(imm >> 12)) & 0xFFFFF) << 5), code, codePos);
	emit32(ORI | reg | (reg << 5) | ((((uint32_t)imm) & 0xFFF) << 10), code, codePos);
	emit32(LU32I_D | reg | ((((uint32_t)(imm >> 32)) & 0xFFFFF) << 5), code, codePos);
	emit32(LU52I_D | reg | (reg << 5) | ((((uint32_t)(imm >> 52)) & 0xFFF) << 10), code, codePos);
}
/*
 * The C7 / C8 / C9 forms of the interpreter
 * add and xor signExtend2sCompl( imm32 )
 * ( superscalar.cpp ), the value of
 * the 32 bit immediate with
 * its sign word kept,
 * so the four words
 * below must carry
 * it too.
 */
static uint64_t signExtend32(uint32_t imm)
{
	return (uint64_t)(int64_t)(int32_t) imm;
}

/* dst = dst + imm, on all four items. */
static void emitAddImmediate(uint32_t dst, uint64_t imm, uint8_t* code, uint32_t& codePos)
{
	emitScalarImmediate(RegImm, imm, code, codePos);

	emit32(XVREPLGR2VR_D | RegImm | (RegImm << 5), code, codePos);
	emit32(XVADD_D | dst | (dst << 5) | (RegImm << 10), code, codePos);
}

/* dst = dst ^ imm, on all four items. */
static void emitXorImmediate(uint32_t dst, uint64_t imm, uint8_t* code, uint32_t& codePos)
{
	emitScalarImmediate(RegImm, imm, code, codePos);

	emit32(XVREPLGR2VR_D | RegImm | (RegImm << 5), code, codePos);
	emit32(XVXOR_V | dst | (dst << 5) | (RegImm << 10), code, codePos);
}

/* dst = dst * randomx_reciprocal(imm), on all four items. */
static void emitLasxMultiply(uint32_t dst, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	emitScalarImmediate(RegImm, randomx_reciprocal(imm), code, codePos);

	emit32(XVREPLGR2VR_D | RegImm | (RegImm << 5), code, codePos);
	emit32(XVMUL_D | dst | (dst << 5) | (RegImm << 10), code, codePos);
}

/*
 * One superscalar instruction. The 256 bit
 * layer spends at most 24 bytes on
 * one, where the 128 bit
 * layer spends 20.
 */
static void emitInstruction(const Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	const uint32_t src = instr.src;
	const uint32_t dst = instr.dst;

	switch (static_cast<SuperscalarInstructionType>(instr.opcode)) {
	case SuperscalarInstructionType::ISUB_R:
		emit32(XVSUB_D | dst | (dst << 5) | (src << 10), code, codePos);
		break;
	case SuperscalarInstructionType::IXOR_R:
		emit32(XVXOR_V | dst | (dst << 5) | (src << 10), code, codePos);
		break;
	case SuperscalarInstructionType::IADD_RS:
		if (instr.getModShift()) {
			emit32(XVSLLI_D | RegShift | (src << 5) | (instr.getModShift() << 10), code, codePos);
			emit32(XVADD_D | dst | (dst << 5) | (RegShift << 10), code, codePos);
		}
		else {
			emit32(XVADD_D | dst | (dst << 5) | (src << 10), code, codePos);
		}
		break;
	case SuperscalarInstructionType::IMUL_R:
		emit32(XVMUL_D | dst | (dst << 5) | (src << 10), code, codePos);
		break;
	case SuperscalarInstructionType::IROR_C:
		emit32(XVROTRI_D | dst | (dst << 5) | ((instr.getImm32() & 63) << 10), code, codePos);
		break;
	case SuperscalarInstructionType::IADD_C7:
	case SuperscalarInstructionType::IADD_C8:
	case SuperscalarInstructionType::IADD_C9:
		emitAddImmediate(dst, signExtend32(instr.getImm32()), code, codePos);
		break;
	case SuperscalarInstructionType::IXOR_C7:
	case SuperscalarInstructionType::IXOR_C8:
	case SuperscalarInstructionType::IXOR_C9:
		emitXorImmediate(dst, signExtend32(instr.getImm32()), code, codePos);
		break;
	case SuperscalarInstructionType::IMULH_R:
		emit32(XVMUH_DU | dst | (dst << 5) | (src << 10), code, codePos);
		break;
	case SuperscalarInstructionType::ISMULH_R:
		emit32(XVMUH_D | dst | (dst << 5) | (src << 10), code, codePos);
		break;
	case SuperscalarInstructionType::IMUL_RCP:
		emitLasxMultiply(dst, instr.getImm32(), code, codePos);
		break;
	default:
		break;
	}
}
/*
 * The whole body of the four item wide
 * item function: the prologue with
 * the constants table, one
 * prefetch, program and
 * mix per access.
 */
size_t generateSuperscalarHash(SuperscalarProgram* programs, size_t accesses, uint8_t* code)
{
	uint32_t codePos = 0;

	uint8_t* p1 = (uint8_t*)randomx_calc_dataset_item_la64_lasx;
	uint8_t* p2 = (uint8_t*)randomx_calc_dataset_item_la64_lasx_prefetch;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

	/*
	 * The cache line index mask, the one value
	 * the JIT compiler owns: the copy of
	 * the body cannot hold it, the two
	 * " la.local " words above it
	 * travel with a linker
	 * relocation, so the
	 * copy reads the
	 * tables of
	 * the image.
	 */
	const uint64_t cacheLineMask = getCacheLineMask();

	/* lu12i.w $r21, hi20 */
	emit32(LU12I_W | 21 | ((((uint32_t)(cacheLineMask >> 12)) & 0xFFFFF) << 5), code, codePos);

	/* ori $r21, $r21, lo12 */
	emit32(ORI | 21 | (21 << 5) | ((((uint32_t)cacheLineMask) & 0xFFF) << 10), code, codePos);

	/* xvreplgr2vr.d $xr9, $r21 */
	emit32(XVREPLGR2VR_D | 9 | (21 << 5), code, codePos);

	for (size_t i = 0; i < accesses; ++i) {
		p1 = (uint8_t*)randomx_calc_dataset_item_la64_lasx_prefetch;
		p2 = (uint8_t*)randomx_calc_dataset_item_la64_lasx_mix;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		SuperscalarProgram& prog = programs[i];
		const size_t progSize = prog.getSize();

		for (size_t j = 0; j < progSize; ++j) {
			emitInstruction(prog(j), code, codePos);
		}

		p1 = (uint8_t*)randomx_calc_dataset_item_la64_lasx_mix;
		p2 = (uint8_t*)randomx_calc_dataset_item_la64_lasx_store_result;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		/* Update registerValue: or $xr8, $xr<address>, $xr<address>. */
		emit32(XVOR_V | 8 | (prog.getAddressRegister() << 5) | (prog.getAddressRegister() << 10), code, codePos);
	}

	p1 = (uint8_t*)randomx_calc_dataset_item_la64_lasx_store_result;
	p2 = (uint8_t*)randomx_calc_dataset_item_la64_lasx_end;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

	return codePos;
}

}
}
namespace randomx {
bool JitCompilerLa64::lasxEnable() const
{
	return xmrig::Cpu::info()->has(xmrig::ICpuInfo::FLAG_LASX);
}

/*
 * The 256 bit item function is written into a
 * buffer of its own, so the " la64 " code
 * of jit_compiler_la64.cpp is
 * never touched here.
 */
void JitCompilerLa64::generateSuperscalarHashLasx(SuperscalarProgram* programs, size_t accesses)
{
	if (!accesses) {
		return;
	}

	lasxCodeSize = la64_lasx::getBlobSize(accesses);
	lasxCode = static_cast<uint8_t*>(allocExecutableMemory(lasxCodeSize, hugePages));

	uint32_t codePos = la64_lasx::copyInitLoop(lasxCode);

	lasxInitFunc = lasxCode;

	codePos += la64_lasx::generateSuperscalarHash(programs, accesses, lasxCode + codePos);

	xmrig::VirtualMemory::flushInstructionCache(reinterpret_cast<char*>(lasxCode), codePos);
}
}
