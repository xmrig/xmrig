/*
Copyright (c) 2018-2020, tevador    <tevador@gmail.com>
Copyright (c) 2019-2020, SChernykh  <https://github.com/SChernykh>
Copyright (c) 2019-2020, XMRig      <https://github.com/xmrig>, <support@xmrig.com>
Copyright (c) 2026, PalindromicBreadLoaf <https://github.com/palindromicbreadloaf>

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

#include "crypto/randomx/jit_compiler_ppc.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "crypto/common/VirtualMemory.h"
#include "crypto/randomx/intrin_portable.h"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/virtual_memory.hpp"

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#   define XMRIG_PPC_JIT_LITTLE_ENDIAN 1
#else
#   define XMRIG_PPC_JIT_LITTLE_ENDIAN 0
#endif

#if defined(__GNUC__)
#   define XMRIG_PPC_JIT_MAYBE_UNUSED __attribute__((unused))
#else
#   define XMRIG_PPC_JIT_MAYBE_UNUSED
#endif

static bool hugePagesJIT = false;
static int optimizedDatasetInit = -1;

void randomx_set_huge_pages_jit(const bool hugePages)
{
	hugePagesJIT = hugePages;
}

void randomx_set_optimized_dataset_init(const int value)
{
	optimizedDatasetInit = value;
}

namespace {

	constexpr uint32_t PPC_BL_NEXT = 0x48000005U;
	constexpr uint32_t PPC_BCTR    = 0x4E800420U;
	constexpr uint32_t PPC_BCTRL   = 0x4E800421U;
	constexpr uint32_t PPC_BLR     = 0x4E800020U;
	constexpr uint32_t PPC_NOP     = 0x60000000U;
	constexpr uint32_t NativeProgramCodeSize = 256U * 1024U;
	constexpr int32_t NativeFrameSize = 176;
	constexpr int32_t SaveR15Offset = 112;
	constexpr int32_t SaveR16Offset = 120;
	constexpr int32_t SaveR17Offset = 128;
	constexpr int32_t SaveTocOffset = 136;
	constexpr int32_t FpConvertSlot0 = 144;
	constexpr int32_t FpConvertSlot1 = 152;
	constexpr int32_t NativeFrameSize32 = 96;
	constexpr int32_t SaveR15Offset32 = 32;
	constexpr int32_t SaveR16Offset32 = 36;
	constexpr int32_t SaveR17Offset32 = 40;
	constexpr int32_t FpConvertSlot032 = 48;
	constexpr int32_t FpConvertSlot132 = 56;
	constexpr int32_t FpConvertConstSlot32 = 64;

	struct BranchFixup {
		size_t pos;
		int target;
	};

	constexpr uint32_t add(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000214U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t addc(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000014U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t adde(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000114U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t addi(uint32_t rd, uint32_t ra, int32_t imm)
	{
		return 0x38000000U | (rd << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFU);
	}

	constexpr uint32_t andReg(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C000038U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t andDot(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C000039U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t andiDot(uint32_t ra, uint32_t rs, uint32_t imm)
	{
		return 0x70000000U | (rs << 21) | (ra << 16) | (imm & 0xFFFFU);
	}

	constexpr uint32_t andisDot(uint32_t ra, uint32_t rs, uint32_t imm)
	{
		return 0x74000000U | (rs << 21) | (ra << 16) | (imm & 0xFFFFU);
	}

	constexpr uint32_t branch(int32_t offset)
	{
		return 0x48000000U | (static_cast<uint32_t>(offset) & 0x03FFFFFCU);
	}

	constexpr uint32_t branchCond(uint32_t bo, uint32_t bi, int32_t offset)
	{
		return 0x40000000U | (bo << 21) | (bi << 16) | (static_cast<uint32_t>(offset) & 0xFFFCU);
	}

	constexpr uint32_t mflr(uint32_t rd)
	{
		return 0x7C0002A6U | (rd << 21) | (8U << 16);
	}

	constexpr uint32_t mtlr(uint32_t rs)
	{
		return 0x7C0003A6U | (rs << 21) | (8U << 16);
	}

	constexpr uint32_t mtctr(uint32_t rs)
	{
		return 0x7C0003A6U | (rs << 21) | (9U << 16);
	}

	constexpr uint32_t mtfsfi(uint32_t bf, uint32_t imm)
	{
		return 0xFC00010CU | ((bf & 7U) << 23) | ((imm & 15U) << 12);
	}

	constexpr uint32_t ld(uint32_t rd, uint32_t ra, int32_t offset)
	{
		return 0xE8000000U | (rd << 21) | (ra << 16) | (static_cast<uint32_t>(offset) & 0xFFFCU);
	}

	constexpr uint32_t lfd(uint32_t frd, uint32_t ra, int32_t offset)
	{
		return 0xC8000000U | (frd << 21) | (ra << 16) | (static_cast<uint32_t>(offset) & 0xFFFFU);
	}

	constexpr uint32_t lwz(uint32_t rd, uint32_t ra, int32_t offset)
	{
		return 0x80000000U | (rd << 21) | (ra << 16) | (static_cast<uint32_t>(offset) & 0xFFFFU);
	}

	constexpr uint32_t lwbrx(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C00042CU | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t ldx(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C00002AU | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t ldbrx(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000428U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t mr(uint32_t ra, uint32_t rs)
	{
		return 0x7C000378U | (rs << 21) | (ra << 16) | (rs << 11);
	}

	constexpr uint32_t mullw(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C0001D6U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t mulhwu(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000016U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t mulld(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C0001D2U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t mulhd(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000092U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t mulhdu(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000012U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t neg(uint32_t rd, uint32_t ra)
	{
		return 0x7C0000D0U | (rd << 21) | (ra << 16);
	}

	constexpr uint32_t orReg(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C000378U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t rotld(uint32_t ra, uint32_t rs, uint32_t rb)
	{
		return 0x78000010U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t rlwinm(uint32_t ra, uint32_t rs, uint32_t sh, uint32_t mb, uint32_t me)
	{
		return 0x54000000U | (rs << 21) | (ra << 16) | ((sh & 31U) << 11) | ((mb & 31U) << 6) | ((me & 31U) << 1);
	}

	constexpr uint32_t sld(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C000036U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t slw(uint32_t ra, uint32_t rs, uint32_t rb)
	{
		return 0x7C000030U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t srw(uint32_t ra, uint32_t rs, uint32_t rb)
	{
		return 0x7C000430U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t subf(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000050U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t subfic(uint32_t rd, uint32_t ra, int32_t imm)
	{
		return 0x20000000U | (rd << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFU);
	}

	constexpr uint32_t subfc(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000010U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t subfe(uint32_t rd, uint32_t ra, uint32_t rb)
	{
		return 0x7C000110U | (rd << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t stw(uint32_t rs, uint32_t ra, int32_t offset)
	{
		return 0x90000000U | (rs << 21) | (ra << 16) | (static_cast<uint32_t>(offset) & 0xFFFFU);
	}

	constexpr uint32_t stwu(uint32_t rs, uint32_t ra, int32_t offset)
	{
		return 0x94000000U | (rs << 21) | (ra << 16) | (static_cast<uint32_t>(offset) & 0xFFFFU);
	}

	constexpr uint32_t std64(uint32_t rs, uint32_t ra, int32_t offset)
	{
		return 0xF8000000U | (rs << 21) | (ra << 16) | (static_cast<uint32_t>(offset) & 0xFFFCU);
	}

	constexpr uint32_t stfd(uint32_t frs, uint32_t ra, int32_t offset)
	{
		return 0xD8000000U | (frs << 21) | (ra << 16) | (static_cast<uint32_t>(offset) & 0xFFFFU);
	}

	constexpr uint32_t stdu(uint32_t rs, uint32_t ra, int32_t offset)
	{
		return std64(rs, ra, offset) | 1U;
	}

	constexpr uint32_t stdx(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C00012AU | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t stdbrx(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C000528U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t stwbrx(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C00052CU | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t xorReg(uint32_t rs, uint32_t ra, uint32_t rb)
	{
		return 0x7C000278U | (rs << 21) | (ra << 16) | (rb << 11);
	}

	constexpr uint32_t extsw(uint32_t ra, uint32_t rs)
	{
		return 0x7C0007B4U | (rs << 21) | (ra << 16);
	}

	constexpr uint32_t fadd(uint32_t frd, uint32_t fra, uint32_t frb)
	{
		return 0xFC00002AU | (frd << 21) | (fra << 16) | (frb << 11);
	}

	constexpr uint32_t fsub(uint32_t frd, uint32_t fra, uint32_t frb)
	{
		return 0xFC000028U | (frd << 21) | (fra << 16) | (frb << 11);
	}

	constexpr uint32_t fmul(uint32_t frd, uint32_t fra, uint32_t frc)
	{
		return 0xFC000032U | (frd << 21) | (fra << 16) | (frc << 6);
	}

	constexpr uint32_t fdiv(uint32_t frd, uint32_t fra, uint32_t frb)
	{
		return 0xFC000024U | (frd << 21) | (fra << 16) | (frb << 11);
	}

	constexpr uint32_t fsqrt(uint32_t frd, uint32_t frb)
	{
		return 0xFC00002CU | (frd << 21) | (frb << 11);
	}

	constexpr uint32_t fcfid(uint32_t frd, uint32_t frb)
	{
		return 0xFC00069CU | (frd << 21) | (frb << 11);
	}

	void emit32(uint8_t* code, size_t& pos, uint32_t value)
	{
		memcpy(code + pos, &value, sizeof(value));
		pos += sizeof(value);
	}

	void emitPtr(uint8_t* code, size_t& pos, uintptr_t value)
	{
		memcpy(code + pos, &value, sizeof(value));
		pos += sizeof(value);
	}

#if defined(__powerpc64__) && defined(_CALL_ELF) && (_CALL_ELF == 1)
	struct FunctionDescriptor {
		uintptr_t entry;
		uintptr_t toc;
		uintptr_t env;
	};

	template<typename Func>
	FunctionDescriptor getFunctionDescriptor(Func func)
	{
		const auto* descriptor = reinterpret_cast<const uintptr_t*>(func);
		return { descriptor[0], descriptor[1], descriptor[2] };
	}
#endif

	XMRIG_PPC_JIT_MAYBE_UNUSED void write32(uint8_t* code, size_t pos, uint32_t value)
	{
		memcpy(code + pos, &value, sizeof(value));
	}

	void patchBranch(uint8_t* code, size_t branchPos, size_t targetPos)
	{
		write32(code, branchPos, branch(static_cast<int32_t>(targetPos - branchPos)));
	}

	void patchBranchCond(uint8_t* code, size_t branchPos, size_t targetPos, uint32_t bo, uint32_t bi)
	{
		write32(code, branchPos, branchCond(bo, bi, static_cast<int32_t>(targetPos - branchPos)));
	}

	bool isIntRegister(const randomx::NativeRegisterFile& regs, const randomx::int_reg_t* ptr, int& index)
	{
		for (int i = 0; i < randomx::RegistersCount; ++i) {
			if (ptr == &regs.r[i]) {
				index = i;

				return true;
			}
		}

		index = -1;

		return false;
	}

	int intRegisterIndex(const randomx::NativeRegisterFile& regs, const randomx::int_reg_t* ptr)
	{
		int index = -1;
		isIntRegister(regs, ptr, index);

		return index;
	}

	int fpRegisterOffset(const randomx::NativeRegisterFile& regs, const rx_vec_f128* ptr)
	{
		const auto* base = reinterpret_cast<const uint8_t*>(&regs);

		for (int i = 0; i < randomx::RegisterCountFlt; ++i) {
			if (ptr == &regs.f[i] || ptr == &regs.e[i] || ptr == &regs.a[i]) {
				return static_cast<int>(reinterpret_cast<const uint8_t*>(ptr) - base);
			}
		}

		return -1;
	}

	void emitLoadNativeReg(uint8_t* code, size_t& pos, uint32_t dst, int reg)
	{
		emit32(code, pos, ld(dst, 15, static_cast<int32_t>(reg * sizeof(randomx::int_reg_t))));
	}

	void emitStoreNativeReg(uint8_t* code, size_t& pos, uint32_t src, int reg)
	{
		emit32(code, pos, std64(src, 15, static_cast<int32_t>(reg * sizeof(randomx::int_reg_t))));
	}

	void emitLoadLiteral64(uint8_t* code, size_t& pos, uint32_t dst, uint64_t value)
	{
		if (pos & 7U) {
			emit32(code, pos, PPC_NOP);
		}

		emit32(code, pos, PPC_BL_NEXT);
		emit32(code, pos, mflr(12));
		emit32(code, pos, ld(dst, 12, 12));
		emit32(code, pos, branch(12));
		emitPtr(code, pos, static_cast<uintptr_t>(value));
	}

	void emitLoadLiteral32(uint8_t* code, size_t& pos, uint32_t dst, uint32_t value)
	{
		emit32(code, pos, PPC_BL_NEXT);
		emit32(code, pos, mflr(7));
		emit32(code, pos, lwz(dst, 7, 12));
		emit32(code, pos, branch(8));
		emit32(code, pos, value);
	}

	void emitLoadLiteralPair32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, uint64_t value)
	{
		emit32(code, pos, PPC_BL_NEXT);
		emit32(code, pos, mflr(7));
		emit32(code, pos, lwz(lo, 7, 16));
		emit32(code, pos, lwz(hi, 7, 20));
		emit32(code, pos, branch(12));
		emit32(code, pos, static_cast<uint32_t>(value));
		emit32(code, pos, static_cast<uint32_t>(value >> 32));
	}

	void emitAddLiteral(uint8_t* code, size_t& pos, uint32_t reg, uint64_t value)
	{
		if (value == 0) {
			return;
		}

		emitLoadLiteral64(code, pos, 11, value);
		emit32(code, pos, add(reg, reg, 11));
	}

	void emitAddPair32(uint8_t* code, size_t& pos, uint32_t dstLo, uint32_t dstHi, uint32_t srcLo, uint32_t srcHi)
	{
		emit32(code, pos, addc(dstLo, dstLo, srcLo));
		emit32(code, pos, adde(dstHi, dstHi, srcHi));
	}

	void emitSubPair32(uint8_t* code, size_t& pos, uint32_t dstLo, uint32_t dstHi, uint32_t srcLo, uint32_t srcHi)
	{
		emit32(code, pos, subfc(dstLo, srcLo, dstLo));
		emit32(code, pos, subfe(dstHi, srcHi, dstHi));
	}

	void emitAddLiteralPair32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, uint64_t value)
	{
		if (value == 0) {
			return;
		}

		emitLoadLiteralPair32(code, pos, 10, 11, value);
		emitAddPair32(code, pos, lo, hi, 10, 11);
	}

	void emitLoadNativeReg32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, int reg)
	{
		const int32_t offset = static_cast<int32_t>(reg * sizeof(randomx::int_reg_t));
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, lwz(lo, 15, offset));
		emit32(code, pos, lwz(hi, 15, offset + 4));
#else
		emit32(code, pos, lwz(hi, 15, offset));
		emit32(code, pos, lwz(lo, 15, offset + 4));
#endif
	}

	void emitStoreNativeReg32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, int reg)
	{
		const int32_t offset = static_cast<int32_t>(reg * sizeof(randomx::int_reg_t));
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, stw(lo, 15, offset));
		emit32(code, pos, stw(hi, 15, offset + 4));
#else
		emit32(code, pos, stw(hi, 15, offset));
		emit32(code, pos, stw(lo, 15, offset + 4));
#endif
	}

	void emitLoadPairNativeOffset32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, int offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, lwz(lo, 15, offset));
		emit32(code, pos, lwz(hi, 15, offset + 4));
#else
		emit32(code, pos, lwz(hi, 15, offset));
		emit32(code, pos, lwz(lo, 15, offset + 4));
#endif
	}

	void emitStorePairNativeOffset32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, int offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, stw(lo, 15, offset));
		emit32(code, pos, stw(hi, 15, offset + 4));
#else
		emit32(code, pos, stw(hi, 15, offset));
		emit32(code, pos, stw(lo, 15, offset + 4));
#endif
	}

	void emitLoadPairStack32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, int offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, lwz(lo, 1, offset));
		emit32(code, pos, lwz(hi, 1, offset + 4));
#else
		emit32(code, pos, lwz(hi, 1, offset));
		emit32(code, pos, lwz(lo, 1, offset + 4));
#endif
	}

	void emitStorePairStack32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, int offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, stw(lo, 1, offset));
		emit32(code, pos, stw(hi, 1, offset + 4));
#else
		emit32(code, pos, stw(hi, 1, offset));
		emit32(code, pos, stw(lo, 1, offset + 4));
#endif
	}

	void emitLoadConfigPair32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, int offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, lwz(lo, 17, offset));
		emit32(code, pos, lwz(hi, 17, offset + 4));
#else
		emit32(code, pos, lwz(hi, 17, offset));
		emit32(code, pos, lwz(lo, 17, offset + 4));
#endif
	}

	void emitLoadSourcePair32(uint8_t* code, size_t& pos, const randomx::NativeRegisterFile& regs, const randomx::InstructionByteCode& ibc, uint32_t lo, uint32_t hi)
	{
		const int src = intRegisterIndex(regs, ibc.isrc);
		if (src >= 0) {
			emitLoadNativeReg32(code, pos, lo, hi, src);
		}
		else if (ibc.isrc == &ibc.imm) {
			emitLoadLiteralPair32(code, pos, lo, hi, ibc.imm);
		}
		else {
			emit32(code, pos, addi(lo, 0, 0));
			emit32(code, pos, addi(hi, 0, 0));
		}
	}

	void emitLoadScratchpadPair32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, uint32_t addr)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, lwz(lo, addr, 0));
		emit32(code, pos, lwz(hi, addr, 4));
#else
		emit32(code, pos, lwbrx(lo, 0, addr));
		emit32(code, pos, addi(7, addr, 4));
		emit32(code, pos, lwbrx(hi, 0, 7));
#endif
	}

	void emitStoreScratchpadPair32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, uint32_t addr)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, stw(lo, addr, 0));
		emit32(code, pos, stw(hi, addr, 4));
#else
		emit32(code, pos, stwbrx(lo, 0, addr));
		emit32(code, pos, addi(7, addr, 4));
		emit32(code, pos, stwbrx(hi, 0, 7));
#endif
	}

	void emitLoadScratchpadWord32(uint8_t* code, size_t& pos, uint32_t dst, uint32_t addr, int32_t offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, lwz(dst, addr, offset));
#else
		if (offset == 0) {
			emit32(code, pos, lwbrx(dst, 0, addr));
		}
		else {
			emit32(code, pos, addi(7, addr, offset));
			emit32(code, pos, lwbrx(dst, 0, 7));
		}
#endif
	}

	void emitScratchpadAddress32(uint8_t* code, size_t& pos, const randomx::NativeRegisterFile& regs, const randomx::InstructionByteCode& ibc)
	{
		const int src = intRegisterIndex(regs, ibc.isrc);
		if (src >= 0) {
			emitLoadNativeReg32(code, pos, 12, 7, src);
		}
		else {
			emit32(code, pos, addi(12, 0, 0));
		}

		if (static_cast<uint32_t>(ibc.imm) != 0) {
			emitLoadLiteral32(code, pos, 11, static_cast<uint32_t>(ibc.imm));
			emit32(code, pos, add(12, 12, 11));
		}

		emitLoadLiteral32(code, pos, 11, ibc.memMask);
		emit32(code, pos, andReg(12, 12, 11));
		emit32(code, pos, add(12, 16, 12));
	}

	void emitShiftLeftPairImm32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, uint32_t shift)
	{
		shift &= 63U;
		if (shift == 0) {
			return;
		}

		if (shift < 32U) {
			emit32(code, pos, rlwinm(7, lo, 32U - shift, shift, 31));
			emit32(code, pos, rlwinm(lo, lo, shift, 0, 31U - shift));
			emit32(code, pos, rlwinm(hi, hi, shift, 0, 31U - shift));
			emit32(code, pos, orReg(hi, hi, 7));
		}
		else if (shift == 32U) {
			emit32(code, pos, mr(hi, lo));
			emit32(code, pos, addi(lo, 0, 0));
		}
		else {
			emit32(code, pos, rlwinm(hi, lo, shift - 32U, 0, 63U - shift));
			emit32(code, pos, addi(lo, 0, 0));
		}
	}

	void emitRotateLeftPairImm32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, uint32_t shift)
	{
		shift &= 63U;
		if (shift == 0) {
			return;
		}

		if (shift < 32U) {
			emit32(code, pos, rlwinm(10, lo, shift, 0, 31U - shift));
			emit32(code, pos, rlwinm(11, hi, 32U - shift, shift, 31));
			emit32(code, pos, orReg(10, 10, 11));
			emit32(code, pos, rlwinm(11, hi, shift, 0, 31U - shift));
			emit32(code, pos, rlwinm(7, lo, 32U - shift, shift, 31));
			emit32(code, pos, orReg(11, hi, 7));
			emit32(code, pos, mr(lo, 10));
		}
		else if (shift == 32U) {
			emit32(code, pos, mr(7, lo));
			emit32(code, pos, mr(lo, hi));
			emit32(code, pos, mr(hi, 7));
		}
		else {
			const uint32_t s = shift - 32U;
			emit32(code, pos, rlwinm(10, hi, s, 0, 31U - s));
			emit32(code, pos, rlwinm(11, lo, 32U - s, s, 31));
			emit32(code, pos, orReg(10, 10, 11));
			emit32(code, pos, rlwinm(11, lo, s, 0, 31U - s));
			emit32(code, pos, rlwinm(7, hi, 32U - s, s, 31));
			emit32(code, pos, orReg(11, hi, 7));
			emit32(code, pos, mr(lo, 10));
		}
	}

	void emitRotateLeftPairVar32(uint8_t* code, size_t& pos, uint32_t lo, uint32_t hi, uint32_t shift)
	{
		emit32(code, pos, andiDot(12, shift, 63));
		const size_t doneBranch = pos;
		emit32(code, pos, branchCond(12, 2, 0));

		emit32(code, pos, andiDot(10, 12, 31));
		const size_t swapBranch = pos;
		emit32(code, pos, branchCond(12, 2, 0));

		emit32(code, pos, andiDot(7, 12, 32));
		const size_t ge32Branch = pos;
		emit32(code, pos, branchCond(4, 2, 0));

		emit32(code, pos, subfic(11, 10, 32));
		emit32(code, pos, slw(7, lo, 10));
		emit32(code, pos, srw(12, hi, 11));
		emit32(code, pos, orReg(7, 7, 12));
		emit32(code, pos, slw(12, hi, 10));
		emit32(code, pos, srw(11, lo, 11));
		emit32(code, pos, orReg(12, hi, 11));
		emit32(code, pos, mr(lo, 7));
		const size_t lt32EndBranch = pos;
		emit32(code, pos, branch(0));

		const size_t swapLabel = pos;
		patchBranchCond(code, swapBranch, swapLabel, 12, 2);
		emit32(code, pos, mr(7, lo));
		emit32(code, pos, mr(lo, hi));
		emit32(code, pos, mr(hi, 7));
		const size_t swapEndBranch = pos;
		emit32(code, pos, branch(0));

		const size_t ge32Label = pos;
		patchBranchCond(code, ge32Branch, ge32Label, 4, 2);
		emit32(code, pos, subfic(11, 10, 32));
		emit32(code, pos, slw(7, hi, 10));
		emit32(code, pos, srw(12, lo, 11));
		emit32(code, pos, orReg(7, 7, 12));
		emit32(code, pos, slw(12, lo, 10));
		emit32(code, pos, srw(11, hi, 11));
		emit32(code, pos, orReg(12, hi, 11));
		emit32(code, pos, mr(lo, 7));

		const size_t end = pos;
		patchBranchCond(code, doneBranch, end, 12, 2);
		patchBranch(code, lt32EndBranch, end);
		patchBranch(code, swapEndBranch, end);
	}

	void emitRotateRightLowWord32(uint8_t* code, size_t& pos, uint32_t out, uint32_t lo, uint32_t hi, uint32_t shift)
	{
		shift &= 63U;
		if (shift == 0) {
			emit32(code, pos, mr(out, lo));
		}
		else if (shift < 32U) {
			emit32(code, pos, rlwinm(out, lo, 32U - shift, shift, 31));
			emit32(code, pos, rlwinm(7, hi, shift, 0, 31U - shift));
			emit32(code, pos, orReg(out, out, 7));
		}
		else if (shift == 32U) {
			emit32(code, pos, mr(out, hi));
		}
		else {
			const uint32_t s = shift - 32U;
			emit32(code, pos, rlwinm(7, lo, s, 0, 31U - s));
			emit32(code, pos, rlwinm(out, hi, 32U - s, s, 31));
			emit32(code, pos, orReg(out, out, 7));
		}
	}

	void emitMulLowPair32(uint8_t* code, size_t& pos)
	{
		emit32(code, pos, mulhwu(7, 8, 10));
		emit32(code, pos, mullw(12, 8, 11));
		emit32(code, pos, add(7, 7, 12));
		emit32(code, pos, mullw(12, 9, 10));
		emit32(code, pos, add(9, 7, 12));
		emit32(code, pos, mullw(8, 8, 10));
	}

	void emitMulHighUnsignedPair32(uint8_t* code, size_t& pos)
	{
		emit32(code, pos, mullw(4, 9, 11));
		emit32(code, pos, mulhwu(3, 9, 11));

		emit32(code, pos, mulhwu(12, 8, 10));
		emit32(code, pos, mullw(6, 8, 11));
		emit32(code, pos, addc(12, 12, 6));
		emit32(code, pos, addi(5, 0, 0));
		emit32(code, pos, adde(5, 5, 5));
		emit32(code, pos, mullw(6, 9, 10));
		emit32(code, pos, addc(12, 12, 6));
		emit32(code, pos, addi(6, 0, 0));
		emit32(code, pos, adde(6, 6, 6));
		emit32(code, pos, add(5, 5, 6));

		emit32(code, pos, mulhwu(12, 8, 11));
		emit32(code, pos, mulhwu(6, 9, 10));
		emit32(code, pos, addc(8, 12, 6));
		emit32(code, pos, addi(9, 0, 0));
		emit32(code, pos, adde(9, 9, 9));
		emit32(code, pos, addc(8, 8, 4));
		emit32(code, pos, addi(6, 0, 0));
		emit32(code, pos, adde(6, 6, 6));
		emit32(code, pos, add(9, 9, 6));
		emit32(code, pos, addc(8, 8, 5));
		emit32(code, pos, addi(6, 0, 0));
		emit32(code, pos, adde(6, 6, 6));
		emit32(code, pos, add(9, 9, 6));
		emit32(code, pos, add(9, 9, 3));
	}

	void emitSignedHighMultiplyCorrection32(uint8_t* code, size_t& pos)
	{
		emit32(code, pos, andisDot(7, 4, 0x8000));
		const size_t aPositiveBranch = pos;
		emit32(code, pos, branchCond(12, 2, 0));
		emitSubPair32(code, pos, 8, 9, 5, 6);
		patchBranchCond(code, aPositiveBranch, pos, 12, 2);

		emit32(code, pos, andisDot(7, 6, 0x8000));
		const size_t bPositiveBranch = pos;
		emit32(code, pos, branchCond(12, 2, 0));
		emitSubPair32(code, pos, 8, 9, 3, 4);
		patchBranchCond(code, bPositiveBranch, pos, 12, 2);
	}

	void emitConvertScratchpadIntPair32(uint8_t* code, size_t& pos)
	{
		emitLoadScratchpadWord32(code, pos, 8, 12, 0);
		emitLoadScratchpadWord32(code, pos, 9, 12, 4);
		emitLoadLiteral32(code, pos, 10, 0x80000000U);
		emitLoadLiteral32(code, pos, 11, 0x43300000U);
		emitStorePairStack32(code, pos, 10, 11, FpConvertConstSlot32);

		emit32(code, pos, xorReg(8, 8, 10));
		emitStorePairStack32(code, pos, 8, 11, FpConvertSlot032);
		emit32(code, pos, lfd(0, 1, FpConvertConstSlot32));
		emit32(code, pos, lfd(1, 1, FpConvertSlot032));
		emit32(code, pos, fsub(1, 1, 0));

		emit32(code, pos, xorReg(9, 9, 10));
		emitStorePairStack32(code, pos, 9, 11, FpConvertSlot132);
		emit32(code, pos, lfd(2, 1, FpConvertSlot132));
		emit32(code, pos, fsub(2, 2, 0));
	}

	void emitMaskConvertedFpPair32(uint8_t* code, size_t& pos)
	{
		emit32(code, pos, stfd(1, 1, FpConvertSlot032));
		emit32(code, pos, stfd(2, 1, FpConvertSlot132));

		emitLoadPairStack32(code, pos, 8, 9, FpConvertSlot032);
		emitLoadLiteralPair32(code, pos, 10, 11, randomx::dynamicMantissaMask);
		emit32(code, pos, andReg(8, 8, 10));
		emit32(code, pos, andReg(9, 9, 11));
		emitLoadConfigPair32(code, pos, 10, 11, static_cast<int32_t>(offsetof(randomx::ProgramConfiguration, eMask)));
		emit32(code, pos, orReg(8, 8, 10));
		emit32(code, pos, orReg(9, 9, 11));
		emitStorePairStack32(code, pos, 8, 9, FpConvertSlot032);

		emitLoadPairStack32(code, pos, 8, 9, FpConvertSlot132);
		emitLoadLiteralPair32(code, pos, 10, 11, randomx::dynamicMantissaMask);
		emit32(code, pos, andReg(8, 8, 10));
		emit32(code, pos, andReg(9, 9, 11));
		emitLoadConfigPair32(code, pos, 10, 11, static_cast<int32_t>(offsetof(randomx::ProgramConfiguration, eMask) + sizeof(uint64_t)));
		emit32(code, pos, orReg(8, 8, 10));
		emit32(code, pos, orReg(9, 9, 11));
		emitStorePairStack32(code, pos, 8, 9, FpConvertSlot132);

		emit32(code, pos, lfd(1, 1, FpConvertSlot032));
		emit32(code, pos, lfd(2, 1, FpConvertSlot132));
	}

	void emitLoadScratchpad(uint8_t* code, size_t& pos, uint32_t dst, uint32_t addr)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, ldx(dst, 0, addr));
#else
		emit32(code, pos, ldbrx(dst, 0, addr));
#endif
	}

	void emitStoreScratchpad(uint8_t* code, size_t& pos, uint32_t src, uint32_t addr)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, stdx(src, 0, addr));
#else
		emit32(code, pos, stdbrx(src, 0, addr));
#endif
	}

	void emitLoadScratchpad32Signed(uint8_t* code, size_t& pos, uint32_t dst, uint32_t addr, int32_t offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, lwz(dst, addr, offset));
#else
		if (offset == 0) {
			emit32(code, pos, lwbrx(dst, 0, addr));
		}
		else {
			emit32(code, pos, addi(12, addr, offset));
			emit32(code, pos, lwbrx(dst, 0, 12));
		}
#endif
		emit32(code, pos, extsw(dst, dst));
	}

	void emitConvertScratchpadIntPair(uint8_t* code, size_t& pos)
	{
		emitLoadScratchpad32Signed(code, pos, 8, 10, 0);
		emitLoadScratchpad32Signed(code, pos, 9, 10, 4);
		emit32(code, pos, std64(8, 1, FpConvertSlot0));
		emit32(code, pos, std64(9, 1, FpConvertSlot1));
		emit32(code, pos, lfd(1, 1, FpConvertSlot0));
		emit32(code, pos, lfd(2, 1, FpConvertSlot1));
		emit32(code, pos, fcfid(1, 1));
		emit32(code, pos, fcfid(2, 2));
	}

	void emitLoadConfigEMask(uint8_t* code, size_t& pos, uint32_t dst, int32_t offset)
	{
#if XMRIG_PPC_JIT_LITTLE_ENDIAN
		emit32(code, pos, ld(dst, 17, offset));
#else
		if (offset == 0) {
			emit32(code, pos, ldbrx(dst, 0, 17));
		}
		else {
			emit32(code, pos, addi(12, 17, offset));
			emit32(code, pos, ldbrx(dst, 0, 12));
		}
#endif
	}

	void emitMaskConvertedFpPair(uint8_t* code, size_t& pos)
	{
		emit32(code, pos, stfd(1, 1, FpConvertSlot0));
		emit32(code, pos, stfd(2, 1, FpConvertSlot1));
		emit32(code, pos, ld(8, 1, FpConvertSlot0));
		emit32(code, pos, ld(9, 1, FpConvertSlot1));
		emitLoadLiteral64(code, pos, 10, randomx::dynamicMantissaMask);
		emit32(code, pos, andReg(8, 8, 10));
		emit32(code, pos, andReg(9, 9, 10));
		emitLoadConfigEMask(code, pos, 10, static_cast<int32_t>(offsetof(randomx::ProgramConfiguration, eMask)));
		emitLoadConfigEMask(code, pos, 11, static_cast<int32_t>(offsetof(randomx::ProgramConfiguration, eMask) + sizeof(uint64_t)));
		emit32(code, pos, orReg(8, 8, 10));
		emit32(code, pos, orReg(9, 9, 11));
		emit32(code, pos, std64(8, 1, FpConvertSlot0));
		emit32(code, pos, std64(9, 1, FpConvertSlot1));
		emit32(code, pos, lfd(1, 1, FpConvertSlot0));
		emit32(code, pos, lfd(2, 1, FpConvertSlot1));
	}

	void emitSetRoundingModeFromReg(uint8_t* code, size_t& pos, uint32_t mode)
	{
		emit32(code, pos, andiDot(10, mode, 2));
		const size_t bit1SetBranch = pos;
		emit32(code, pos, branchCond(4, 2, 0));

		emit32(code, pos, andiDot(10, mode, 1));
		const size_t mode1Branch = pos;
		emit32(code, pos, branchCond(4, 2, 0));

		emit32(code, pos, mtfsfi(7, 0));
		const size_t mode0EndBranch = pos;
		emit32(code, pos, branch(0));

		const size_t mode1Label = pos;
		patchBranchCond(code, mode1Branch, mode1Label, 4, 2);
		emit32(code, pos, mtfsfi(7, 3));
		const size_t mode1EndBranch = pos;
		emit32(code, pos, branch(0));

		const size_t bit1SetLabel = pos;
		patchBranchCond(code, bit1SetBranch, bit1SetLabel, 4, 2);
		emit32(code, pos, andiDot(10, mode, 1));
		const size_t mode3Branch = pos;
		emit32(code, pos, branchCond(4, 2, 0));

		emit32(code, pos, mtfsfi(7, 2));
		const size_t mode2EndBranch = pos;
		emit32(code, pos, branch(0));

		const size_t mode3Label = pos;
		patchBranchCond(code, mode3Branch, mode3Label, 4, 2);
		emit32(code, pos, mtfsfi(7, 1));

		const size_t end = pos;
		patchBranch(code, mode0EndBranch, end);
		patchBranch(code, mode1EndBranch, end);
		patchBranch(code, mode2EndBranch, end);
	}

	void emitScratchpadAddress(uint8_t* code, size_t& pos, const randomx::NativeRegisterFile& regs, const randomx::InstructionByteCode& ibc)
	{
		const int src = intRegisterIndex(regs, ibc.isrc);
		if (src >= 0) {
			emitLoadNativeReg(code, pos, 10, src);
		}
		else {
			emitLoadLiteral64(code, pos, 10, 0);
		}

		emitAddLiteral(code, pos, 10, ibc.imm);
		emitLoadLiteral64(code, pos, 11, ibc.memMask);
		emit32(code, pos, andReg(10, 10, 11));
		emit32(code, pos, add(10, 16, 10));
	}

	void emitCallBytecodeHelper(uint8_t* code, size_t& pos, randomx::InstructionByteCode* ibc)
	{
#if defined(__powerpc64__) || defined(__ppc64__)
		emitLoadLiteral64(code, pos, 3, reinterpret_cast<uintptr_t>(ibc));
#else
		emitLoadLiteral32(code, pos, 3, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ibc)));
#endif
		emit32(code, pos, mr(4, 16));
		emit32(code, pos, mr(5, 17));

#if defined(__powerpc64__) && defined(_CALL_ELF) && (_CALL_ELF == 1)
		const FunctionDescriptor descriptor = getFunctionDescriptor(&randomx::JitCompilerPPC::executeBytecodeInstruction);
		emitLoadLiteral64(code, pos, 2, descriptor.toc);
		emitLoadLiteral64(code, pos, 12, descriptor.entry);
#elif defined(__powerpc64__) || defined(__ppc64__)
		emitLoadLiteral64(code, pos, 12, reinterpret_cast<uintptr_t>(&randomx::JitCompilerPPC::executeBytecodeInstruction));
#else
		emitLoadLiteral32(code, pos, 12, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&randomx::JitCompilerPPC::executeBytecodeInstruction)));
#endif
		emit32(code, pos, mtctr(12));
		emit32(code, pos, PPC_BCTRL);
	}

	XMRIG_PPC_JIT_MAYBE_UNUSED bool emitNativeInstruction(uint8_t* code, size_t& pos, randomx::NativeRegisterFile& regs, randomx::InstructionByteCode* bytecode, int i, BranchFixup* fixups, int& fixupCount)
	{
		using randomx::InstructionType;

		auto& ibc = bytecode[i];
		auto dstReg = [&]() { return intRegisterIndex(regs, ibc.idst); };
		auto srcReg = [&]() { return intRegisterIndex(regs, ibc.isrc); };

		switch (ibc.type) {
		case InstructionType::IADD_RS:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0 || src < 0) {
				return false;
			}

			emitLoadNativeReg(code, pos, 8, dst);
			emitLoadNativeReg(code, pos, 9, src);
			if (ibc.shift) {
				emitLoadLiteral64(code, pos, 10, ibc.shift);
				emit32(code, pos, sld(9, 9, 10));
			}
			emit32(code, pos, add(8, 8, 9));
			emitAddLiteral(code, pos, 8, ibc.imm);
			emitStoreNativeReg(code, pos, 8, dst);
			return true;
		}

		case InstructionType::IADD_M:
		case InstructionType::ISUB_M:
		case InstructionType::IMUL_M:
		case InstructionType::IMULH_M:
		case InstructionType::ISMULH_M:
		case InstructionType::IXOR_M:
		{
			const int dst = dstReg();
			if (dst < 0) {
				return false;
			}

			emitScratchpadAddress(code, pos, regs, ibc);
			emitLoadScratchpad(code, pos, 9, 10);
			emitLoadNativeReg(code, pos, 8, dst);

			if (ibc.type == InstructionType::IADD_M) {
				emit32(code, pos, add(8, 8, 9));
			}
			else if (ibc.type == InstructionType::ISUB_M) {
				emit32(code, pos, subf(8, 9, 8));
			}
			else if (ibc.type == InstructionType::IMUL_M) {
				emit32(code, pos, mulld(8, 8, 9));
			}
			else if (ibc.type == InstructionType::IMULH_M) {
				emit32(code, pos, mulhdu(8, 8, 9));
			}
			else if (ibc.type == InstructionType::ISMULH_M) {
				emit32(code, pos, mulhd(8, 8, 9));
			}
			else {
				emit32(code, pos, xorReg(8, 8, 9));
			}

			emitStoreNativeReg(code, pos, 8, dst);
			return true;
		}

		case InstructionType::ISUB_R:
		case InstructionType::IMUL_R:
		case InstructionType::IMULH_R:
		case InstructionType::ISMULH_R:
		case InstructionType::IXOR_R:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0) {
				return false;
			}

			emitLoadNativeReg(code, pos, 8, dst);
			if (src >= 0) {
				emitLoadNativeReg(code, pos, 9, src);
			}
			else if (ibc.isrc == &ibc.imm) {
				emitLoadLiteral64(code, pos, 9, ibc.imm);
			}
			else {
				return false;
			}

			if (ibc.type == InstructionType::ISUB_R) {
				emit32(code, pos, subf(8, 9, 8));
			}
			else if (ibc.type == InstructionType::IMUL_R) {
				emit32(code, pos, mulld(8, 8, 9));
			}
			else if (ibc.type == InstructionType::IMULH_R) {
				emit32(code, pos, mulhdu(8, 8, 9));
			}
			else if (ibc.type == InstructionType::ISMULH_R) {
				emit32(code, pos, mulhd(8, 8, 9));
			}
			else {
				emit32(code, pos, xorReg(8, 8, 9));
			}

			emitStoreNativeReg(code, pos, 8, dst);
			return true;
		}

		case InstructionType::INEG_R:
		{
			const int dst = dstReg();
			if (dst < 0) {
				return false;
			}

			emitLoadNativeReg(code, pos, 8, dst);
			emit32(code, pos, neg(8, 8));
			emitStoreNativeReg(code, pos, 8, dst);
			return true;
		}

		case InstructionType::IROR_R:
		case InstructionType::IROL_R:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0) {
				return false;
			}

			emitLoadNativeReg(code, pos, 8, dst);
			if (src >= 0) {
				emitLoadNativeReg(code, pos, 9, src);
			}
			else if (ibc.isrc == &ibc.imm) {
				emitLoadLiteral64(code, pos, 9, ibc.imm);
			}
			else {
				return false;
			}

			if (ibc.type == InstructionType::IROR_R) {
				emit32(code, pos, neg(10, 9));
				emit32(code, pos, rotld(8, 8, 10));
			}
			else {
				emit32(code, pos, rotld(8, 8, 9));
			}

			emitStoreNativeReg(code, pos, 8, dst);
			return true;
		}

		case InstructionType::ISWAP_R:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0 || src < 0) {
				return false;
			}

			emitLoadNativeReg(code, pos, 8, dst);
			emitLoadNativeReg(code, pos, 9, src);
			emitStoreNativeReg(code, pos, 8, src);
			emitStoreNativeReg(code, pos, 9, dst);
			return true;
		}

		case InstructionType::FSWAP_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return false;
			}

			emit32(code, pos, ld(8, 15, dst));
			emit32(code, pos, ld(9, 15, dst + 8));
			emit32(code, pos, std64(8, 15, dst + 8));
			emit32(code, pos, std64(9, 15, dst));
			return true;
		}

		case InstructionType::FADD_R:
		case InstructionType::FSUB_R:
		case InstructionType::FMUL_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			const int src = fpRegisterOffset(regs, ibc.fsrc);
			if (dst < 0 || src < 0) {
				return false;
			}

			emit32(code, pos, lfd(0, 15, dst));
			emit32(code, pos, lfd(1, 15, src));
			if (ibc.type == InstructionType::FADD_R) {
				emit32(code, pos, fadd(0, 0, 1));
			}
			else if (ibc.type == InstructionType::FSUB_R) {
				emit32(code, pos, fsub(0, 0, 1));
			}
			else {
				emit32(code, pos, fmul(0, 0, 1));
			}
			emit32(code, pos, stfd(0, 15, dst));

			emit32(code, pos, lfd(0, 15, dst + 8));
			emit32(code, pos, lfd(1, 15, src + 8));
			if (ibc.type == InstructionType::FADD_R) {
				emit32(code, pos, fadd(0, 0, 1));
			}
			else if (ibc.type == InstructionType::FSUB_R) {
				emit32(code, pos, fsub(0, 0, 1));
			}
			else {
				emit32(code, pos, fmul(0, 0, 1));
			}
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::FADD_M:
		case InstructionType::FSUB_M:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return false;
			}

			emitScratchpadAddress(code, pos, regs, ibc);
			emitConvertScratchpadIntPair(code, pos);

			emit32(code, pos, lfd(0, 15, dst));
			if (ibc.type == InstructionType::FADD_M) {
				emit32(code, pos, fadd(0, 0, 1));
			}
			else {
				emit32(code, pos, fsub(0, 0, 1));
			}
			emit32(code, pos, stfd(0, 15, dst));

			emit32(code, pos, lfd(0, 15, dst + 8));
			if (ibc.type == InstructionType::FADD_M) {
				emit32(code, pos, fadd(0, 0, 2));
			}
			else {
				emit32(code, pos, fsub(0, 0, 2));
			}
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::FSCAL_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return false;
			}

			emitLoadLiteral64(code, pos, 10, 0x80F0000000000000ULL);
			emit32(code, pos, ld(8, 15, dst));
			emit32(code, pos, ld(9, 15, dst + 8));
			emit32(code, pos, xorReg(8, 8, 10));
			emit32(code, pos, xorReg(9, 9, 10));
			emit32(code, pos, std64(8, 15, dst));
			emit32(code, pos, std64(9, 15, dst + 8));
			return true;
		}

		case InstructionType::FDIV_M:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return false;
			}

			emitScratchpadAddress(code, pos, regs, ibc);
			emitConvertScratchpadIntPair(code, pos);
			emitMaskConvertedFpPair(code, pos);

			emit32(code, pos, lfd(0, 15, dst));
			emit32(code, pos, fdiv(0, 0, 1));
			emit32(code, pos, stfd(0, 15, dst));

			emit32(code, pos, lfd(0, 15, dst + 8));
			emit32(code, pos, fdiv(0, 0, 2));
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::FSQRT_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return false;
			}

			emit32(code, pos, lfd(0, 15, dst));
			emit32(code, pos, fsqrt(0, 0));
			emit32(code, pos, stfd(0, 15, dst));
			emit32(code, pos, lfd(0, 15, dst + 8));
			emit32(code, pos, fsqrt(0, 0));
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::CBRANCH:
		{
			const int dst = dstReg();
			if (dst < 0) {
				return false;
			}

			emitLoadNativeReg(code, pos, 8, dst);
			emitAddLiteral(code, pos, 8, ibc.imm);
			emitStoreNativeReg(code, pos, 8, dst);
			emitLoadLiteral64(code, pos, 9, ibc.memMask);
			emit32(code, pos, andDot(8, 8, 9));
			emit32(code, pos, branchCond(4, 2, 8));
			fixups[fixupCount++] = { pos, std::max<int>(0, ibc.target + 1) };
			emit32(code, pos, branch(0));
			return true;
		}

		case InstructionType::CFROUND:
		{
			const int src = srcReg();
			if (src < 0) {
				return false;
			}

			emitLoadNativeReg(code, pos, 8, src);
			if (ibc.imm) {
				emit32(code, pos, addi(9, 0, static_cast<int32_t>((64 - ibc.imm) & 63)));
				emit32(code, pos, rotld(8, 8, 9));
			}

			if (RandomX_CurrentConfig.Tweak_V2_CFROUND) {
				emit32(code, pos, andiDot(10, 8, 60));
				const size_t skipBranch = pos;
				emit32(code, pos, branchCond(4, 2, 0));
				emitSetRoundingModeFromReg(code, pos, 8);
				patchBranchCond(code, skipBranch, pos, 4, 2);
			}
			else {
				emitSetRoundingModeFromReg(code, pos, 8);
			}
			return true;
		}

		case InstructionType::ISTORE:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0 || src < 0) {
				return false;
			}

			emitScratchpadAddress(code, pos, regs, ibc);
			emitLoadNativeReg(code, pos, 8, src);
			emitStoreScratchpad(code, pos, 8, 10);
			return true;
		}

		case InstructionType::NOP:
			return true;

		default:
			emitCallBytecodeHelper(code, pos, &bytecode[i]);
			return true;
		}
	}

	XMRIG_PPC_JIT_MAYBE_UNUSED bool emitNativeInstruction32(uint8_t* code, size_t& pos, randomx::NativeRegisterFile& regs, randomx::InstructionByteCode* bytecode, int i, BranchFixup* fixups, int& fixupCount)
	{
		using randomx::InstructionType;

		auto& ibc = bytecode[i];
		auto dstReg = [&]() { return intRegisterIndex(regs, ibc.idst); };
		auto srcReg = [&]() { return intRegisterIndex(regs, ibc.isrc); };
		auto helper = [&]() {
			emitCallBytecodeHelper(code, pos, &bytecode[i]);
			return true;
		};

		switch (ibc.type) {
		case InstructionType::IADD_RS:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0 || src < 0) {
				return helper();
			}

			emitLoadNativeReg32(code, pos, 8, 9, dst);
			emitLoadNativeReg32(code, pos, 10, 11, src);
			emitShiftLeftPairImm32(code, pos, 10, 11, ibc.shift);
			emitAddPair32(code, pos, 8, 9, 10, 11);
			emitAddLiteralPair32(code, pos, 8, 9, ibc.imm);
			emitStoreNativeReg32(code, pos, 8, 9, dst);
			return true;
		}

		case InstructionType::IADD_M:
		case InstructionType::ISUB_M:
		case InstructionType::IMUL_M:
		case InstructionType::IMULH_M:
		case InstructionType::ISMULH_M:
		case InstructionType::IXOR_M:
		{
			const int dst = dstReg();
			if (dst < 0) {
				return helper();
			}

			emitScratchpadAddress32(code, pos, regs, ibc);
			emitLoadScratchpadPair32(code, pos, 10, 11, 12);
			emitLoadNativeReg32(code, pos, 8, 9, dst);

			if (ibc.type == InstructionType::IADD_M) {
				emitAddPair32(code, pos, 8, 9, 10, 11);
			}
			else if (ibc.type == InstructionType::ISUB_M) {
				emitSubPair32(code, pos, 8, 9, 10, 11);
			}
			else if (ibc.type == InstructionType::IMUL_M) {
				emitMulLowPair32(code, pos);
			}
			else if (ibc.type == InstructionType::IMULH_M) {
				emitMulHighUnsignedPair32(code, pos);
			}
			else if (ibc.type == InstructionType::ISMULH_M) {
				emitMulHighUnsignedPair32(code, pos);
				emitLoadNativeReg32(code, pos, 3, 4, dst);
				emitScratchpadAddress32(code, pos, regs, ibc);
				emitLoadScratchpadPair32(code, pos, 5, 6, 12);
				emitSignedHighMultiplyCorrection32(code, pos);
			}
			else {
				emit32(code, pos, xorReg(8, 8, 10));
				emit32(code, pos, xorReg(9, 9, 11));
			}

			emitStoreNativeReg32(code, pos, 8, 9, dst);
			return true;
		}

		case InstructionType::ISUB_R:
		case InstructionType::IMUL_R:
		case InstructionType::IMULH_R:
		case InstructionType::ISMULH_R:
		case InstructionType::IXOR_R:
		{
			const int dst = dstReg();
			if (dst < 0) {
				return helper();
			}

			emitLoadNativeReg32(code, pos, 8, 9, dst);
			emitLoadSourcePair32(code, pos, regs, ibc, 10, 11);

			if (ibc.type == InstructionType::ISUB_R) {
				emitSubPair32(code, pos, 8, 9, 10, 11);
			}
			else if (ibc.type == InstructionType::IMUL_R) {
				emitMulLowPair32(code, pos);
			}
			else if (ibc.type == InstructionType::IMULH_R) {
				emitMulHighUnsignedPair32(code, pos);
			}
			else if (ibc.type == InstructionType::ISMULH_R) {
				emitMulHighUnsignedPair32(code, pos);
				emitLoadNativeReg32(code, pos, 3, 4, dst);
				emitLoadSourcePair32(code, pos, regs, ibc, 5, 6);
				emitSignedHighMultiplyCorrection32(code, pos);
			}
			else {
				emit32(code, pos, xorReg(8, 8, 10));
				emit32(code, pos, xorReg(9, 9, 11));
			}

			emitStoreNativeReg32(code, pos, 8, 9, dst);
			return true;
		}

		case InstructionType::INEG_R:
		{
			const int dst = dstReg();
			if (dst < 0) {
				return helper();
			}

			emitLoadNativeReg32(code, pos, 8, 9, dst);
			emit32(code, pos, addi(10, 0, 0));
			emit32(code, pos, addi(11, 0, 0));
			emit32(code, pos, subfc(8, 8, 10));
			emit32(code, pos, subfe(9, 9, 11));
			emitStoreNativeReg32(code, pos, 8, 9, dst);
			return true;
		}

		case InstructionType::IROR_R:
		case InstructionType::IROL_R:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0) {
				return helper();
			}

			emitLoadNativeReg32(code, pos, 8, 9, dst);
			if (src >= 0) {
				emitLoadNativeReg32(code, pos, 10, 11, src);
				if (ibc.type == InstructionType::IROR_R) {
					emit32(code, pos, neg(10, 10));
				}
				emitRotateLeftPairVar32(code, pos, 8, 9, 10);
			}
			else if (ibc.isrc == &ibc.imm) {
				const uint32_t shift = static_cast<uint32_t>(ibc.imm & 63U);
				emitRotateLeftPairImm32(code, pos, 8, 9, ibc.type == InstructionType::IROR_R ? ((64U - shift) & 63U) : shift);
			}
			else {
				return helper();
			}
			emitStoreNativeReg32(code, pos, 8, 9, dst);
			return true;
		}

		case InstructionType::ISWAP_R:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0 || src < 0) {
				return helper();
			}

			emitLoadNativeReg32(code, pos, 8, 9, dst);
			emitLoadNativeReg32(code, pos, 10, 11, src);
			emitStoreNativeReg32(code, pos, 8, 9, src);
			emitStoreNativeReg32(code, pos, 10, 11, dst);
			return true;
		}

		case InstructionType::FSWAP_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return helper();
			}

			emit32(code, pos, lfd(0, 15, dst));
			emit32(code, pos, lfd(1, 15, dst + 8));
			emit32(code, pos, stfd(0, 15, dst + 8));
			emit32(code, pos, stfd(1, 15, dst));
			return true;
		}

		case InstructionType::FADD_R:
		case InstructionType::FSUB_R:
		case InstructionType::FMUL_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			const int src = fpRegisterOffset(regs, ibc.fsrc);
			if (dst < 0 || src < 0) {
				return helper();
			}

			emit32(code, pos, lfd(0, 15, dst));
			emit32(code, pos, lfd(1, 15, src));
			if (ibc.type == InstructionType::FADD_R) {
				emit32(code, pos, fadd(0, 0, 1));
			}
			else if (ibc.type == InstructionType::FSUB_R) {
				emit32(code, pos, fsub(0, 0, 1));
			}
			else {
				emit32(code, pos, fmul(0, 0, 1));
			}
			emit32(code, pos, stfd(0, 15, dst));

			emit32(code, pos, lfd(0, 15, dst + 8));
			emit32(code, pos, lfd(1, 15, src + 8));
			if (ibc.type == InstructionType::FADD_R) {
				emit32(code, pos, fadd(0, 0, 1));
			}
			else if (ibc.type == InstructionType::FSUB_R) {
				emit32(code, pos, fsub(0, 0, 1));
			}
			else {
				emit32(code, pos, fmul(0, 0, 1));
			}
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::FADD_M:
		case InstructionType::FSUB_M:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return helper();
			}

			emitScratchpadAddress32(code, pos, regs, ibc);
			emitConvertScratchpadIntPair32(code, pos);

			emit32(code, pos, lfd(0, 15, dst));
			if (ibc.type == InstructionType::FADD_M) {
				emit32(code, pos, fadd(0, 0, 1));
			}
			else {
				emit32(code, pos, fsub(0, 0, 1));
			}
			emit32(code, pos, stfd(0, 15, dst));

			emit32(code, pos, lfd(0, 15, dst + 8));
			if (ibc.type == InstructionType::FADD_M) {
				emit32(code, pos, fadd(0, 0, 2));
			}
			else {
				emit32(code, pos, fsub(0, 0, 2));
			}
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::FSCAL_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return helper();
			}

			emitLoadLiteralPair32(code, pos, 10, 11, 0x80F0000000000000ULL);
			emitLoadPairNativeOffset32(code, pos, 8, 9, dst);
			emit32(code, pos, xorReg(8, 8, 10));
			emit32(code, pos, xorReg(9, 9, 11));
			emitStorePairNativeOffset32(code, pos, 8, 9, dst);
			emitLoadPairNativeOffset32(code, pos, 8, 9, dst + 8);
			emit32(code, pos, xorReg(8, 8, 10));
			emit32(code, pos, xorReg(9, 9, 11));
			emitStorePairNativeOffset32(code, pos, 8, 9, dst + 8);
			return true;
		}

		case InstructionType::FDIV_M:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return helper();
			}

			emitScratchpadAddress32(code, pos, regs, ibc);
			emitConvertScratchpadIntPair32(code, pos);
			emitMaskConvertedFpPair32(code, pos);

			emit32(code, pos, lfd(0, 15, dst));
			emit32(code, pos, fdiv(0, 0, 1));
			emit32(code, pos, stfd(0, 15, dst));

			emit32(code, pos, lfd(0, 15, dst + 8));
			emit32(code, pos, fdiv(0, 0, 2));
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::FSQRT_R:
		{
			const int dst = fpRegisterOffset(regs, ibc.fdst);
			if (dst < 0) {
				return helper();
			}

			emit32(code, pos, lfd(0, 15, dst));
			emit32(code, pos, fsqrt(0, 0));
			emit32(code, pos, stfd(0, 15, dst));
			emit32(code, pos, lfd(0, 15, dst + 8));
			emit32(code, pos, fsqrt(0, 0));
			emit32(code, pos, stfd(0, 15, dst + 8));
			return true;
		}

		case InstructionType::CBRANCH:
		{
			const int dst = dstReg();
			if (dst < 0) {
				return helper();
			}

			emitLoadNativeReg32(code, pos, 8, 9, dst);
			emitAddLiteralPair32(code, pos, 8, 9, ibc.imm);
			emitStoreNativeReg32(code, pos, 8, 9, dst);
			emitLoadLiteral32(code, pos, 10, ibc.memMask);
			emit32(code, pos, andDot(8, 8, 10));
			emit32(code, pos, branchCond(4, 2, 8));
			fixups[fixupCount++] = { pos, std::max<int>(0, ibc.target + 1) };
			emit32(code, pos, branch(0));
			return true;
		}

		case InstructionType::CFROUND:
		{
			const int src = srcReg();
			if (src < 0) {
				return helper();
			}

			emitLoadNativeReg32(code, pos, 8, 9, src);
			emitRotateRightLowWord32(code, pos, 8, 8, 9, static_cast<uint32_t>(ibc.imm));

			if (RandomX_CurrentConfig.Tweak_V2_CFROUND) {
				emit32(code, pos, andiDot(10, 8, 60));
				const size_t skipBranch = pos;
				emit32(code, pos, branchCond(4, 2, 0));
				emitSetRoundingModeFromReg(code, pos, 8);
				patchBranchCond(code, skipBranch, pos, 4, 2);
			}
			else {
				emitSetRoundingModeFromReg(code, pos, 8);
			}
			return true;
		}

		case InstructionType::ISTORE:
		{
			const int dst = dstReg();
			const int src = srcReg();
			if (dst < 0 || src < 0) {
				return helper();
			}

			emitScratchpadAddress32(code, pos, regs, ibc);
			emitLoadNativeReg32(code, pos, 8, 9, src);
			emitStoreScratchpadPair32(code, pos, 8, 9, 12);
			return true;
		}

		case InstructionType::NOP:
			return true;

		default:
			return helper();
		}
	}

} // namespace

namespace randomx {

thread_local JitCompilerPPC* JitCompilerPPC::current = nullptr;

JitCompilerPPC::JitCompilerPPC(bool hugePagesEnable, bool) :
	hugePages(hugePagesJIT && hugePagesEnable)
{
	cacheView.jit = this;
	cacheView.datasetInit = &datasetInit;
	allocate();
}

JitCompilerPPC::~JitCompilerPPC()
{
	freePagedMemory(programCode, programAllocatedSize);
	freePagedMemory(code, allocatedSize);
}

void JitCompilerPPC::allocate()
{
	if (code) {
		return;
	}

	constexpr size_t requestedSize = 4096;
	allocatedSize = hugePages ? xmrig::VirtualMemory::align(requestedSize) : requestedSize;
	code = static_cast<uint8_t*>(allocExecutableMemory(allocatedSize, hugePages));
	emitTrampoline();
}

void JitCompilerPPC::allocateProgramCode()
{
	if (programCode) {
		return;
	}

	programAllocatedSize = hugePages ? xmrig::VirtualMemory::align(NativeProgramCodeSize) : NativeProgramCodeSize;
	programCode = static_cast<uint8_t*>(allocExecutableMemory(programAllocatedSize, hugePages));
}

void JitCompilerPPC::emitTrampoline()
{
	size_t pos = 0;
	ProgramFunc* entry = &programEntry;

#if defined(__powerpc64__) && defined(_CALL_ELF) && (_CALL_ELF == 1)
	const FunctionDescriptor descriptor = getFunctionDescriptor(entry);

	emit32(code, pos, mflr(11));         // r11 = caller return address
	emit32(code, pos, PPC_BL_NEXT);      // lr = current pc
	emit32(code, pos, mflr(12));         // r12 = literal base
	emit32(code, pos, ld(0, 12, 24));    // r0 = helper entry
	emit32(code, pos, ld(2, 12, 32));    // r2 = helper TOC
	emit32(code, pos, mtctr(0));
	emit32(code, pos, mtlr(11));         // restore caller return address
	emit32(code, pos, PPC_BCTR);
	emitPtr(code, pos, descriptor.entry);
	emitPtr(code, pos, descriptor.toc);

	functionDescriptor[0] = reinterpret_cast<uintptr_t>(code);
	functionDescriptor[1] = descriptor.toc;
	functionDescriptor[2] = 0;
#elif defined(__powerpc64__) || defined(__ppc64__)
	const uintptr_t target = reinterpret_cast<uintptr_t>(entry);

	emit32(code, pos, mflr(0));          // r0 = caller return address
	emit32(code, pos, PPC_BL_NEXT);      // lr = current pc
	emit32(code, pos, mflr(12));         // r12 = literal base
	emit32(code, pos, ld(12, 12, 24));   // r12 = helper entry
	emit32(code, pos, mtctr(12));
	emit32(code, pos, mtlr(0));          // restore caller return address
	emit32(code, pos, PPC_BCTR);
	emit32(code, pos, PPC_NOP);          // keep the literal 8-byte aligned
	emitPtr(code, pos, target);
#else
	const uintptr_t target = reinterpret_cast<uintptr_t>(entry);

	emit32(code, pos, mflr(0));          // r0 = caller return address
	emit32(code, pos, PPC_BL_NEXT);      // lr = current pc
	emit32(code, pos, mflr(12));         // r12 = literal base
	emit32(code, pos, lwz(12, 12, 20));  // r12 = helper entry
	emit32(code, pos, mtctr(12));
	emit32(code, pos, mtlr(0));          // restore caller return address
	emit32(code, pos, PPC_BCTR);
	emitPtr(code, pos, target);
#endif

	codeSize = pos;
	xmrig::VirtualMemory::flushInstructionCache(code, codeSize);
}

void JitCompilerPPC::generateProgram(Program& program, ProgramConfiguration& config, uint32_t)
{
	compileProgram(program, config, Mode::Full, 0);
}

void JitCompilerPPC::generateProgramLight(Program& program, ProgramConfiguration& config, uint32_t datasetOffset)
{
	compileProgram(program, config, Mode::Light, datasetOffset);
}

void JitCompilerPPC::compileProgram(Program& program, ProgramConfiguration& config, Mode mode, uint32_t datasetOffset)
{
	programConfig = config;
	executionMode = mode;
	lightDatasetOffset = datasetOffset;
	BytecodeMachine::compileProgram(program, bytecode, nativeRegs);
	compileNativeProgram();
}

void JitCompilerPPC::compileNativeProgram()
{
	nativeProgramAvailable = false;
	programCodeSize = 0;

#if defined(__powerpc64__) || defined(__ppc64__)
	allocateProgramCode();

#	ifdef XMRIG_SECURE_JIT
	xmrig::VirtualMemory::protectRW(programCode, programAllocatedSize);
#	endif

	size_t pos = 0;
	size_t labels[RANDOMX_PROGRAM_MAX_SIZE + 1]{};
	BranchFixup fixups[RANDOMX_PROGRAM_MAX_SIZE]{};
	int fixupCount = 0;

	emit32(programCode, pos, mflr(0));
	emit32(programCode, pos, std64(0, 1, 16));
	emit32(programCode, pos, stdu(1, 1, -NativeFrameSize));
	emit32(programCode, pos, std64(15, 1, SaveR15Offset));
	emit32(programCode, pos, std64(16, 1, SaveR16Offset));
	emit32(programCode, pos, std64(17, 1, SaveR17Offset));
#	if defined(_CALL_ELF) && (_CALL_ELF == 1)
	emit32(programCode, pos, std64(2, 1, SaveTocOffset));
#	endif
	emit32(programCode, pos, mr(15, 4));
	emit32(programCode, pos, mr(16, 5));
	emit32(programCode, pos, mr(17, 6));

	for (uint32_t i = 0; i < RandomX_CurrentConfig.ProgramSize; ++i) {
		labels[i] = pos;
		if (!emitNativeInstruction(programCode, pos, nativeRegs, bytecode, static_cast<int>(i), fixups, fixupCount)) {
			nativeProgramAvailable = false;

			return;
		}
	}

	labels[RandomX_CurrentConfig.ProgramSize] = pos;

	emit32(programCode, pos, ld(15, 1, SaveR15Offset));
	emit32(programCode, pos, ld(16, 1, SaveR16Offset));
	emit32(programCode, pos, ld(17, 1, SaveR17Offset));
#	if defined(_CALL_ELF) && (_CALL_ELF == 1)
	emit32(programCode, pos, ld(2, 1, SaveTocOffset));
#	endif
	emit32(programCode, pos, ld(0, 1, NativeFrameSize + 16));
	emit32(programCode, pos, addi(1, 1, NativeFrameSize));
	emit32(programCode, pos, mtlr(0));
	emit32(programCode, pos, PPC_BLR);

	for (int i = 0; i < fixupCount; ++i) {
		const int target = std::min<int>(fixups[i].target, RandomX_CurrentConfig.ProgramSize);
		const int32_t offset = static_cast<int32_t>(labels[target] - fixups[i].pos);
		write32(programCode, fixups[i].pos, branch(offset));
	}

	programCodeSize = pos;
	xmrig::VirtualMemory::flushInstructionCache(programCode, programCodeSize);

#	if defined(_CALL_ELF) && (_CALL_ELF == 1)
	const FunctionDescriptor descriptor = getFunctionDescriptor(&executeBytecodeInstruction);
	nativeProgramDescriptor[0] = reinterpret_cast<uintptr_t>(programCode);
	nativeProgramDescriptor[1] = descriptor.toc;
	nativeProgramDescriptor[2] = 0;
#	endif

	nativeProgramAvailable = true;
#else
	allocateProgramCode();

#	ifdef XMRIG_SECURE_JIT
	xmrig::VirtualMemory::protectRW(programCode, programAllocatedSize);
#	endif

	size_t pos = 0;
	size_t labels[RANDOMX_PROGRAM_MAX_SIZE + 1]{};
	BranchFixup fixups[RANDOMX_PROGRAM_MAX_SIZE]{};
	int fixupCount = 0;

	emit32(programCode, pos, mflr(0));
	emit32(programCode, pos, stw(0, 1, 4));
	emit32(programCode, pos, stwu(1, 1, -NativeFrameSize32));
	emit32(programCode, pos, stw(15, 1, SaveR15Offset32));
	emit32(programCode, pos, stw(16, 1, SaveR16Offset32));
	emit32(programCode, pos, stw(17, 1, SaveR17Offset32));
	emit32(programCode, pos, mr(15, 4));
	emit32(programCode, pos, mr(16, 5));
	emit32(programCode, pos, mr(17, 6));

	for (uint32_t i = 0; i < RandomX_CurrentConfig.ProgramSize; ++i) {
		labels[i] = pos;
		if (!emitNativeInstruction32(programCode, pos, nativeRegs, bytecode, static_cast<int>(i), fixups, fixupCount)) {
			nativeProgramAvailable = false;

			return;
		}
	}

	labels[RandomX_CurrentConfig.ProgramSize] = pos;

	emit32(programCode, pos, lwz(15, 1, SaveR15Offset32));
	emit32(programCode, pos, lwz(16, 1, SaveR16Offset32));
	emit32(programCode, pos, lwz(17, 1, SaveR17Offset32));
	emit32(programCode, pos, lwz(0, 1, NativeFrameSize32 + 4));
	emit32(programCode, pos, addi(1, 1, NativeFrameSize32));
	emit32(programCode, pos, mtlr(0));
	emit32(programCode, pos, PPC_BLR);

	for (int i = 0; i < fixupCount; ++i) {
		const int target = std::min<int>(fixups[i].target, RandomX_CurrentConfig.ProgramSize);
		const int32_t offset = static_cast<int32_t>(labels[target] - fixups[i].pos);
		write32(programCode, fixups[i].pos, branch(offset));
	}

	programCodeSize = pos;
	xmrig::VirtualMemory::flushInstructionCache(programCode, programCodeSize);
	nativeProgramAvailable = true;
#endif
}

JitCompilerPPC::NativeProgramFunc* JitCompilerPPC::getNativeProgramFunc()
{
#if defined(__powerpc64__) && defined(_CALL_ELF) && (_CALL_ELF == 1)
	return reinterpret_cast<NativeProgramFunc*>(nativeProgramDescriptor);
#else
	return reinterpret_cast<NativeProgramFunc*>(programCode);
#endif
}

ProgramFunc* JitCompilerPPC::getProgramFunc()
{
	current = this;

#ifdef XMRIG_SECURE_JIT
	enableExecution();
#endif

#if defined(__powerpc64__) && defined(_CALL_ELF) && (_CALL_ELF == 1)
	return reinterpret_cast<ProgramFunc*>(functionDescriptor);
#else
	return reinterpret_cast<ProgramFunc*>(code);
#endif
}

DatasetInitFunc* JitCompilerPPC::getDatasetInitFunc() const
{
	return &datasetInit;
}

void JitCompilerPPC::enableWriting() const
{
#ifdef XMRIG_SECURE_JIT
	if (code) {
		xmrig::VirtualMemory::protectRW(code, allocatedSize);
	}
	if (programCode) {
		xmrig::VirtualMemory::protectRW(programCode, programAllocatedSize);
	}
#endif
}

void JitCompilerPPC::enableExecution() const
{
#ifdef XMRIG_SECURE_JIT
	if (code) {
		xmrig::VirtualMemory::flushInstructionCache(code, codeSize);
		xmrig::VirtualMemory::protectRX(code, allocatedSize);
	}
	if (programCode) {
		xmrig::VirtualMemory::flushInstructionCache(programCode, programCodeSize);
		xmrig::VirtualMemory::protectRX(programCode, programAllocatedSize);
	}
#endif
}

void JitCompilerPPC::programEntry(RegisterFile& reg, MemoryRegisters& mem, uint8_t* scratchpad, uint64_t iterations)
{
	if (current) {
		current->execute(reg, mem, scratchpad, iterations);
	}
}

void JitCompilerPPC::executeBytecodeInstruction(InstructionByteCode* ibc, uint8_t* scratchpad, ProgramConfiguration* config)
{
	int pc = 0;
	BytecodeMachine::executeInstruction(*ibc, pc, scratchpad, *config);
}

void JitCompilerPPC::datasetInit(randomx_cache* cache, uint8_t* dataset, uint32_t startBlock, uint32_t endBlock)
{
	initDataset(cache, dataset, startBlock, endBlock);
}

void JitCompilerPPC::execute(RegisterFile& reg, MemoryRegisters& mem, uint8_t* scratchpad, uint64_t iterations)
{
	memset(nativeRegs.r, 0, sizeof(nativeRegs.r));

	for (unsigned i = 0; i < RegisterCountFlt; ++i) {
		nativeRegs.a[i] = rx_load_vec_f128(&reg.a[i].lo);
	}

	uint32_t spAddr0 = mem.mx;
	uint32_t spAddr1 = mem.ma;

	for (uint64_t ic = 0; ic < iterations; ++ic) {
		const uint64_t spMix = nativeRegs.r[programConfig.readReg0] ^ nativeRegs.r[programConfig.readReg1];
		spAddr0 ^= spMix;
		spAddr0 &= ScratchpadL3Mask64;
		spAddr1 ^= spMix >> 32;
		spAddr1 &= ScratchpadL3Mask64;

		for (unsigned i = 0; i < RegistersCount; ++i) {
			nativeRegs.r[i] ^= load64(scratchpad + spAddr0 + 8 * i);
		}

		for (unsigned i = 0; i < RegisterCountFlt; ++i) {
			nativeRegs.f[i] = rx_cvt_packed_int_vec_f128(scratchpad + spAddr1 + 8 * i);
		}

		for (unsigned i = 0; i < RegisterCountFlt; ++i) {
			nativeRegs.e[i] = BytecodeMachine::maskRegisterExponentMantissa(
				programConfig,
				rx_cvt_packed_int_vec_f128(scratchpad + spAddr1 + 8 * (RegisterCountFlt + i))
			);
		}

		if (nativeProgramAvailable) {
			getNativeProgramFunc()(bytecode, &nativeRegs, scratchpad, &programConfig);
		}
		else {
			BytecodeMachine::executeBytecode(bytecode, scratchpad, programConfig);
		}

		const uint64_t readPtr = mem.ma & CacheLineAlignMask;
		auto& mp = RandomX_CurrentConfig.Tweak_V2_PREFETCH ? mem.ma : mem.mx;
		mp ^= nativeRegs.r[programConfig.readReg2] ^ nativeRegs.r[programConfig.readReg3];

		if (executionMode == Mode::Full) {
			rx_prefetch_nta(mem.memory + (mp & CacheLineAlignMask));

			const uint64_t* datasetLine = reinterpret_cast<const uint64_t*>(mem.memory + readPtr);
			for (unsigned i = 0; i < RegistersCount; ++i) {
				nativeRegs.r[i] ^= datasetLine[i];
			}
		}
		else {
			cacheView.memory = mem.memory;

			int_reg_t rl[RegistersCount];
			const uint32_t itemNumber = static_cast<uint32_t>((lightDatasetOffset + readPtr) / CacheLineSize);
			initDatasetItem(&cacheView, reinterpret_cast<uint8_t*>(rl), itemNumber);

			for (unsigned i = 0; i < RegistersCount; ++i) {
				nativeRegs.r[i] ^= rl[i];
			}
		}

		std::swap(mem.mx, mem.ma);

		for (unsigned i = 0; i < RegistersCount; ++i) {
			store64(scratchpad + spAddr1 + 8 * i, nativeRegs.r[i]);
		}

		for (unsigned i = 0; i < RegisterCountFlt; ++i) {
			nativeRegs.f[i] = rx_xor_vec_f128(nativeRegs.f[i], nativeRegs.e[i]);
		}

		for (unsigned i = 0; i < RegisterCountFlt; ++i) {
			rx_store_vec_f128(reinterpret_cast<double*>(scratchpad + spAddr0 + 16 * i), nativeRegs.f[i]);
		}

		spAddr0 = 0;
		spAddr1 = 0;
	}

	for (unsigned i = 0; i < RegistersCount; ++i) {
		store64(&reg.r[i], nativeRegs.r[i]);
	}

	for (unsigned i = 0; i < RegisterCountFlt; ++i) {
		rx_store_vec_f128(&reg.f[i].lo, nativeRegs.f[i]);
	}

	for (unsigned i = 0; i < RegisterCountFlt; ++i) {
		rx_store_vec_f128(&reg.e[i].lo, nativeRegs.e[i]);
	}
}

} // namespace randomx
