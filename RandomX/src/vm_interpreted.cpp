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

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cfloat>
#include "vm_interpreted.hpp"
#include "dataset.hpp"
#include "intrin_portable.h"
#include "reciprocal.h"

namespace randomx {

	static int_reg_t Zero = 0;

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::setDataset(randomx_dataset* dataset) {
		datasetPtr = dataset;
		mem.memory = dataset->memory;
	}

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::run(void* seed) {
		VmBase<Allocator, softAes>::generateProgram(seed);
		randomx_vm::initialize();
		execute();
	}

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::executeBytecode(int_reg_t(&r)[RegistersCount], rx_vec_f128(&f)[RegisterCountFlt], rx_vec_f128(&e)[RegisterCountFlt], rx_vec_f128(&a)[RegisterCountFlt]) {
		for (int pc = 0; pc < RANDOMX_PROGRAM_SIZE; ++pc) {
			executeBytecode(pc, r, f, e, a);
		}
	}

	template<class Allocator, bool softAes>
	FORCE_INLINE void* InterpretedVm<Allocator, softAes>::getScratchpadAddress(InstructionByteCode& ibc) {
		uint32_t addr = (*ibc.isrc + ibc.imm) & ibc.memMask;
		return scratchpad + addr;
	}

	template<class Allocator, bool softAes>
	FORCE_INLINE rx_vec_f128 InterpretedVm<Allocator, softAes>::maskRegisterExponentMantissa(rx_vec_f128 x) {
		const rx_vec_f128 xmantissaMask = rx_set_vec_f128(dynamicMantissaMask, dynamicMantissaMask);
		const rx_vec_f128 xexponentMask = rx_load_vec_f128((const double*)&config.eMask);
		x = rx_and_vec_f128(x, xmantissaMask);
		x = rx_or_vec_f128(x, xexponentMask);
		return x;
	}

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::executeBytecode(int& pc, int_reg_t(&r)[RegistersCount], rx_vec_f128(&f)[RegisterCountFlt], rx_vec_f128(&e)[RegisterCountFlt], rx_vec_f128(&a)[RegisterCountFlt]) {
		auto& ibc = byteCode[pc];
		switch (ibc.type)
		{
			case InstructionType::IADD_RS: {
				*ibc.idst += (*ibc.isrc << ibc.shift) + ibc.imm;
			} break;

			case InstructionType::IADD_M: {
				*ibc.idst += load64(getScratchpadAddress(ibc));
			} break;

			case InstructionType::ISUB_R: {
				*ibc.idst -= *ibc.isrc;
			} break;

			case InstructionType::ISUB_M: {
				*ibc.idst -= load64(getScratchpadAddress(ibc));
			} break;

			case InstructionType::IMUL_R: { //also handles IMUL_RCP
				*ibc.idst *= *ibc.isrc;
			} break;

			case InstructionType::IMUL_M: {
				*ibc.idst *= load64(getScratchpadAddress(ibc));
			} break;

			case InstructionType::IMULH_R: {
				*ibc.idst = mulh(*ibc.idst, *ibc.isrc);
			} break;

			case InstructionType::IMULH_M: {
				*ibc.idst = mulh(*ibc.idst, load64(getScratchpadAddress(ibc)));
			} break;

			case InstructionType::ISMULH_R: {
				*ibc.idst = smulh(unsigned64ToSigned2sCompl(*ibc.idst), unsigned64ToSigned2sCompl(*ibc.isrc));
			} break;

			case InstructionType::ISMULH_M: {
				*ibc.idst = smulh(unsigned64ToSigned2sCompl(*ibc.idst), unsigned64ToSigned2sCompl(load64(getScratchpadAddress(ibc))));
			} break;

			case InstructionType::INEG_R: {
				*ibc.idst = ~(*ibc.idst) + 1; //two's complement negative
			} break;

			case InstructionType::IXOR_R: {
				*ibc.idst ^= *ibc.isrc;
			} break;

			case InstructionType::IXOR_M: {
				*ibc.idst ^= load64(getScratchpadAddress(ibc));
			} break;

			case InstructionType::IROR_R: {
				*ibc.idst = rotr(*ibc.idst, *ibc.isrc & 63);
			} break;

			case InstructionType::IROL_R: {
				*ibc.idst = rotl(*ibc.idst, *ibc.isrc & 63);
			} break;

			case InstructionType::ISWAP_R: {
				int_reg_t temp = *ibc.isrc;
				*ibc.isrc = *ibc.idst;
				*ibc.idst = temp;
			} break;

			case InstructionType::FSWAP_R: {
				*ibc.fdst = rx_swap_vec_f128(*ibc.fdst);
			} break;

			case InstructionType::FADD_R: {
				*ibc.fdst = rx_add_vec_f128(*ibc.fdst, *ibc.fsrc);
			} break;

			case InstructionType::FADD_M: {
				rx_vec_f128 fsrc = rx_cvt_packed_int_vec_f128(getScratchpadAddress(ibc));
				*ibc.fdst = rx_add_vec_f128(*ibc.fdst, fsrc);
			} break;

			case InstructionType::FSUB_R: {
				*ibc.fdst = rx_sub_vec_f128(*ibc.fdst, *ibc.fsrc);
			} break;

			case InstructionType::FSUB_M: {
				rx_vec_f128 fsrc = rx_cvt_packed_int_vec_f128(getScratchpadAddress(ibc));
				*ibc.fdst = rx_sub_vec_f128(*ibc.fdst, fsrc);
			} break;

			case InstructionType::FSCAL_R: {
				const rx_vec_f128 mask = rx_set1_vec_f128(0x80F0000000000000);
				*ibc.fdst = rx_xor_vec_f128(*ibc.fdst, mask);
			} break;

			case InstructionType::FMUL_R: {
				*ibc.fdst = rx_mul_vec_f128(*ibc.fdst, *ibc.fsrc);
			} break;

			case InstructionType::FDIV_M: {
				rx_vec_f128 fsrc = maskRegisterExponentMantissa(rx_cvt_packed_int_vec_f128(getScratchpadAddress(ibc)));
				*ibc.fdst = rx_div_vec_f128(*ibc.fdst, fsrc);
			} break;

			case InstructionType::FSQRT_R: {
				*ibc.fdst = rx_sqrt_vec_f128(*ibc.fdst);
			} break;

			case InstructionType::CBRANCH: {
				*ibc.isrc += ibc.imm;
				if ((*ibc.isrc & ibc.memMask) == 0) {
					pc = ibc.target;
				}
			} break;

			case InstructionType::CFROUND: {
				rx_set_rounding_mode(rotr(*ibc.isrc, ibc.imm) % 4);
			} break;

			case InstructionType::ISTORE: {
				store64(scratchpad + ((*ibc.idst + ibc.imm) & ibc.memMask), *ibc.isrc);
			} break;

			case InstructionType::NOP: {
				//nothing
			} break;

			default:
				UNREACHABLE;
		}
	}

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::execute() {
		int_reg_t r[RegistersCount] = { 0 };
		rx_vec_f128 f[RegisterCountFlt];
		rx_vec_f128 e[RegisterCountFlt];
		rx_vec_f128 a[RegisterCountFlt];

		for(unsigned i = 0; i < RegisterCountFlt; ++i)
			a[i] = rx_load_vec_f128(&reg.a[i].lo);

		precompileProgram(r, f, e, a);

		uint32_t spAddr0 = mem.mx;
		uint32_t spAddr1 = mem.ma;

		for(unsigned ic = 0; ic < RANDOMX_PROGRAM_ITERATIONS; ++ic) {
			uint64_t spMix = r[config.readReg0] ^ r[config.readReg1];
			spAddr0 ^= spMix;
			spAddr0 &= ScratchpadL3Mask64;
			spAddr1 ^= spMix >> 32;
			spAddr1 &= ScratchpadL3Mask64;
			
			for (unsigned i = 0; i < RegistersCount; ++i)
				r[i] ^= load64(scratchpad + spAddr0 + 8 * i);

			for (unsigned i = 0; i < RegisterCountFlt; ++i)
				f[i] = rx_cvt_packed_int_vec_f128(scratchpad + spAddr1 + 8 * i);

			for (unsigned i = 0; i < RegisterCountFlt; ++i)
				e[i] = maskRegisterExponentMantissa(rx_cvt_packed_int_vec_f128(scratchpad + spAddr1 + 8 * (RegisterCountFlt + i)));

			executeBytecode(r, f, e, a);

			mem.mx ^= r[config.readReg2] ^ r[config.readReg3];
			mem.mx &= CacheLineAlignMask;
			datasetPrefetch(datasetOffset + mem.mx);
			datasetRead(datasetOffset + mem.ma, r);
			std::swap(mem.mx, mem.ma);

			for (unsigned i = 0; i < RegistersCount; ++i)
				store64(scratchpad + spAddr1 + 8 * i, r[i]);

			for (unsigned i = 0; i < RegisterCountFlt; ++i)
				f[i] = rx_xor_vec_f128(f[i], e[i]);

			for (unsigned i = 0; i < RegisterCountFlt; ++i)
				rx_store_vec_f128((double*)(scratchpad + spAddr0 + 16 * i), f[i]);

			spAddr0 = 0;
			spAddr1 = 0;
		}

		for (unsigned i = 0; i < RegistersCount; ++i)
			store64(&reg.r[i], r[i]);

		for (unsigned i = 0; i < RegisterCountFlt; ++i)
			rx_store_vec_f128(&reg.f[i].lo, f[i]);

		for (unsigned i = 0; i < RegisterCountFlt; ++i)
			rx_store_vec_f128(&reg.e[i].lo, e[i]);
	}

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::datasetRead(uint64_t address, int_reg_t(&r)[RegistersCount]) {
		uint64_t* datasetLine = (uint64_t*)(mem.memory + address);
		for (int i = 0; i < RegistersCount; ++i)
			r[i] ^= datasetLine[i];
	}

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::datasetPrefetch(uint64_t address) {
		rx_prefetch_nta(mem.memory + address);
	}

#include "instruction_weights.hpp"

	template<class Allocator, bool softAes>
	void InterpretedVm<Allocator, softAes>::precompileProgram(int_reg_t(&r)[RegistersCount], rx_vec_f128(&f)[RegisterCountFlt], rx_vec_f128(&e)[RegisterCountFlt], rx_vec_f128(&a)[RegisterCountFlt]) {
		int registerUsage[RegistersCount];
		for (unsigned i = 0; i < RegistersCount; ++i) {
			registerUsage[i] = -1;
		}
		for (unsigned i = 0; i < RANDOMX_PROGRAM_SIZE; ++i) {
			auto& instr = program(i);
			auto& ibc = byteCode[i];
			switch (instr.opcode) {
				CASE_REP(IADD_RS) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IADD_RS;
					ibc.idst = &r[dst];
					if (dst != RegisterNeedsDisplacement) {
						ibc.isrc = &r[src];
						ibc.shift = instr.getModShift();
						ibc.imm = 0;
					}
					else {
						ibc.isrc = &r[src];
						ibc.shift = instr.getModShift();
						ibc.imm = signExtend2sCompl(instr.getImm32());
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IADD_M) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IADD_M;
					ibc.idst = &r[dst];
					ibc.imm = signExtend2sCompl(instr.getImm32());
					if (src != dst) {
						ibc.isrc = &r[src];
						ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					}
					else {
						ibc.isrc = &Zero;
						ibc.memMask = ScratchpadL3Mask;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(ISUB_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::ISUB_R;
					ibc.idst = &r[dst];
					if (src != dst) {
						ibc.isrc = &r[src];
					}
					else {
						ibc.imm = signExtend2sCompl(instr.getImm32());
						ibc.isrc = &ibc.imm;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(ISUB_M) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::ISUB_M;
					ibc.idst = &r[dst];
					ibc.imm = signExtend2sCompl(instr.getImm32());
					if (src != dst) {
						ibc.isrc = &r[src];
						ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					}
					else {
						ibc.isrc = &Zero;
						ibc.memMask = ScratchpadL3Mask;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IMUL_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IMUL_R;
					ibc.idst = &r[dst];
					if (src != dst) {
						ibc.isrc = &r[src];
					}
					else {
						ibc.imm = signExtend2sCompl(instr.getImm32());
						ibc.isrc = &ibc.imm;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IMUL_M) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IMUL_M;
					ibc.idst = &r[dst];
					ibc.imm = signExtend2sCompl(instr.getImm32());
					if (src != dst) {
						ibc.isrc = &r[src];
						ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					}
					else {
						ibc.isrc = &Zero;
						ibc.memMask = ScratchpadL3Mask;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IMULH_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IMULH_R;
					ibc.idst = &r[dst];
					ibc.isrc = &r[src];
					registerUsage[dst] = i;
				} break;

				CASE_REP(IMULH_M) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IMULH_M;
					ibc.idst = &r[dst];
					ibc.imm = signExtend2sCompl(instr.getImm32());
					if (src != dst) {
						ibc.isrc = &r[src];
						ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					}
					else {
						ibc.isrc = &Zero;
						ibc.memMask = ScratchpadL3Mask;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(ISMULH_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::ISMULH_R;
					ibc.idst = &r[dst];
					ibc.isrc = &r[src];
					registerUsage[dst] = i;
				} break;

				CASE_REP(ISMULH_M) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::ISMULH_M;
					ibc.idst = &r[dst];
					ibc.imm = signExtend2sCompl(instr.getImm32());
					if (src != dst) {
						ibc.isrc = &r[src];
						ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					}
					else {
						ibc.isrc = &Zero;
						ibc.memMask = ScratchpadL3Mask;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IMUL_RCP) {
					uint64_t divisor = instr.getImm32();
					if (!isPowerOf2(divisor)) {
						auto dst = instr.dst % RegistersCount;
						ibc.type = InstructionType::IMUL_R;
						ibc.idst = &r[dst];
						ibc.imm = randomx_reciprocal(divisor);
						ibc.isrc = &ibc.imm;
						registerUsage[dst] = i;
					}
					else {
						ibc.type = InstructionType::NOP;
					}
				} break;

				CASE_REP(INEG_R) {
					auto dst = instr.dst % RegistersCount;
					ibc.type = InstructionType::INEG_R;
					ibc.idst = &r[dst];
					registerUsage[dst] = i;
				} break;

				CASE_REP(IXOR_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IXOR_R;
					ibc.idst = &r[dst];
					if (src != dst) {
						ibc.isrc = &r[src];
					}
					else {
						ibc.imm = signExtend2sCompl(instr.getImm32());
						ibc.isrc = &ibc.imm;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IXOR_M) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IXOR_M;
					ibc.idst = &r[dst];
					ibc.imm = signExtend2sCompl(instr.getImm32());
					if (src != dst) {
						ibc.isrc = &r[src];
						ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					}
					else {
						ibc.isrc = &Zero;
						ibc.memMask = ScratchpadL3Mask;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IROR_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IROR_R;
					ibc.idst = &r[dst];
					if (src != dst) {
						ibc.isrc = &r[src];
					}
					else {
						ibc.imm = instr.getImm32();
						ibc.isrc = &ibc.imm;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(IROL_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::IROL_R;
					ibc.idst = &r[dst];
					if (src != dst) {
						ibc.isrc = &r[src];
					}
					else {
						ibc.imm = instr.getImm32();
						ibc.isrc = &ibc.imm;
					}
					registerUsage[dst] = i;
				} break;

				CASE_REP(ISWAP_R) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					if (src != dst) {
						ibc.idst = &r[dst];
						ibc.isrc = &r[src];
						ibc.type = InstructionType::ISWAP_R;
						registerUsage[dst] = i;
						registerUsage[src] = i;
					}
					else {
						ibc.type = InstructionType::NOP;
					}
				} break;

				CASE_REP(FSWAP_R) {
					auto dst = instr.dst % RegistersCount;
					ibc.type = InstructionType::FSWAP_R;
					if (dst < RegisterCountFlt)
						ibc.fdst = &f[dst];
					else
						ibc.fdst = &e[dst - RegisterCountFlt];
				} break;

				CASE_REP(FADD_R) {
					auto dst = instr.dst % RegisterCountFlt;
					auto src = instr.src % RegisterCountFlt;
					ibc.type = InstructionType::FADD_R;
					ibc.fdst = &f[dst];
					ibc.fsrc = &a[src];
				} break;

				CASE_REP(FADD_M) {
					auto dst = instr.dst % RegisterCountFlt;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::FADD_M;
					ibc.fdst = &f[dst];
					ibc.isrc = &r[src];
					ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					ibc.imm = signExtend2sCompl(instr.getImm32());
				} break;

				CASE_REP(FSUB_R) {
					auto dst = instr.dst % RegisterCountFlt;
					auto src = instr.src % RegisterCountFlt;
					ibc.type = InstructionType::FSUB_R;
					ibc.fdst = &f[dst];
					ibc.fsrc = &a[src];
				} break;

				CASE_REP(FSUB_M) {
					auto dst = instr.dst % RegisterCountFlt;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::FSUB_M;
					ibc.fdst = &f[dst];
					ibc.isrc = &r[src];
					ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					ibc.imm = signExtend2sCompl(instr.getImm32());
				} break;

				CASE_REP(FSCAL_R) {
					auto dst = instr.dst % RegisterCountFlt;
					ibc.fdst = &f[dst];
					ibc.type = InstructionType::FSCAL_R;
				} break;

				CASE_REP(FMUL_R) {
					auto dst = instr.dst % RegisterCountFlt;
					auto src = instr.src % RegisterCountFlt;
					ibc.type = InstructionType::FMUL_R;
					ibc.fdst = &e[dst];
					ibc.fsrc = &a[src];
				} break;

				CASE_REP(FDIV_M) {
					auto dst = instr.dst % RegisterCountFlt;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::FDIV_M;
					ibc.fdst = &e[dst];
					ibc.isrc = &r[src];
					ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					ibc.imm = signExtend2sCompl(instr.getImm32());
				} break;

				CASE_REP(FSQRT_R) {
					auto dst = instr.dst % RegisterCountFlt;
					ibc.type = InstructionType::FSQRT_R;
					ibc.fdst = &e[dst];
				} break;

				CASE_REP(CBRANCH) {
					ibc.type = InstructionType::CBRANCH;
					//jump condition
					int reg = instr.dst % RegistersCount;
					ibc.isrc = &r[reg];
					ibc.target = registerUsage[reg];
					int shift = instr.getModCond() + ConditionOffset;
					const uint64_t conditionMask = ConditionMask << shift;
					ibc.imm = signExtend2sCompl(instr.getImm32()) | (1ULL << shift);
					if (ConditionOffset > 0 || shift > 0) //clear the bit below the condition mask - this limits the number of successive jumps to 2
						ibc.imm &= ~(1ULL << (shift - 1));
					ibc.memMask = ConditionMask << shift;
					//mark all registers as used
					for (unsigned j = 0; j < RegistersCount; ++j) {
						registerUsage[j] = i;
					}
				} break;

				CASE_REP(CFROUND) {
					auto src = instr.src % RegistersCount;
					ibc.isrc = &r[src];
					ibc.type = InstructionType::CFROUND;
					ibc.imm = instr.getImm32() & 63;
				} break;

				CASE_REP(ISTORE) {
					auto dst = instr.dst % RegistersCount;
					auto src = instr.src % RegistersCount;
					ibc.type = InstructionType::ISTORE;
					ibc.idst = &r[dst];
					ibc.isrc = &r[src];
					ibc.imm = signExtend2sCompl(instr.getImm32());
					if (instr.getModCond() < StoreL3Condition)
						ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
					else
						ibc.memMask = ScratchpadL3Mask;
				} break;

				CASE_REP(NOP) {
					ibc.type = InstructionType::NOP;
				} break;

				default:
					UNREACHABLE;
			}
		}
	}

	template class InterpretedVm<AlignedAllocator<CacheLineSize>, false>;
	template class InterpretedVm<AlignedAllocator<CacheLineSize>, true>;
	template class InterpretedVm<LargePageAllocator, false>;
	template class InterpretedVm<LargePageAllocator, true>;
}