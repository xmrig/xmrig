/*
Copyright (c) 2019, tevador <tevador@gmail.com>

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

#include "bytecode_machine.hpp"
#include "reciprocal.h"

namespace randomx {

	const int_reg_t BytecodeMachine::zero = 0;

#define INSTR_CASE(x) case InstructionType::x: \
	exe_ ## x(ibc, pc, scratchpad, config); \
	break;

	void BytecodeMachine::executeInstruction(RANDOMX_EXE_ARGS) {
		switch (ibc.type)
		{
			INSTR_CASE(IADD_RS)
			INSTR_CASE(IADD_M)
			INSTR_CASE(ISUB_R)
			INSTR_CASE(ISUB_M)
			INSTR_CASE(IMUL_R)
			INSTR_CASE(IMUL_M)
			INSTR_CASE(IMULH_R)
			INSTR_CASE(IMULH_M)
			INSTR_CASE(ISMULH_R)
			INSTR_CASE(ISMULH_M)
			INSTR_CASE(INEG_R)
			INSTR_CASE(IXOR_R)
			INSTR_CASE(IXOR_M)
			INSTR_CASE(IROR_R)
			INSTR_CASE(IROL_R)
			INSTR_CASE(ISWAP_R)
			INSTR_CASE(FSWAP_R)
			INSTR_CASE(FADD_R)
			INSTR_CASE(FADD_M)
			INSTR_CASE(FSUB_R)
			INSTR_CASE(FSUB_M)
			INSTR_CASE(FSCAL_R)
			INSTR_CASE(FMUL_R)
			INSTR_CASE(FDIV_M)
			INSTR_CASE(FSQRT_R)
			INSTR_CASE(CBRANCH)
			INSTR_CASE(CFROUND)
			INSTR_CASE(ISTORE)

		case InstructionType::NOP:
			break;

		case InstructionType::IMUL_RCP: //executed as IMUL_R
		default:
			UNREACHABLE;
		}
	}

	void BytecodeMachine::compileInstruction(RANDOMX_GEN_ARGS) {
		int opcode = instr.opcode;

		if (opcode < RandomX_CurrentConfig.CEIL_IADD_RS) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IADD_RS;
			ibc.idst = &nreg->r[dst];
			if (dst != RegisterNeedsDisplacement) {
				ibc.isrc = &nreg->r[src];
				ibc.shift = instr.getModShift();
				ibc.imm = 0;
			}
			else {
				ibc.isrc = &nreg->r[src];
				ibc.shift = instr.getModShift();
				ibc.imm = signExtend2sCompl(instr.getImm32());
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IADD_M) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IADD_M;
			ibc.idst = &nreg->r[dst];
			ibc.imm = signExtend2sCompl(instr.getImm32());
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
				ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			}
			else {
				ibc.isrc = &zero;
				ibc.memMask = ScratchpadL3Mask;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_ISUB_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::ISUB_R;
			ibc.idst = &nreg->r[dst];
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
			}
			else {
				ibc.imm = signExtend2sCompl(instr.getImm32());
				ibc.isrc = &ibc.imm;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_ISUB_M) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::ISUB_M;
			ibc.idst = &nreg->r[dst];
			ibc.imm = signExtend2sCompl(instr.getImm32());
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
				ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			}
			else {
				ibc.isrc = &zero;
				ibc.memMask = ScratchpadL3Mask;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IMUL_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IMUL_R;
			ibc.idst = &nreg->r[dst];
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
			}
			else {
				ibc.imm = signExtend2sCompl(instr.getImm32());
				ibc.isrc = &ibc.imm;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IMUL_M) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IMUL_M;
			ibc.idst = &nreg->r[dst];
			ibc.imm = signExtend2sCompl(instr.getImm32());
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
				ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			}
			else {
				ibc.isrc = &zero;
				ibc.memMask = ScratchpadL3Mask;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IMULH_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IMULH_R;
			ibc.idst = &nreg->r[dst];
			ibc.isrc = &nreg->r[src];
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IMULH_M) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IMULH_M;
			ibc.idst = &nreg->r[dst];
			ibc.imm = signExtend2sCompl(instr.getImm32());
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
				ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			}
			else {
				ibc.isrc = &zero;
				ibc.memMask = ScratchpadL3Mask;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_ISMULH_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::ISMULH_R;
			ibc.idst = &nreg->r[dst];
			ibc.isrc = &nreg->r[src];
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_ISMULH_M) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::ISMULH_M;
			ibc.idst = &nreg->r[dst];
			ibc.imm = signExtend2sCompl(instr.getImm32());
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
				ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			}
			else {
				ibc.isrc = &zero;
				ibc.memMask = ScratchpadL3Mask;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IMUL_RCP) {
			uint64_t divisor = instr.getImm32();
			if (!isPowerOf2(divisor)) {
				auto dst = instr.dst % RegistersCount;
				ibc.type = InstructionType::IMUL_R;
				ibc.idst = &nreg->r[dst];
				ibc.imm = randomx_reciprocal(divisor);
				ibc.isrc = &ibc.imm;
				registerUsage[dst] = i;
			}
			else {
				ibc.type = InstructionType::NOP;
			}
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_INEG_R) {
			auto dst = instr.dst % RegistersCount;
			ibc.type = InstructionType::INEG_R;
			ibc.idst = &nreg->r[dst];
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IXOR_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IXOR_R;
			ibc.idst = &nreg->r[dst];
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
			}
			else {
				ibc.imm = signExtend2sCompl(instr.getImm32());
				ibc.isrc = &ibc.imm;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IXOR_M) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IXOR_M;
			ibc.idst = &nreg->r[dst];
			ibc.imm = signExtend2sCompl(instr.getImm32());
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
				ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			}
			else {
				ibc.isrc = &zero;
				ibc.memMask = ScratchpadL3Mask;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IROR_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IROR_R;
			ibc.idst = &nreg->r[dst];
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
			}
			else {
				ibc.imm = instr.getImm32();
				ibc.isrc = &ibc.imm;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_IROL_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::IROL_R;
			ibc.idst = &nreg->r[dst];
			if (src != dst) {
				ibc.isrc = &nreg->r[src];
			}
			else {
				ibc.imm = instr.getImm32();
				ibc.isrc = &ibc.imm;
			}
			registerUsage[dst] = i;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_ISWAP_R) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			if (src != dst) {
				ibc.idst = &nreg->r[dst];
				ibc.isrc = &nreg->r[src];
				ibc.type = InstructionType::ISWAP_R;
				registerUsage[dst] = i;
				registerUsage[src] = i;
			}
			else {
				ibc.type = InstructionType::NOP;
			}
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FSWAP_R) {
			auto dst = instr.dst % RegistersCount;
			ibc.type = InstructionType::FSWAP_R;
			if (dst < RegisterCountFlt)
				ibc.fdst = &nreg->f[dst];
			else
				ibc.fdst = &nreg->e[dst - RegisterCountFlt];
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FADD_R) {
			auto dst = instr.dst % RegisterCountFlt;
			auto src = instr.src % RegisterCountFlt;
			ibc.type = InstructionType::FADD_R;
			ibc.fdst = &nreg->f[dst];
			ibc.fsrc = &nreg->a[src];
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FADD_M) {
			auto dst = instr.dst % RegisterCountFlt;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::FADD_M;
			ibc.fdst = &nreg->f[dst];
			ibc.isrc = &nreg->r[src];
			ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			ibc.imm = signExtend2sCompl(instr.getImm32());
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FSUB_R) {
			auto dst = instr.dst % RegisterCountFlt;
			auto src = instr.src % RegisterCountFlt;
			ibc.type = InstructionType::FSUB_R;
			ibc.fdst = &nreg->f[dst];
			ibc.fsrc = &nreg->a[src];
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FSUB_M) {
			auto dst = instr.dst % RegisterCountFlt;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::FSUB_M;
			ibc.fdst = &nreg->f[dst];
			ibc.isrc = &nreg->r[src];
			ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			ibc.imm = signExtend2sCompl(instr.getImm32());
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FSCAL_R) {
			auto dst = instr.dst % RegisterCountFlt;
			ibc.fdst = &nreg->f[dst];
			ibc.type = InstructionType::FSCAL_R;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FMUL_R) {
			auto dst = instr.dst % RegisterCountFlt;
			auto src = instr.src % RegisterCountFlt;
			ibc.type = InstructionType::FMUL_R;
			ibc.fdst = &nreg->e[dst];
			ibc.fsrc = &nreg->a[src];
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FDIV_M) {
			auto dst = instr.dst % RegisterCountFlt;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::FDIV_M;
			ibc.fdst = &nreg->e[dst];
			ibc.isrc = &nreg->r[src];
			ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			ibc.imm = signExtend2sCompl(instr.getImm32());
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_FSQRT_R) {
			auto dst = instr.dst % RegisterCountFlt;
			ibc.type = InstructionType::FSQRT_R;
			ibc.fdst = &nreg->e[dst];
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_CBRANCH) {
			ibc.type = InstructionType::CBRANCH;
			//jump condition
			int creg = instr.dst % RegistersCount;
			ibc.idst = &nreg->r[creg];
			ibc.target = registerUsage[creg];
			int shift = instr.getModCond() + RandomX_CurrentConfig.JumpOffset;
			ibc.imm = signExtend2sCompl(instr.getImm32()) | (1ULL << shift);
			if (RandomX_CurrentConfig.JumpOffset > 0 || shift > 0) //clear the bit below the condition mask - this limits the number of successive jumps to 2
				ibc.imm &= ~(1ULL << (shift - 1));
			ibc.memMask = RandomX_CurrentConfig.ConditionMask_Calculated << shift;
			//mark all registers as used
			for (unsigned j = 0; j < RegistersCount; ++j) {
				registerUsage[j] = i;
			}
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_CFROUND) {
			auto src = instr.src % RegistersCount;
			ibc.isrc = &nreg->r[src];
			ibc.type = InstructionType::CFROUND;
			ibc.imm = instr.getImm32() & 63;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_ISTORE) {
			auto dst = instr.dst % RegistersCount;
			auto src = instr.src % RegistersCount;
			ibc.type = InstructionType::ISTORE;
			ibc.idst = &nreg->r[dst];
			ibc.isrc = &nreg->r[src];
			ibc.imm = signExtend2sCompl(instr.getImm32());
			if (instr.getModCond() < StoreL3Condition)
				ibc.memMask = (instr.getModMem() ? ScratchpadL1Mask : ScratchpadL2Mask);
			else
				ibc.memMask = ScratchpadL3Mask;
			return;
		}

		if (opcode < RandomX_CurrentConfig.CEIL_NOP) {
			ibc.type = InstructionType::NOP;
			return;
		}

		UNREACHABLE;
	}
}
