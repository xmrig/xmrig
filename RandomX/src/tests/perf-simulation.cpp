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

#include "utility.hpp"
#include "../common.hpp"
#include "../aes_hash.hpp"
#include "../program.hpp"
#include "../blake2/blake2.h"
#include <algorithm>
#include <iomanip>

int analyze(randomx::Program& p);
int executeInOrder(randomx::Program& p, randomx::Program& original, bool print, int executionPorts, int memoryPorts, bool speculate, int pipeline);
int executeOutOfOrder(randomx::Program& p, randomx::Program& original, bool print, int executionPorts, int memoryPorts, bool speculate, int pipeline);

constexpr uint32_t DST_NOP = 0;
constexpr uint32_t DST_INT = 1;
constexpr uint32_t DST_FLT = 2;
constexpr uint32_t DST_MEM = 3;
constexpr uint32_t MASK_DST = 3;

constexpr uint32_t SRC_NOP = 0;
constexpr uint32_t SRC_INT = 4;
constexpr uint32_t SRC_FLT = 8;
constexpr uint32_t SRC_MEM = 12;
constexpr uint32_t MASK_SRC = 12;

constexpr uint32_t OP_CFROUND = 16;
constexpr uint32_t OP_SWAP = 32;
constexpr uint32_t OP_BRANCH = 48;
constexpr uint32_t MASK_EXT = 48;

constexpr uint32_t OP_FLOAT = 64;
constexpr uint32_t BRANCH_TARGET = 128;

//template<bool softAes>
void generate(randomx::Program& p, uint32_t nonce) {
	alignas(16) uint64_t hash[8];
	blake2b(hash, sizeof(hash), &nonce, sizeof(nonce), nullptr, 0);
	fillAes1Rx4<false>((void*)hash, sizeof(p), &p);
}

bool has(randomx::Instruction& instr, uint32_t mask, uint32_t prop) {
	return (instr.opcode & mask) == prop;
}

bool has(randomx::Instruction& instr, uint32_t prop) {
	return (instr.opcode & prop) != 0;
}

int main(int argc, char** argv) {
	int nonces, seed, executionPorts, memoryPorts, pipeline;
	bool print, reorder, speculate;
	readOption("--print", argc, argv, print);
	readOption("--reorder", argc, argv, reorder);
	readOption("--speculate", argc, argv, speculate);
	readIntOption("--nonces", argc, argv, nonces, 1);
	readIntOption("--seed", argc, argv, seed, 0);
	readIntOption("--executionPorts", argc, argv, executionPorts, 4);
	readIntOption("--memoryPorts", argc, argv, memoryPorts, 2);
	readIntOption("--pipeline", argc, argv, pipeline, 3);
	randomx::Program p, original;
	double totalCycles = 0.0;
	double jumpCount = 0;
	for (int i = 0; i < nonces; ++i) {
		generate(original, i ^ seed);
		memcpy(&p, &original, sizeof(p));
		jumpCount += analyze(p);
		totalCycles +=
			reorder
			?
			executeOutOfOrder(p, original, print, executionPorts, memoryPorts, speculate, pipeline)
			:
			executeInOrder(p, original, print, executionPorts, memoryPorts, speculate, pipeline);
	}
	totalCycles /= nonces;
	jumpCount /= nonces;
	std::cout << "Execution took " << totalCycles << " cycles per program" << std::endl;
	//std::cout << "Jump count: " << jumpCount << std::endl;
	return 0;
}

int executeInOrder(randomx::Program& p, randomx::Program& original, bool print, int executionPorts, int memoryPorts, bool speculate, int pipeline) {
	int cycle = pipeline - 1;
	int index = 0;
	int branchCount = 0;
	int int_reg_ready[randomx::RegistersCount] = { 0 };
	int flt_reg_ready[randomx::RegistersCount] = { 0 };
	//each workgroup takes 1 or 2 cycles (2 cycles if any instruction has a memory operand)
	while (index < RANDOMX_PROGRAM_SIZE) {
		int memoryAccesses = 0;
		bool hasRound = false;
		int workers = 0;
		//std::cout << "-----------" << std::endl;
		for (; workers < executionPorts && memoryAccesses < memoryPorts && index < RANDOMX_PROGRAM_SIZE; ++workers) {
			auto& instr = p(index);
			auto& origi = original(index);
			origi.dst %= randomx::RegistersCount;
			origi.src %= randomx::RegistersCount;

			//check dependencies
			if (has(instr, MASK_SRC, SRC_INT) && int_reg_ready[instr.src] > cycle)
				break;

			if (has(instr, MASK_SRC, SRC_MEM) && int_reg_ready[instr.src] > cycle - 1)
				break;

			if (has(instr, MASK_DST, DST_MEM) && int_reg_ready[instr.dst] > cycle - 1)
				break;

			if (has(instr, MASK_DST, DST_FLT) && flt_reg_ready[instr.dst] > cycle)
				break;

			if (has(instr, MASK_DST, DST_INT) && int_reg_ready[instr.dst] > cycle)
				break;

			if (hasRound && has(instr, OP_FLOAT))
				break;

			//execute
			index++;

			if (has(instr, MASK_EXT, OP_BRANCH)) {
				branchCount++;
			}

			if (has(instr, MASK_DST, DST_FLT))
				flt_reg_ready[instr.dst] = cycle + 1;

			if (has(instr, MASK_DST, DST_INT))
				int_reg_ready[instr.dst] = cycle + 1;

			if (has(instr, MASK_EXT, OP_SWAP)) {
				int_reg_ready[instr.src] = cycle + 1;
			}

			if (has(instr, MASK_EXT, OP_CFROUND))
				hasRound = true;

			if (has(instr, MASK_SRC, SRC_MEM) || has(instr, MASK_DST, DST_MEM)) {
				memoryAccesses++;
			}

			if (print)
				std::cout << std::setw(2) << (cycle + 1) << ": " << origi;

			//non-speculative execution must stall after branch
			if (!speculate && has(instr, MASK_EXT, OP_BRANCH)) {
				cycle += pipeline - 1;
				break;
			}
		}
		//std::cout << " workers: " << workers << std::endl;
		cycle++;
	}
	if (speculate) {
		//account for mispredicted branches
		int i = 0;
		while (branchCount--) {
			auto entropy = p.getEntropy(i / 8);
			entropy >> (i % 8) * 8;
			if ((entropy & 0xff) == 0) // 1/256 chance to flush the pipeline
				cycle += pipeline - 1;
		}
	}
	return cycle;
}

int executeOutOfOrder(randomx::Program& p, randomx::Program& original, bool print, int executionPorts, int memoryPorts, bool speculate, int pipeline) {
	int index = 0;
	int busyExecutionPorts[2 * RANDOMX_PROGRAM_SIZE] = { 0 };
	int busyMemoryPorts[2 * RANDOMX_PROGRAM_SIZE] = { 0 };
	int int_reg_ready[randomx::RegistersCount] = { 0 };
	int flt_reg_ready[randomx::RegistersCount] = { 0 };
	int fprcReady = 0;
	int lastBranch = 0;
	int branchCount = 0;
	for (; index < RANDOMX_PROGRAM_SIZE; ++index) {
		auto& instr = p(index);
		int retireCycle = pipeline - 1;

		//non-speculative execution cannot reorder across branches
		if (!speculate && !has(instr, MASK_EXT, OP_BRANCH))
			retireCycle = std::max(lastBranch + pipeline - 1, retireCycle);

		//check dependencies
		if (has(instr, MASK_SRC, SRC_INT)) {
			retireCycle = std::max(retireCycle, int_reg_ready[instr.src]);
			int_reg_ready[instr.src] = retireCycle;
		}

		if (has(instr, MASK_SRC, SRC_MEM)) {
			retireCycle = std::max(retireCycle, int_reg_ready[instr.src] + 1);
			//find free memory port
			while (busyMemoryPorts[retireCycle - 1] >= memoryPorts) {
				retireCycle++;
			}
			busyMemoryPorts[retireCycle - 1]++;
		}

		if (has(instr, MASK_DST, DST_FLT)) {
			retireCycle = std::max(retireCycle, flt_reg_ready[instr.dst]);
		}

		if (has(instr, MASK_DST, DST_INT)) {
			retireCycle = std::max(retireCycle, int_reg_ready[instr.dst]);
		}

		//floating point operations depend on the fprc register
		if (has(instr, OP_FLOAT))
			retireCycle = std::max(retireCycle, fprcReady);

		//execute
		if (has(instr, MASK_DST, DST_MEM)) {
			retireCycle = std::max(retireCycle, int_reg_ready[instr.dst] + 1);
			//find free memory port
			while (busyMemoryPorts[retireCycle - 1] >= memoryPorts) {
				retireCycle++;
			}
			busyMemoryPorts[retireCycle - 1]++;
			retireCycle++;
		}

		if (has(instr, MASK_DST, DST_FLT)) {
			//find free execution port
			do {
				retireCycle++;
			} while (busyExecutionPorts[retireCycle - 1] >= executionPorts);
			busyExecutionPorts[retireCycle - 1]++;
			flt_reg_ready[instr.dst] = retireCycle;
		}

		if (has(instr, MASK_DST, DST_INT)) {
			//find free execution port
			do {
				retireCycle++;
			} while (busyExecutionPorts[retireCycle - 1] >= executionPorts);
			busyExecutionPorts[retireCycle - 1]++;
			int_reg_ready[instr.dst] = retireCycle;
		}

		if (has(instr, MASK_EXT, OP_SWAP)) {
			int_reg_ready[instr.src] = retireCycle;
		}

		if (has(instr, MASK_EXT, OP_CFROUND)) {
			do {
				retireCycle++;
			} while (busyExecutionPorts[retireCycle - 1] >= executionPorts);
			busyExecutionPorts[retireCycle - 1]++;
			fprcReady = retireCycle;
		}

		if (has(instr, MASK_EXT, OP_BRANCH)) {
			/*if (!speculate && instr.mod == 1) { //simulated predication
				do {
					retireCycle++;
				} while (busyExecutionPorts[retireCycle - 1] >= executionPorts);
				busyExecutionPorts[retireCycle - 1]++;
				int_reg_ready[instr.dst] = retireCycle;
			}*/
			//else {
				lastBranch = std::max(lastBranch, retireCycle);
				branchCount++;
			//}
		}

		//print
		auto& origi = original(index);
		origi.dst %= randomx::RegistersCount;
		origi.src %= randomx::RegistersCount;
		if (print) {
			std::cout << std::setw(2) << retireCycle << ": " << origi;
			if (has(instr, MASK_EXT, OP_BRANCH)) {
				std::cout << "    jump: " << (int)instr.mod << std::endl;
			}
		}
	}
	int cycle = 0;
	for (int i = 0; i < randomx::RegistersCount; ++i) {
		cycle = std::max(cycle, int_reg_ready[i]);
	}
	for (int i = 0; i < randomx::RegistersCount; ++i) {
		cycle = std::max(cycle, flt_reg_ready[i]);
	}
	if (speculate) {
		//account for mispredicted branches
		int i = 0;
		while (branchCount--) {
			auto entropy = p.getEntropy(i / 8);
			entropy >> (i % 8) * 8;
			if ((entropy & 0xff) == 0) // 1/256 chance to flush the pipeline
				cycle += pipeline - 1;
		}
	}
	return cycle;
}

#include "../instruction_weights.hpp"

//old register selection
struct RegisterUsage {
	int32_t lastUsed;
	int32_t count;
};

inline int getConditionRegister(RegisterUsage(&registerUsage)[randomx::RegistersCount]) {
	int min = INT_MAX;
	int minCount = 0;
	int minIndex;
			//prefer registers that have been used as a condition register fewer times
		for (unsigned i = 0; i < randomx::RegistersCount; ++i) {
		if (registerUsage[i].lastUsed < min || (registerUsage[i].lastUsed == min && registerUsage[i].count < minCount)) {
			min = registerUsage[i].lastUsed;
			minCount = registerUsage[i].count;
			minIndex = i;
		}
	}
	return minIndex;
}

int analyze(randomx::Program& p) {
	int jumpCount = 0;
	RegisterUsage registerUsage[randomx::RegistersCount];
	for (unsigned i = 0; i < randomx::RegistersCount; ++i) {
		registerUsage[i].lastUsed = -1;
		registerUsage[i].count = 0;
	}
	for (unsigned i = 0; i < RANDOMX_PROGRAM_SIZE; ++i) {
		auto& instr = p(i);
		int opcode = instr.opcode;
		instr.opcode = 0;
		switch (opcode) {
			CASE_REP(IADD_RS) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_INT;
				instr.opcode |= DST_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IADD_M) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_MEM;
				instr.opcode |= DST_INT;
				if (instr.src != instr.dst) {
					instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
				}
				else {
					instr.imm32 &= randomx::ScratchpadL3Mask;
				}
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(ISUB_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				instr.opcode |= SRC_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(ISUB_M) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_MEM;
				instr.opcode |= DST_INT;
				if (instr.src != instr.dst) {
					instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
				}
				else {
					instr.imm32 &= randomx::ScratchpadL3Mask;
				}
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IMUL_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				instr.opcode |= SRC_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IMUL_M) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_MEM;
				instr.opcode |= DST_INT;
				if (instr.src != instr.dst) {
					instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
				}
				else {
					instr.imm32 &= randomx::ScratchpadL3Mask;
				}
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IMULH_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				instr.opcode |= SRC_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IMULH_M) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_MEM;
				instr.opcode |= DST_INT;
				if (instr.src != instr.dst) {
					instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
				}
				else {
					instr.imm32 &= randomx::ScratchpadL3Mask;
				}
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(ISMULH_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				instr.opcode |= SRC_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(ISMULH_M) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_MEM;
				instr.opcode |= DST_INT;
				if (instr.src != instr.dst) {
					instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
				}
				else {
					instr.imm32 &= randomx::ScratchpadL3Mask;
				}
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IMUL_RCP) {
				uint64_t divisor = instr.getImm32();
				if (!randomx::isPowerOf2(divisor)) {
					instr.dst = instr.dst % randomx::RegistersCount;
					instr.opcode |= DST_INT;
					registerUsage[instr.dst].lastUsed = i;
				}
			} break;

			CASE_REP(INEG_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IXOR_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				instr.opcode |= SRC_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IXOR_M) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_MEM;
				instr.opcode |= DST_INT;
				if (instr.src != instr.dst) {
					instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
				}
				else {
					instr.imm32 &= randomx::ScratchpadL3Mask;
				}
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IROR_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				instr.opcode |= SRC_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(IROL_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_INT;
				instr.opcode |= SRC_INT;
				registerUsage[instr.dst].lastUsed = i;
			} break;

			CASE_REP(ISWAP_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				if (instr.src != instr.dst) {
					instr.opcode |= DST_INT;
					instr.opcode |= SRC_INT;
					instr.opcode |= OP_SWAP;
					registerUsage[instr.dst].lastUsed = i;
					registerUsage[instr.src].lastUsed = i;
				}
			} break;

			CASE_REP(FSWAP_R) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.opcode |= DST_FLT;
			} break;

			CASE_REP(FADD_R) {
				instr.dst = instr.dst % randomx::RegisterCountFlt;
				instr.opcode |= DST_FLT;
				instr.opcode |= OP_FLOAT;
			} break;

			CASE_REP(FADD_M) {
				instr.dst = instr.dst % randomx::RegisterCountFlt;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_FLT;
				instr.opcode |= SRC_MEM;
				instr.opcode |= OP_FLOAT;
				instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
			} break;

			CASE_REP(FSUB_R) {
				instr.dst = instr.dst % randomx::RegisterCountFlt;
				instr.opcode |= DST_FLT;
				instr.opcode |= OP_FLOAT;
			} break;

			CASE_REP(FSUB_M) {
				instr.dst = instr.dst % randomx::RegisterCountFlt;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_FLT;
				instr.opcode |= SRC_MEM;
				instr.opcode |= OP_FLOAT;
				instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
			} break;

			CASE_REP(FSCAL_R) {
				instr.dst = instr.dst % randomx::RegisterCountFlt;
				instr.opcode |= DST_FLT;
			} break;

			CASE_REP(FMUL_R) {
				instr.dst = 4 + instr.dst % randomx::RegisterCountFlt;
				instr.opcode |= DST_FLT;
				instr.opcode |= OP_FLOAT;
			} break;

			CASE_REP(FDIV_M) {
				instr.dst = 4 + instr.dst % randomx::RegisterCountFlt;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_FLT;
				instr.opcode |= SRC_MEM;
				instr.opcode |= OP_FLOAT;
				instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
			} break;

			CASE_REP(FSQRT_R) {
				instr.dst = 4 + instr.dst % randomx::RegisterCountFlt;
				instr.opcode |= DST_FLT;
				instr.opcode |= OP_FLOAT;
			} break;

			CASE_REP(CBRANCH) {
				instr.opcode |= OP_BRANCH;
				instr.opcode |= DST_INT;
				//jump condition
				//int reg = getConditionRegister(registerUsage);
				int reg = instr.dst % randomx::RegistersCount;
				int target = registerUsage[reg].lastUsed;
				int offset = (i - target);
				instr.mod = offset;
				jumpCount += offset;
				p(target + 1).opcode |= BRANCH_TARGET;
				registerUsage[reg].count++;
				instr.dst = reg;
				//mark all registers as used
				for (unsigned j = 0; j < randomx::RegistersCount; ++j) {
					registerUsage[j].lastUsed = i;
				}
			} break;

			CASE_REP(CFROUND) {
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= SRC_INT;
				instr.opcode |= OP_CFROUND;
			} break;

			CASE_REP(ISTORE) {
				instr.dst = instr.dst % randomx::RegistersCount;
				instr.src = instr.src % randomx::RegistersCount;
				instr.opcode |= DST_MEM;
				if (instr.getModCond() < randomx::StoreL3Condition)
					instr.imm32 = (instr.getModMem() ? randomx::ScratchpadL1Mask : randomx::ScratchpadL2Mask);
				else
					instr.imm32 &= randomx::ScratchpadL3Mask;
			} break;

			CASE_REP(NOP) {

			} break;

		default:
			UNREACHABLE;
		}
	}
	return jumpCount;
}
