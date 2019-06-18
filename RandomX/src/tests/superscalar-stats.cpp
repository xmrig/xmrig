#include <iostream>
#include <cstdint>
#include "../superscalar.hpp"
#include "../blake2_generator.hpp"

const uint8_t seed[32] = { 191, 182, 222, 175, 249, 89, 134, 104, 241, 68, 191, 62, 162, 166, 61, 64, 123, 191, 227, 193, 118, 60, 188, 53, 223, 133, 175, 24, 123, 230, 55, 74 };

int main() {

	constexpr int count = 1000000;
	int isnCounts[randomx::SuperscalarInstructionType::COUNT] = { 0 };
	int64_t asicLatency = 0;
	int64_t codesize = 0;
	int64_t cpuLatency = 0;
	int64_t macroOps = 0;
	int64_t mulCount = 0;
	int64_t size = 0;
	for (int i = 0; i < count; ++i) {
		randomx::SuperscalarProgram prog;
		randomx::Blake2Generator gen(seed, i);
		randomx::generateSuperscalar(prog, gen);
		asicLatency += prog.asicLatency;
		codesize += prog.codeSize;
		cpuLatency += prog.cpuLatency;
		macroOps += prog.macroOps;
		mulCount += prog.mulCount;
		size += prog.getSize();

		for (unsigned j = 0; j < prog.getSize(); ++j) {
			isnCounts[prog(j).opcode]++;
		}

		if ((i + 1) % (count / 100) == 0) {
			std::cout << "Completed " << ((i + 1) / (count / 100)) << "% ..." << std::endl;
		}
	}

	std::cout << "Avg. IPC: " << (macroOps / (double)cpuLatency) << std::endl;
	std::cout << "Avg. ASIC latency: " << (asicLatency / (double)count) << std::endl;
	std::cout << "Avg. CPU latency: " << (cpuLatency / (double)count) << std::endl;
	std::cout << "Avg. code size: " << (codesize / (double)count) << std::endl;
	std::cout << "Avg. x86 ops: " << (macroOps / (double)count) << std::endl;
	std::cout << "Avg. mul. count: " << (mulCount / (double)count) << std::endl;
	std::cout << "Avg. RandomX ops: " << (size / (double)count) << std::endl;

	std::cout << "Frequencies: " << std::endl;
	for (unsigned j = 0; j < randomx::SuperscalarInstructionType::COUNT; ++j) {
		std::cout << j << " " << isnCounts[j] << " " << isnCounts[j] / (double)size << std::endl;
	}

	return 0;
}