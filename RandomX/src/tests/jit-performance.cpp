#include "../aes_hash.hpp"
#include "../jit_compiler_x86.hpp"
#include "../program.hpp"
#include "utility.hpp"
#include "stopwatch.hpp"
#include "../blake2/blake2.h"
#include "../reciprocal.h"

int main(int argc, char** argv) {
	int count;
	readInt(argc, argv, count, 1000000);

	const char seed[] = "JIT performance test seed";
	uint8_t hash[64];

	blake2b(&hash, sizeof hash, &seed, sizeof seed, nullptr, 0);

	randomx::ProgramConfiguration config;

	randomx::Program program;
	randomx::JitCompilerX86 jit;

	std::cout << "Compiling " << count << " programs..." << std::endl;

	Stopwatch sw(true);

	for (int i = 0; i < count; ++i) {
		fillAes1Rx4<false>(hash, sizeof(program), &program);
		auto addressRegisters = program.getEntropy(12);
		config.readReg0 = 0 + (addressRegisters & 1);
		addressRegisters >>= 1;
		config.readReg1 = 2 + (addressRegisters & 1);
		addressRegisters >>= 1;
		config.readReg2 = 4 + (addressRegisters & 1);
		addressRegisters >>= 1;
		config.readReg3 = 6 + (addressRegisters & 1);
		jit.generateProgram(program, config);
	}

	std::cout << "Elapsed: " << sw.getElapsed() << " s" << std::endl;

	dump((const char*)jit.getProgramFunc(), randomx::CodeSize, "program.bin");
	return 0;
}