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
#include "../assembly_generator_x86.hpp"
#include "../superscalar.hpp"
#include "../aes_hash.hpp"
#include "../blake2/blake2.h"
#include "../program.hpp"

const uint8_t seed[32] = { 191, 182, 222, 175, 249, 89, 134, 104, 241, 68, 191, 62, 162, 166, 61, 64, 123, 191, 227, 193, 118, 60, 188, 53, 223, 133, 175, 24, 123, 230, 55, 74 };

const uint8_t blockTemplate_[] = {
		0x07, 0x07, 0xf7, 0xa4, 0xf0, 0xd6, 0x05, 0xb3, 0x03, 0x26, 0x08, 0x16, 0xba, 0x3f, 0x10, 0x90, 0x2e, 0x1a, 0x14,
		0x5a, 0xc5, 0xfa, 0xd3, 0xaa, 0x3a, 0xf6, 0xea, 0x44, 0xc1, 0x18, 0x69, 0xdc, 0x4f, 0x85, 0x3f, 0x00, 0x2b, 0x2e,
		0xea, 0x00, 0x00, 0x00, 0x00, 0x77, 0xb2, 0x06, 0xa0, 0x2c, 0xa5, 0xb1, 0xd4, 0xce, 0x6b, 0xbf, 0xdf, 0x0a, 0xca,
		0xc3, 0x8b, 0xde, 0xd3, 0x4d, 0x2d, 0xcd, 0xee, 0xf9, 0x5c, 0xd2, 0x0c, 0xef, 0xc1, 0x2f, 0x61, 0xd5, 0x61, 0x09
};

template<bool softAes>
void generateAsm(uint32_t nonce) {
	alignas(16) uint64_t hash[8];
	uint8_t blockTemplate[sizeof(blockTemplate_)];
	memcpy(blockTemplate, blockTemplate_, sizeof(blockTemplate));
	store32(blockTemplate + 39, nonce);
	blake2b(hash, sizeof(hash), blockTemplate, sizeof(blockTemplate), nullptr, 0);
	uint8_t scratchpad[randomx::ScratchpadSize];
	fillAes1Rx4<softAes>((void*)hash, randomx::ScratchpadSize, scratchpad);
	randomx::AssemblyGeneratorX86 asmX86;
	randomx::Program p;
	fillAes1Rx4<softAes>(hash, sizeof(p), &p);
	asmX86.generateProgram(p);
	asmX86.printCode(std::cout);
}

template<bool softAes>
void generateNative(uint32_t nonce) {
	alignas(16) uint64_t hash[8];
	uint8_t blockTemplate[sizeof(blockTemplate_)];
	memcpy(blockTemplate, blockTemplate_, sizeof(blockTemplate));
	store32(blockTemplate + 39, nonce);
	blake2b(hash, sizeof(hash), blockTemplate, sizeof(blockTemplate), nullptr, 0);
	uint8_t scratchpad[randomx::ScratchpadSize];
	fillAes1Rx4<softAes>((void*)hash, randomx::ScratchpadSize, scratchpad);
	alignas(16) randomx::Program prog;
	fillAes1Rx4<softAes>((void*)hash, sizeof(prog), &prog);
	std::cout << prog << std::endl;
}

void printUsage(const char* executable) {
	std::cout << "Usage: " << executable << " [OPTIONS]" << std::endl;
	std::cout << "Supported options:" << std::endl;
	std::cout << "  --softAes         use software AES (default: x86 AES-NI)" << std::endl;
	std::cout << "  --nonce  N        seed nonce (default: 1000)" << std::endl;
	std::cout << "  --genAsm          generate x86-64 asm code for nonce N" << std::endl;
	std::cout << "  --genNative       generate RandomX code for nonce N" << std::endl;
	std::cout << "  --genSuperscalar  generate superscalar program for nonce N" << std::endl;
}

int main(int argc, char** argv) {
	bool softAes, genAsm, genNative, genSuperscalar;
	int nonce;

	readOption("--softAes", argc, argv, softAes);
	readOption("--genAsm", argc, argv, genAsm);
	readIntOption("--nonce", argc, argv, nonce, 1000);
	readOption("--genNative", argc, argv, genNative);
	readOption("--genSuperscalar", argc, argv, genSuperscalar);

	if (genSuperscalar) {
		randomx::SuperscalarProgram p;
		randomx::Blake2Generator gen(seed, nonce);
		randomx::generateSuperscalar(p, gen);
		randomx::AssemblyGeneratorX86 asmX86;
		asmX86.generateAsm(p);
		asmX86.printCode(std::cout);
		return 0;
	}

	if (genAsm) {
		if (softAes)
			generateAsm<true>(nonce);
		else
			generateAsm<false>(nonce);
		return 0;
	}

	if (genNative) {
		if (softAes)
			generateNative<true>(nonce);
		else
			generateNative<false>(nonce);
		return 0;
	}

	printUsage(argv[0]);
	return 0;
}