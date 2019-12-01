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

#include "defyx.h"
#include "crypto/randomx/blake2/blake2.h"
#include "crypto/randomx/vm_interpreted.hpp"
#include "crypto/randomx/vm_interpreted_light.hpp"
#include "crypto/randomx/vm_compiled.hpp"
#include "crypto/randomx/vm_compiled_light.hpp"
#include "crypto/randomx/jit_compiler_x86_static.hpp"

#include <cassert>

extern "C" {
#include "yescrypt.h"
#include "KangarooTwelve.h"
} 

#define YESCRYPT_FLAGS YESCRYPT_RW
#define YESCRYPT_BASE_N 2048
#define YESCRYPT_R 8
#define YESCRYPT_P 1

RandomX_ConfigurationScala::RandomX_ConfigurationScala()
{
	ArgonMemory       = 131072;
        ArgonIterations   = 2;
	ArgonSalt         = "DefyXScala\x13";
        CacheAccesses     = 2;
        DatasetBaseSize   = 33554432;
        ProgramSize       = 64;
        ProgramIterations = 1024;
	ProgramCount      = 4;
	ScratchpadL3_Size = 262144;
	ScratchpadL2_Size = 131072;
	ScratchpadL1_Size = 65536;

	RANDOMX_FREQ_IADD_RS = 25;
	RANDOMX_FREQ_CBRANCH = 16;
}

RandomX_ConfigurationScala RandomX_ScalaConfig;

int sipesh(void *out, size_t outlen, const void *in, size_t inlen, const void *salt, size_t saltlen, unsigned int t_cost, unsigned int m_cost)
{
	yescrypt_local_t local;
	int retval;

	if (yescrypt_init_local(&local))
		return -1;
	retval = yescrypt_kdf(NULL, &local, (const uint8_t*)in, inlen, (const uint8_t*)salt, saltlen,
	    (uint64_t)YESCRYPT_BASE_N << m_cost, YESCRYPT_R, YESCRYPT_P,
	    t_cost, 0, YESCRYPT_FLAGS, (uint8_t*)out, outlen);
	if (yescrypt_free_local(&local))
		return -1;
	return retval;
}

int k12(const void *data, size_t length, void *hash)
{

  int kDo = KangarooTwelve((const unsigned char *)data, length, (unsigned char *)hash, 32, 0, 0);
  return kDo;
}


extern "C" {

	void defyx_calculate_hash(randomx_vm *machine, const void *input, size_t inputSize, void *output) {
		assert(machine != nullptr);
		assert(inputSize == 0 || input != nullptr);
		assert(output != nullptr);
		alignas(16) uint64_t tempHash[8];
		//rx_blake2b(tempHash, sizeof(tempHash), input, inputSize, nullptr, 0);
		sipesh(tempHash, sizeof(tempHash), input, inputSize, input, inputSize, 0, 0);
		k12(input, inputSize, tempHash);
		machine->initScratchpad(&tempHash);
		machine->resetRoundingMode();
		for (uint32_t chain = 0; chain < RandomX_CurrentConfig.ProgramCount - 1; ++chain) {
			machine->run(&tempHash);
			rx_blake2b(tempHash, sizeof(tempHash), machine->getRegisterFile(), sizeof(randomx::RegisterFile), nullptr, 0);
		}
		machine->run(&tempHash);
		machine->getFinalResult(output, RANDOMX_HASH_SIZE);
	}

	void defyx_calculate_hash_first(randomx_vm* machine, uint64_t (&tempHash)[8], const void* input, size_t inputSize) {
		//rx_blake2b(tempHash, sizeof(tempHash), input, inputSize, nullptr, 0);
		sipesh(tempHash, sizeof(tempHash), input, inputSize, input, inputSize, 0, 0);
		k12(input, inputSize, tempHash);
		machine->initScratchpad(tempHash);
	}

	void defyx_calculate_hash_next(randomx_vm* machine, uint64_t (&tempHash)[8], const void* nextInput, size_t nextInputSize, void* output) {
		machine->resetRoundingMode();
		for (uint32_t chain = 0; chain < RandomX_CurrentConfig.ProgramCount - 1; ++chain) {
			machine->run(&tempHash);
			rx_blake2b(tempHash, sizeof(tempHash), machine->getRegisterFile(), sizeof(randomx::RegisterFile), nullptr, 0);
		}
		machine->run(&tempHash);

		// Finish current hash and fill the scratchpad for the next hash at the same time
		//rx_blake2b(tempHash, sizeof(tempHash), nextInput, nextInputSize, nullptr, 0);
		sipesh(tempHash, sizeof(tempHash), nextInput, nextInputSize, nextInput, nextInputSize, 0, 0);
		k12(nextInput, nextInputSize, tempHash);
		machine->hashAndFill(output, RANDOMX_HASH_SIZE, tempHash);
	}

}
