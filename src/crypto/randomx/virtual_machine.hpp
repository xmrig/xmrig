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

#pragma once

#include <cstdint>
#include "crypto/randomx/common.hpp"
#include "crypto/randomx/program.hpp"

/* Global namespace for C binding */
class randomx_vm
{
public:
	virtual ~randomx_vm() = 0;
	virtual void setScratchpad(uint8_t *scratchpad) = 0;
	virtual void getFinalResult(void* out) = 0;
	virtual void hashAndFill(void* out, uint64_t (&fill_state)[8]) = 0;
	virtual void setDataset(randomx_dataset* dataset) { }
	virtual void setCache(randomx_cache* cache) { }
	virtual void initScratchpad(void* seed) = 0;
	virtual void run(void* seed) = 0;
	void resetRoundingMode();

	void setFlags(uint32_t flags) { vm_flags = flags; }
	uint32_t getFlags() const { return vm_flags; }

	randomx::RegisterFile *getRegisterFile() {
		return &reg;
	}

	const void* getScratchpad() {
		return scratchpad;
	}

	const randomx::Program& getProgram()
	{
		return program;
	}

protected:
	void initialize();
	alignas(64) randomx::Program program;
	alignas(64) randomx::RegisterFile reg;
	alignas(16) randomx::ProgramConfiguration config;
	randomx::MemoryRegisters mem;
	uint8_t* scratchpad = nullptr;
	union {
		randomx_cache* cachePtr = nullptr;
		randomx_dataset* datasetPtr;
	};
	uint64_t datasetOffset;
	uint32_t vm_flags;
};

namespace randomx {

	template<int softAes>
	class VmBase : public randomx_vm
	{
	public:
		~VmBase() override;
		void setScratchpad(uint8_t *scratchpad) override;
		void initScratchpad(void* seed) override;
		void getFinalResult(void* out) override;
		void hashAndFill(void* out, uint64_t (&fill_state)[8]) override;

	protected:
		void generateProgram(void* seed);
	};

}
