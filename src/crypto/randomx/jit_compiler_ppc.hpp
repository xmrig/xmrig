/*
Copyright (c) 2018-2020, tevador    <tevador@gmail.com>
Copyright (c) 2019-2020, SChernykh  <https://github.com/SChernykh>
Copyright (c) 2019-2020, XMRig      <https://github.com/xmrig>, <support@xmrig.com>
Copyright (c) 2026 PalindromicBreadLoaf <https://github.com/palindromicbreadloaf>

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

#include <cstddef>
#include <cstdint>

#include "crypto/randomx/bytecode_machine.hpp"
#include "crypto/randomx/common.hpp"
#include "crypto/randomx/dataset.hpp"

namespace randomx {

	class Program;
	struct ProgramConfiguration;
	class SuperscalarProgram;

	class JitCompilerPPC : private BytecodeMachine {
	public:
		using NativeProgramFunc = void(InstructionByteCode*, NativeRegisterFile*, uint8_t*, ProgramConfiguration*);

		explicit JitCompilerPPC(bool hugePagesEnable, bool optimizedInitDatasetEnable);
		~JitCompilerPPC();

		void prepare() {}
		void generateProgram(Program& program, ProgramConfiguration& config, uint32_t flags);
		void generateProgramLight(Program& program, ProgramConfiguration& config, uint32_t datasetOffset);

		template<size_t N>
		void generateSuperscalarHash(SuperscalarProgram(&programs)[N])
		{
			for (size_t i = 0; i < N && i < RANDOMX_CACHE_MAX_ACCESSES; ++i) {
				cacheView.programs[i] = programs[i];
			}
		}

		void generateDatasetInitCode() {}

		ProgramFunc* getProgramFunc();
		DatasetInitFunc* getDatasetInitFunc() const;
		uint8_t* getCode() { return code; }
		size_t getCodeSize() const { return codeSize; }

		void enableWriting() const;
		void enableExecution() const;

		static void executeBytecodeInstruction(InstructionByteCode* ibc, uint8_t* scratchpad, ProgramConfiguration* config);

	private:
		enum class Mode {
			Full,
			Light
		};

		void allocate();
		void allocateProgramCode();
		void compileProgram(Program& program, ProgramConfiguration& config, Mode mode, uint32_t datasetOffset);
		void compileNativeProgram();
		NativeProgramFunc* getNativeProgramFunc();
		void emitTrampoline();
		void execute(RegisterFile& reg, MemoryRegisters& mem, uint8_t* scratchpad, uint64_t iterations);

		static void programEntry(RegisterFile& reg, MemoryRegisters& mem, uint8_t* scratchpad, uint64_t iterations);
		static void datasetInit(randomx_cache* cache, uint8_t* dataset, uint32_t startBlock, uint32_t endBlock);

		static thread_local JitCompilerPPC* current;

		const bool hugePages;
		uint8_t* code = nullptr;
		size_t allocatedSize = 0;
		size_t codeSize = 0;
		uint8_t* programCode = nullptr;
		size_t programAllocatedSize = 0;
		size_t programCodeSize = 0;
		bool nativeProgramAvailable = false;

#if defined(__powerpc64__) && defined(_CALL_ELF) && (_CALL_ELF == 1)
		uintptr_t functionDescriptor[3]{};
		uintptr_t nativeProgramDescriptor[3]{};
#endif

		NativeRegisterFile nativeRegs;
		InstructionByteCode bytecode[RANDOMX_PROGRAM_MAX_SIZE]{};
		ProgramConfiguration programConfig{};
		randomx_cache cacheView{};
		uint32_t lightDatasetOffset = 0;
		Mode executionMode = Mode::Full;
	};
}
