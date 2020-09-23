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

#include <new>
#include "crypto/randomx/vm_compiled.hpp"

namespace randomx {

	template<int softAes>
	class CompiledLightVm : public CompiledVm<softAes>
	{
	public:
		void* operator new(size_t, void* ptr) { return ptr; }
		void operator delete(void*) {}

		void setCache(randomx_cache* cache) override;
		void setDataset(randomx_dataset* dataset) override { }
		void run(void* seed) override;

		using CompiledVm<softAes>::mem;
		using CompiledVm<softAes>::compiler;
		using CompiledVm<softAes>::program;
		using CompiledVm<softAes>::config;
		using CompiledVm<softAes>::cachePtr;
		using CompiledVm<softAes>::datasetOffset;
	};

	using CompiledLightVmDefault = CompiledLightVm<1>;
	using CompiledLightVmHardAes = CompiledLightVm<0>;
}
