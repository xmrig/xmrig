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

#include <new>
#include "crypto/randomx/allocator.hpp"
#include "crypto/randomx/intrin_portable.h"
#include "crypto/randomx/virtual_memory.hpp"
#include "crypto/randomx/common.hpp"

namespace randomx {

	template<size_t alignment>
	void* AlignedAllocator<alignment>::allocMemory(size_t count) {
		void *mem = rx_aligned_alloc(count, alignment);
		if (mem == nullptr)
			throw std::bad_alloc();
		return mem;
	}

	template<size_t alignment>
	void AlignedAllocator<alignment>::freeMemory(void* ptr, size_t count) {
		rx_aligned_free(ptr);
	}

	template struct AlignedAllocator<CacheLineSize>;

	void* LargePageAllocator::allocMemory(size_t count) {
		return allocLargePagesMemory(count);
	}

	void LargePageAllocator::freeMemory(void* ptr, size_t count) {
		freePagedMemory(ptr, count);
	};

}
