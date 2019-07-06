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

#include "virtual_memory.hpp"

#include <stdexcept>

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#ifdef __APPLE__
#include <mach/vm_statistics.h>
#endif
#include <sys/types.h>
#include <sys/mman.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
std::string getErrorMessage(const char* function) {
	LPSTR messageBuffer = nullptr;
	size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
	std::string message(messageBuffer, size);
	LocalFree(messageBuffer);
	return std::string(function) + std::string(": ") + message;
}
#endif

void* allocExecutableMemory(std::size_t bytes) {
	void* mem;
#if defined(_WIN32) || defined(__CYGWIN__)
	mem = VirtualAlloc(nullptr, bytes, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
	if (mem == nullptr)
		throw std::runtime_error(getErrorMessage("allocExecutableMemory - VirtualAlloc"));
#else
	mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	if (mem == MAP_FAILED)
		throw std::runtime_error("allocExecutableMemory - mmap failed");
#endif
	return mem;
}

constexpr std::size_t align(std::size_t pos, std::size_t align) {
	return ((pos - 1) / align + 1) * align;
}

void* allocLargePagesMemory(std::size_t bytes) {
	void* mem;
#if defined(_WIN32) || defined(__CYGWIN__)
	auto pageMinimum = GetLargePageMinimum();
	if (pageMinimum > 0)
		mem = VirtualAlloc(NULL, align(bytes, pageMinimum), MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
	else
		throw std::runtime_error("allocLargePagesMemory - Large pages are not supported");
	if (mem == nullptr)
		throw std::runtime_error(getErrorMessage("allocLargePagesMemory - VirtualAlloc"));
#else
#ifdef __APPLE__
	mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
#elif defined(__FreeBSD__)
	mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER, -1, 0);
#else
	mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
#endif
	if (mem == MAP_FAILED)
		throw std::runtime_error("allocLargePagesMemory - mmap failed");
#endif
	return mem;
}

void freePagedMemory(void* ptr, std::size_t bytes) {
#if defined(_WIN32) || defined(__CYGWIN__)
	VirtualFree(ptr, 0, MEM_RELEASE);
#else
	munmap(ptr, bytes);
#endif
}
