#ifndef __ASM_BITS_H__
#define __ASM_BITS_H__
#include "libcpuid.h"


#if defined(_MSC_VER)
#	define COMPILER_MICROSOFT
#elif defined(__GNUC__)
#	define COMPILER_GCC
#endif


#if defined(__x86_64__) || defined(_M_AMD64)
#	define PLATFORM_X64
#elif defined(__i386__) || defined(_M_IX86)
#	define PLATFORM_X86
#endif


#if (defined(COMPILER_GCC) && defined(PLATFORM_X64)) || defined(PLATFORM_X86)
#	define INLINE_ASM_SUPPORTED
#endif

int cpuid_exists_by_eflags(void);
void exec_cpuid(uint32_t *regs);
void busy_sse_loop(int cycles);

#endif 
