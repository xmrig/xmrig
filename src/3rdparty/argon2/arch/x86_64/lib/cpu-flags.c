#include <stdbool.h>
#include <stdint.h>


#include "cpu-flags.h"

#include <stdio.h>

#ifdef _MSC_VER
#   include <intrin.h>
#else
#   include <cpuid.h>
#endif

#ifndef bit_OSXSAVE
#   define bit_OSXSAVE (1 << 27)
#endif

#ifndef bit_SSE2
#   define bit_SSE2 (1 << 26)
#endif

#ifndef bit_SSSE3
#   define bit_SSSE3 (1 << 9)
#endif

#ifndef bit_AVX2
#   define bit_AVX2 (1 << 5)
#endif

#ifndef bit_AVX512F
#   define bit_AVX512F (1 << 16)
#endif

#ifndef bit_XOP
#   define bit_XOP (1 << 11)
#endif

#define PROCESSOR_INFO    (1)
#define EXTENDED_FEATURES (7)

#define EAX_Reg  (0)
#define EBX_Reg  (1)
#define ECX_Reg  (2)
#define EDX_Reg  (3)


enum {
    X86_64_FEATURE_SSE2     = (1 << 0),
    X86_64_FEATURE_SSSE3    = (1 << 1),
    X86_64_FEATURE_XOP      = (1 << 2),
    X86_64_FEATURE_AVX2     = (1 << 3),
    X86_64_FEATURE_AVX512F  = (1 << 4),
};

static unsigned int cpu_flags;


static inline void cpuid(uint32_t level, int32_t output[4])
{
#   ifdef _MSC_VER
    __cpuid(output, (int) level);
#   else
    __cpuid_count(level, 0, output[0], output[1], output[2], output[3]);
#   endif
}


static bool has_feature(uint32_t level, uint32_t reg, int32_t bit)
{
    int32_t cpu_info[4] = { 0 };
    cpuid(level, cpu_info);

    return (cpu_info[reg] & bit) != 0;
}


void cpu_flags_get(void)
{
    if (has_feature(PROCESSOR_INFO, EDX_Reg, bit_SSE2)) {
        cpu_flags |= X86_64_FEATURE_SSE2;
    }

    if (has_feature(PROCESSOR_INFO, ECX_Reg, bit_SSSE3)) {
        cpu_flags |= X86_64_FEATURE_SSSE3;
    }

    if (!has_feature(PROCESSOR_INFO, ECX_Reg, bit_OSXSAVE)) {
        return;
    }

    if (has_feature(EXTENDED_FEATURES, EBX_Reg, bit_AVX2)) {
        cpu_flags |= X86_64_FEATURE_AVX2;
    }

    if (has_feature(EXTENDED_FEATURES, EBX_Reg, bit_AVX512F)) {
        cpu_flags |= X86_64_FEATURE_AVX512F;
    }

    if (has_feature(0x80000001, ECX_Reg, bit_XOP)) {
        cpu_flags |= X86_64_FEATURE_XOP;
    }
}

int cpu_flags_have_sse2(void)
{
    return cpu_flags & X86_64_FEATURE_SSE2;
}

int cpu_flags_have_ssse3(void)
{
    return cpu_flags & X86_64_FEATURE_SSSE3;
}

int cpu_flags_have_xop(void)
{
    return cpu_flags & X86_64_FEATURE_XOP;
}

int cpu_flags_have_avx2(void)
{
    return cpu_flags & X86_64_FEATURE_AVX2;
}

int cpu_flags_have_avx512f(void)
{
    return cpu_flags & X86_64_FEATURE_AVX512F;
}

