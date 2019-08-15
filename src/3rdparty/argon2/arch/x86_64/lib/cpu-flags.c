#include "cpu-flags.h"

#include <cpuid.h>

enum {
    X86_64_FEATURE_SSE2     = (1 << 0),
    X86_64_FEATURE_SSSE3    = (1 << 1),
    X86_64_FEATURE_XOP      = (1 << 2),
    X86_64_FEATURE_AVX2     = (1 << 3),
    X86_64_FEATURE_AVX512F  = (1 << 4),
};

static unsigned int cpu_flags;

static unsigned int get_cpuid(int ext, unsigned int level, unsigned int *ebx,
                              unsigned int *ecx, unsigned int *edx)
{
    unsigned int eax;
    __cpuid(ext ? (0x80000000 | level) : level,
            eax, *ebx, *ecx, *edx);
    return eax;
}

static unsigned int get_cpuid_count(int ext, unsigned int level,
                                    unsigned int count, unsigned int *ebx,
                                    unsigned int *ecx, unsigned int *edx)
{
    unsigned int eax;
    __cpuid_count(ext ? (0x80000000 | level) : level,
                  count, eax, *ebx, *ecx, *edx);
    return 1;
}

void cpu_flags_get(void)
{
    unsigned int ebx, ecx, edx;
    unsigned int level, level_ext;

    cpu_flags = 0;
    level     = get_cpuid(0, 0, &ebx, &ecx, &edx);
    level_ext = get_cpuid(1, 0, &ebx, &ecx, &edx);

    if (level >= 1 && get_cpuid(0, 1, &ebx, &ecx, &edx)) {
        if (edx & (1 << 26)) {
            cpu_flags |= X86_64_FEATURE_SSE2;
        }
        if (ecx & (1 << 9)) {
            cpu_flags |= X86_64_FEATURE_SSSE3;
        }
    }
    if (level >= 7 && get_cpuid_count(0, 7, 0, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 5)) {
            cpu_flags |= X86_64_FEATURE_AVX2;
        }
        if (ebx & (1 << 16)) {
            cpu_flags |= X86_64_FEATURE_AVX512F;
        }
    }
    if (level_ext >= 1 && get_cpuid(1, 1, &ebx, &ecx, &edx)) {
        if (ecx & (1 << 11)) {
            cpu_flags |= X86_64_FEATURE_XOP;
        }
    }
    /* FIXME: check also OS support! */
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

