cmake_minimum_required (VERSION 2.8)
project (cpuid C)

add_definitions(/DVERSION="0.4.0")

set(HEADERS
    libcpuid.h
    libcpuid_types.h
    libcpuid_constants.h
    libcpuid_internal.h
amd_code_t.h
intel_code_t.h
recog_amd.h
recog_intel.h
asm-bits.h
libcpuid_util.h
    )

set(SOURCES
    cpuid_main.c
    asm-bits.c
    recog_amd.c
    recog_intel.c
    libcpuid_util.c
   )

add_library(cpuid STATIC
    ${HEADERS}
    ${SOURCES}
    )
