cmake_minimum_required (VERSION 2.8)
project (cpuid C)

add_definitions(/DVERSION="0.4.0")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Os")

set(HEADERS
    amd_code_t.h
    asm-bits.h
    intel_code_t.h
    libcpuid.h
    libcpuid_constants.h
    libcpuid_internal.h
    libcpuid_types.h
    libcpuid_util.h
    rdtsc.h
    recog_amd.h
    recog_intel.h
    )

set(SOURCES
    asm-bits.c
    cpuid_main.c
    libcpuid_util.c
    msrdriver.c
    rdmsr.c
    rdtsc.c
    recog_amd.c
    recog_intel.c
	)

if (CMAKE_CL_64)
    enable_language(ASM_MASM)
    set(SOURCES_ASM masm-x64.asm)
endif()

add_library(cpuid STATIC
    ${HEADERS}
    ${SOURCES}
    ${SOURCES_ASM}
    )
