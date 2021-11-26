cmake_minimum_required(VERSION 2.8.12)
project(GhostRider)

set(HEADERS
    sph_types.h
    sph_blake.h
    sph_bmw.h
    sph_cubehash.h
    sph_echo.h
    sph_fugue.h
    sph_groestl.h
    sph_hamsi.h
    sph_jh.h
    sph_keccak.h
    sph_luffa.h
    sph_sha2.h
    sph_shabal.h
    sph_shavite.h
    sph_simd.h
    sph_skein.h
    sph_whirlpool.h
    ghostrider.h
)

set(SOURCES
    sph_blake.c
    sph_bmw.c
    sph_cubehash.c
    sph_echo.c
    sph_fugue.c
    sph_groestl.c
    sph_hamsi.c
    sph_jh.c
    sph_keccak.c
    sph_luffa.c
    sph_shabal.c
    sph_shavite.c
    sph_simd.c
    sph_sha2.c
    sph_skein.c
    sph_whirlpool.c
    ghostrider.cpp
)

if (CMAKE_C_COMPILER_ID MATCHES GNU)
    # gcc 11.2.0 crashes with -ftree-vrp
    set_source_files_properties(sph_jh.c PROPERTIES COMPILE_FLAGS "-Ofast -fno-tree-vrp")

    # gcc 11.2.0 creates incorrect code with -O3
    set_source_files_properties(sph_sha2.c PROPERTIES COMPILE_FLAGS "-O2")

    set_source_files_properties(sph_luffa.c PROPERTIES COMPILE_FLAGS "-Ofast -Wno-unused-const-variable")
endif()

include_directories(.)
include_directories(../..)
include_directories(${UV_INCLUDE_DIR})

add_library(ghostrider STATIC ${HEADERS} ${SOURCES})
