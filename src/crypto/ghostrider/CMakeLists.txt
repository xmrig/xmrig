cmake_minimum_required(VERSION 3.1)
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

if (CMAKE_C_COMPILER_ID MATCHES MSVC)
    set_source_files_properties(sph_blake.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_bmw.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_cubehash.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_echo.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_fugue.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_groestl.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_hamsi.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_jh.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_keccak.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_luffa.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_shabal.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_shavite.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_simd.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_sha2.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_skein.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
    set_source_files_properties(sph_whirlpool.c PROPERTIES COMPILE_FLAGS_RELEASE "/O1 /Oi /Os")
elseif (CMAKE_C_COMPILER_ID MATCHES GNU OR CMAKE_C_COMPILER_ID MATCHES Clang)
    set_source_files_properties(sph_blake.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_bmw.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_cubehash.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_echo.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_fugue.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_groestl.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_hamsi.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_jh.c PROPERTIES COMPILE_FLAGS "-Os -fno-tree-vrp")
    set_source_files_properties(sph_keccak.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_luffa.c PROPERTIES COMPILE_FLAGS "-Os -Wno-unused-const-variable")
    set_source_files_properties(sph_shabal.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_shavite.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_simd.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_sha2.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_skein.c PROPERTIES COMPILE_FLAGS "-Os")
    set_source_files_properties(sph_whirlpool.c PROPERTIES COMPILE_FLAGS "-Os")
endif()

include_directories(.)
include_directories(../..)
include_directories(${UV_INCLUDE_DIR})

add_library(ghostrider STATIC ${HEADERS} ${SOURCES})
