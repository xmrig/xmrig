cmake_minimum_required(VERSION 3.5)
project (ethash C)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Os")

set(HEADERS
    data_sizes.h
    endian.h
    ethash.h
    ethash_internal.h
    fnv.h
    )

set(SOURCES
    ethash_internal.c
    keccakf800.c
   )

include_directories(../..)

add_library(ethash STATIC
    ${HEADERS}
    ${SOURCES}
    )
