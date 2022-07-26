cmake_minimum_required(VERSION 3.9)
cmake_policy(SET CMP0069 NEW)

project(sha256)

#set(HEADERS
#    sha2.h
#)

#set(SOURCES
#    sha2.c
#)

if (CMAKE_C_COMPILER_ID MATCHES MSVC)
#    set_source_files_properties(sha2.c PROPERTIES COMPILE_FLAGS "/O1 /Oi /Os")
elseif (CMAKE_C_COMPILER_ID MATCHES GNU OR CMAKE_C_COMPILER_ID MATCHES Clang)
#    set_source_files_properties(sha2.c PROPERTIES COMPILE_FLAGS "-Os -s")
endif()

include_directories(.)
include_directories(../..)
include_directories(${UV_INCLUDE_DIR})

add_library(sha256 STATIC ${HEADERS} ${SOURCES})
