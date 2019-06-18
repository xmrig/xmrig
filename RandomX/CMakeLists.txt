# Copyright (c) 2019, The Monero Project
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
#    of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be
#    used to endorse or promote products derived from this software without specific
#    prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 2.8.7)

set (randomx_sources
src/aes_hash.cpp
src/argon2_ref.c
src/dataset.cpp
src/soft_aes.cpp
src/virtual_memory.cpp
src/vm_interpreted.cpp
src/allocator.cpp
src/assembly_generator_x86.cpp
src/instruction.cpp
src/randomx.cpp
src/superscalar.cpp
src/vm_compiled.cpp
src/vm_interpreted_light.cpp
src/argon2_core.c
src/blake2_generator.cpp
src/instructions_portable.cpp
src/reciprocal.c
src/virtual_machine.cpp
src/vm_compiled_light.cpp
src/blake2/blake2b.c)

if (NOT ARCH_ID)
  set(ARCH_ID ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (ARCH_ID STREQUAL "x86_64" OR ARCH_ID STREQUAL "x86-64" OR ARCH_ID STREQUAL "amd64")
  list(APPEND randomx_sources
    src/jit_compiler_x86_static.S
    src/jit_compiler_x86.cpp)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")
endif()

set(RANDOMX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE STRING "RandomX Include path")

add_library(randomx
  ${randomx_sources})
target_link_libraries(randomx
  PRIVATE
    ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET randomx PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET randomx PROPERTY CXX_STANDARD 11)

# cheat because cmake and ccache hate each other
set_property(SOURCE src/jit_compiler_x86_static.S PROPERTY LANGUAGE C)

add_definitions(/UUNICODE)