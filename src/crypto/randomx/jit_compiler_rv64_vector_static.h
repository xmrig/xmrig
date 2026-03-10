/*
Copyright (c) 2018-2020, tevador    <tevador@gmail.com>
Copyright (c) 2019-2021, XMRig      <https://github.com/xmrig>, <support@xmrig.com>
Copyright (c) 2025, SChernykh       <https://github.com/SChernykh>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#if defined(__cplusplus)
#include <cstdint>
#else
#include <stdint.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

struct randomx_cache;

void randomx_riscv64_vector_sshash_begin();
void randomx_riscv64_vector_sshash_imul_rcp_literals();
void randomx_riscv64_vector_sshash_dataset_init(struct randomx_cache* cache, uint8_t* output_buf, uint32_t startBlock, uint32_t endBlock);
void randomx_riscv64_vector_sshash_cache_prefetch();
void randomx_riscv64_vector_sshash_generated_instructions();
void randomx_riscv64_vector_sshash_generated_instructions_end();
void randomx_riscv64_vector_sshash_cache_prefetch();
void randomx_riscv64_vector_sshash_xor();
void randomx_riscv64_vector_sshash_set_cache_index();
void randomx_riscv64_vector_sshash_end();

#if defined(__cplusplus)
}
#endif
