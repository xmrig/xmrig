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

#include "configuration.h"

// Compatibility macros

#if !defined(RANDOMX_CACHE_ACCESSES) && defined(RANDOMX_CACHE_MAX_ACCESSES)
#define RANDOMX_CACHE_ACCESSES RANDOMX_CACHE_MAX_ACCESSES
#endif

#if defined(RANDOMX_ARGON_MEMORY)
#define RANDOMX_CACHE_MASK RANDOMX_ARGON_MEMORY * 1024 / 64 - 1
#elif defined(RANDOMX_CACHE_MAX_SIZE)
#define RANDOMX_CACHE_MASK RANDOMX_CACHE_MAX_SIZE / 64 - 1
#endif

#define DECL(x) x

.text

.option arch, rv64gcv_zicbop
.option pic

.global DECL(randomx_riscv64_vector_sshash_begin)
.global DECL(randomx_riscv64_vector_sshash_imul_rcp_literals)
.global DECL(randomx_riscv64_vector_sshash_dataset_init)
.global DECL(randomx_riscv64_vector_sshash_generated_instructions)
.global DECL(randomx_riscv64_vector_sshash_generated_instructions_end)
.global DECL(randomx_riscv64_vector_sshash_cache_prefetch)
.global DECL(randomx_riscv64_vector_sshash_xor)
.global DECL(randomx_riscv64_vector_sshash_set_cache_index)
.global DECL(randomx_riscv64_vector_sshash_end)

.balign 8

DECL(randomx_riscv64_vector_sshash_begin):

sshash_constant_0: .dword 6364136223846793005
sshash_constant_1: .dword 9298411001130361340
sshash_constant_2: .dword 12065312585734608966
sshash_constant_3: .dword 9306329213124626780
sshash_constant_4: .dword 5281919268842080866
sshash_constant_5: .dword 10536153434571861004
sshash_constant_6: .dword 3398623926847679864
sshash_constant_7: .dword 9549104520008361294
sshash_offsets:    .dword 0,1,2,3
store_offsets:     .dword 0,64,128,192

DECL(randomx_riscv64_vector_sshash_imul_rcp_literals): .fill 512,8,0

/*
Reference: https://github.com/tevador/RandomX/blob/master/doc/specs.md#73-dataset-block-generation

Register layout
---------------
x5	= temporary

x10	= randomx cache
x11	= output buffer
x12	= startBlock
x13	= endBlock

x14	= cache mask
x15	= imul_rcp literal pointer

v0-v7	= r0-r7
v8	= itemNumber
v9	= cacheIndex, then a pointer into cache->memory (for prefetch), then a byte offset into cache->memory

v10-v17	= sshash constants

v18	= temporary

v19	= dataset item store offsets
*/

DECL(randomx_riscv64_vector_sshash_dataset_init):
	// Process 4 64-bit values at a time
	li x5, 4
	vsetvli x5, x5, e64, m1, ta, ma

	// Load cache->memory pointer
	ld x10, (x10)

	// Init cache mask
	li x14, RANDOMX_CACHE_MASK

	// Init dataset item store offsets
	lla x5, store_offsets
	vle64.v v19, (x5)

	// Init itemNumber vector to (startBlock, startBlock + 1, startBlock + 2, startBlock + 3)
	lla x5, sshash_offsets
	vle64.v v8, (x5)
	vadd.vx v8, v8, x12

	// Load constants (stride = x0 = 0, so a 64-bit value will be broadcast into each element of a vector)
	lla x5, sshash_constant_0
	vlse64.v v10, (x5), x0

	lla x5, sshash_constant_1
	vlse64.v v11, (x5), x0

	lla x5, sshash_constant_2
	vlse64.v v12, (x5), x0

	lla x5, sshash_constant_3
	vlse64.v v13, (x5), x0

	lla x5, sshash_constant_4
	vlse64.v v14, (x5), x0

	lla x5, sshash_constant_5
	vlse64.v v15, (x5), x0

	lla x5, sshash_constant_6
	vlse64.v v16, (x5), x0

	lla x5, sshash_constant_7
	vlse64.v v17, (x5), x0

	// Calculate the end pointer for dataset init
	sub x13, x13, x12
	slli x13, x13, 6
	add x13, x13, x11

init_item:
	// Step 1. Init r0-r7

	// r0 = (itemNumber + 1) * 6364136223846793005
	vmv.v.v v0, v8
	vmadd.vv v0, v10, v10

	// r_i = r0 ^ c_i for i = 1..7
	vxor.vv v1, v0, v11
	vxor.vv v2, v0, v12
	vxor.vv v3, v0, v13
	vxor.vv v4, v0, v14
	vxor.vv v5, v0, v15
	vxor.vv v6, v0, v16
	vxor.vv v7, v0, v17

	// Step 2. Let cacheIndex = itemNumber
	vmv.v.v v9, v8

	// Step 3 is implicit (all iterations are inlined, there is no "i")

	// Init imul_rcp literal pointer
	lla x15, randomx_riscv64_vector_sshash_imul_rcp_literals

DECL(randomx_riscv64_vector_sshash_generated_instructions):
	// Generated by JIT compiler
	//
	// Step 4. randomx_riscv64_vector_sshash_cache_prefetch
	// Step 5. SuperscalarHash[i]
	// Step 6. randomx_riscv64_vector_sshash_xor
	// Step 7. randomx_riscv64_vector_sshash_set_cache_index
	//
	// Above steps will be repeated RANDOMX_CACHE_ACCESSES times
	.fill RANDOMX_CACHE_ACCESSES * 2048, 4, 0

DECL(randomx_riscv64_vector_sshash_generated_instructions_end):
	// Step 9. Concatenate registers r0-r7 in little endian format to get the final Dataset item data.
	vsuxei64.v v0, (x11), v19

	add x5, x11, 8
	vsuxei64.v v1, (x5), v19

	add x5, x11, 16
	vsuxei64.v v2, (x5), v19

	add x5, x11, 24
	vsuxei64.v v3, (x5), v19

	add x5, x11, 32
	vsuxei64.v v4, (x5), v19

	add x5, x11, 40
	vsuxei64.v v5, (x5), v19

	add x5, x11, 48
	vsuxei64.v v6, (x5), v19

	add x5, x11, 56
	vsuxei64.v v7, (x5), v19

	// Iterate to the next 4 items
	vadd.vi v8, v8, 4
	add x11, x11, 256
	bltu x11, x13, init_item

	ret

// Step 4. Load a 64-byte item from the Cache. The item index is given by cacheIndex modulo the total number of 64-byte items in Cache.
DECL(randomx_riscv64_vector_sshash_cache_prefetch):
	// v9 = convert from cacheIndex to a direct pointer into cache->memory
	vand.vx v9, v9, x14
	vsll.vi v9, v9, 6
	vadd.vx v9, v9, x10

	// Prefetch element 0
	vmv.x.s x5, v9
	prefetch.r (x5)

	// Prefetch element 1
	vslidedown.vi v18, v9, 1
	vmv.x.s x5, v18
	prefetch.r (x5)

	// Prefetch element 2
	vslidedown.vi v18, v9, 2
	vmv.x.s x5, v18
	prefetch.r (x5)

	// Prefetch element 3
	vslidedown.vi v18, v9, 3
	vmv.x.s x5, v18
	prefetch.r (x5)

	// v9 = byte offset into cache->memory
	vsub.vx v9, v9, x10

// Step 6. XOR all registers with data loaded from randomx cache
DECL(randomx_riscv64_vector_sshash_xor):
	vluxei64.v v18, (x10), v9
	vxor.vv v0, v0, v18

	add x5, x10, 8
	vluxei64.v v18, (x5), v9
	vxor.vv v1, v1, v18

	add x5, x10, 16
	vluxei64.v v18, (x5), v9
	vxor.vv v2, v2, v18

	add x5, x10, 24
	vluxei64.v v18, (x5), v9
	vxor.vv v3, v3, v18

	add x5, x10, 32
	vluxei64.v v18, (x5), v9
	vxor.vv v4, v4, v18

	add x5, x10, 40
	vluxei64.v v18, (x5), v9
	vxor.vv v5, v5, v18

	add x5, x10, 48
	vluxei64.v v18, (x5), v9
	vxor.vv v6, v6, v18

	add x5, x10, 56
	vluxei64.v v18, (x5), v9
	vxor.vv v7, v7, v18

// Step 7. Set cacheIndex to the value of the register that has the longest dependency chain in the SuperscalarHash function executed in step 5.
DECL(randomx_riscv64_vector_sshash_set_cache_index):
	// JIT compiler will pick a single instruction reading from the required register
	vmv.v.v v9, v0
	vmv.v.v v9, v1
	vmv.v.v v9, v2
	vmv.v.v v9, v3
	vmv.v.v v9, v4
	vmv.v.v v9, v5
	vmv.v.v v9, v6
	vmv.v.v v9, v7

DECL(randomx_riscv64_vector_sshash_end):
