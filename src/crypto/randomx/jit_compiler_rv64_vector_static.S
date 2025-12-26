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

#ifndef __riscv_v
#error This file requires rv64gcv
#endif

.option pic

.global DECL(randomx_riscv64_vector_code_begin)

.global DECL(randomx_riscv64_vector_sshash_begin)
.global DECL(randomx_riscv64_vector_sshash_imul_rcp_literals)
.global DECL(randomx_riscv64_vector_sshash_dataset_init)
.global DECL(randomx_riscv64_vector_sshash_generated_instructions)
.global DECL(randomx_riscv64_vector_sshash_generated_instructions_end)
.global DECL(randomx_riscv64_vector_sshash_cache_prefetch)
.global DECL(randomx_riscv64_vector_sshash_xor)
.global DECL(randomx_riscv64_vector_sshash_end)

.global DECL(randomx_riscv64_vector_program_params)
.global DECL(randomx_riscv64_vector_program_imul_rcp_literals)
.global DECL(randomx_riscv64_vector_program_begin)
.global DECL(randomx_riscv64_vector_program_main_loop_instructions)
.global DECL(randomx_riscv64_vector_program_main_loop_instructions_end)
.global DECL(randomx_riscv64_vector_program_main_loop_mx_xor)
.global DECL(randomx_riscv64_vector_program_main_loop_spaddr_xor)

.global DECL(randomx_riscv64_vector_program_main_loop_light_mode_data)
.global DECL(randomx_riscv64_vector_program_main_loop_instructions_end_light_mode)
.global DECL(randomx_riscv64_vector_program_main_loop_mx_xor_light_mode)
.global DECL(randomx_riscv64_vector_program_scratchpad_prefetch)
.global DECL(randomx_riscv64_vector_program_scratchpad_prefetch_end)

.global DECL(randomx_riscv64_vector_program_end)

.global DECL(randomx_riscv64_vector_code_end)

.balign 8

DECL(randomx_riscv64_vector_code_begin):

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
	vsetivli zero, 4, e64, m1, ta, ma

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
#ifdef __riscv_zicbop
	prefetch.r (x5)
#else
	ld x5, (x5)
#endif

	// Prefetch element 1
	vslidedown.vi v18, v9, 1
	vmv.x.s x5, v18
#ifdef __riscv_zicbop
	prefetch.r (x5)
#else
	ld x5, (x5)
#endif

	// Prefetch element 2
	vslidedown.vi v18, v9, 2
	vmv.x.s x5, v18
#ifdef __riscv_zicbop
	prefetch.r (x5)
#else
	ld x5, (x5)
#endif

	// Prefetch element 3
	vslidedown.vi v18, v9, 3
	vmv.x.s x5, v18
#ifdef __riscv_zicbop
	prefetch.r (x5)
#else
	ld x5, (x5)
#endif

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

DECL(randomx_riscv64_vector_sshash_end):

/*
Reference: https://github.com/tevador/RandomX/blob/master/doc/specs.md#46-vm-execution

C declarations:

struct RegisterFile {
	uint64_t r[8];
	double f[4][2];
	double e[4][2];
	double a[4][2];
};

struct MemoryRegisters {
	uint32_t mx, ma;
	uint8_t* memory; // dataset (fast mode) or cache (light mode)
};

void ProgramFunc(RegisterFile* reg, MemoryRegisters* mem, uint8_t* scratchpad, uint64_t iterations);

Register layout
---------------
x0	= zero
x1	= scratchpad L3 mask
x2	= stack pointer
x3	= global pointer (unused)
x4	= thread pointer (unused)
x5	= temporary
x6	= temporary
x7	= branch mask (unshifted)
x8	= frame pointer, also 64-bit literal inside the loop
x9	= scratchpad L3 mask (64-byte aligned)
x10	= RegisterFile* reg, also 64-bit literal inside the loop
x11	= MemoryRegisters* mem, then dataset/cache pointer
x12	= scratchpad
x13	= iterations
x14	= mx, ma (always stored with dataset mask applied)
x15	= spAddr0, spAddr1
x16	= scratchpad L1 mask
x17	= scratchpad L2 mask
x18	= IMUL_RCP literals pointer
x19	= dataset mask
x20-x27	= r0-r7
x28-x31 = 64-bit literals

f0-f7   = 64-bit literals
f10-f17 = 64-bit literals
f28-f31 = 64-bit literals

v0-v3	= f0-f3
v4-v7	= e0-e3
v8-v11	= a0-a3
v12	= E 'and' mask = 0x00ffffffffffffff'00ffffffffffffff
v13	= E 'or' mask  = 0x3*00000000******'3*00000000******
v14	= scale mask   = 0x80f0000000000000'80f0000000000000

v15	= unused
v16	= temporary
v17	= unused
v18	= temporary

v19-v31	= unused
*/

.balign 8

DECL(randomx_riscv64_vector_program_params):

// JIT compiler will adjust these values for different RandomX variants
randomx_masks:	.dword 16376, 262136, 2097144, 2147483584, 255

DECL(randomx_riscv64_vector_program_imul_rcp_literals):

imul_rcp_literals:	.fill RANDOMX_PROGRAM_MAX_SIZE, 8, 0

DECL(randomx_riscv64_vector_program_begin):
	addi sp, sp, -112
	sd x8, 96(sp)		// save old frame pointer
	addi x8, sp, 112	// setup new frame pointer
	sd x1, 104(sp)		// save return address

	// Save callee-saved registers
	sd x9, 0(sp)
	sd x18, 8(sp)
	sd x19, 16(sp)
	sd x20, 24(sp)
	sd x21, 32(sp)
	sd x22, 40(sp)
	sd x23, 48(sp)
	sd x24, 56(sp)
	sd x25, 64(sp)
	sd x26, 72(sp)
	sd x27, 80(sp)

	// Save x10 as it will be used as an IMUL_RCP literal
	sd x10, 88(sp)

	// Load mx, ma and dataset pointer
	ld x14, (x11)
	ld x11, 8(x11)

	// Initialize spAddr0-spAddr1
	mv x15, x14

	// Set registers r0-r7 to zero
	li x20, 0
	li x21, 0
	li x22, 0
	li x23, 0
	li x24, 0
	li x25, 0
	li x26, 0
	li x27, 0

	// Load masks
	lla x5, randomx_masks
	ld x16, 0(x5)
	ld x17, 8(x5)
	ld x1, 16(x5)
	ld x19, 24(x5)
	ld x7, 32(x5)
	addi x9, x1, -56

	// Set vector registers to 2x64 bit
	vsetivli zero, 2, e64, m1, ta, ma

	// Apply dataset mask to mx, ma
	slli x5, x19, 32
	or x5, x5, x19
	and x14, x14, x5

	// Load group A registers
	addi x5, x10, 192
	vle64.v v8, (x5)

	addi x5, x10, 208
	vle64.v v9, (x5)

	addi x5, x10, 224
	vle64.v v10, (x5)

	addi x5, x10, 240
	vle64.v v11, (x5)

	// Load E 'and' mask
	vmv.v.i v12, -1
	vsrl.vi v12, v12, 8

	// Load E 'or' mask (stored in reg.f[0])
	addi x5, x10, 64
	vle64.v v13, (x5)

	// Load scale mask
	lui x5, 0x80f00
	slli x5, x5, 32
	vmv.v.x v14, x5

	// IMUL_RCP literals pointer
	lla x18, imul_rcp_literals

	// Load IMUL_RCP literals
	ld   x8,   0(x18)
	ld  x10,   8(x18)
	ld  x28,  16(x18)
	ld  x29,  24(x18)
	ld  x30,  32(x18)
	ld  x31,  40(x18)
	fld  f0,  48(x18)
	fld  f1,  56(x18)
	fld  f2,  64(x18)
	fld  f3,  72(x18)
	fld  f4,  80(x18)
	fld  f5,  88(x18)
	fld  f6,  96(x18)
	fld  f7, 104(x18)
	fld f10, 112(x18)
	fld f11, 120(x18)
	fld f12, 128(x18)
	fld f13, 136(x18)
	fld f14, 144(x18)
	fld f15, 152(x18)
	fld f16, 160(x18)
	fld f17, 168(x18)
	fld f28, 176(x18)
	fld f29, 184(x18)
	fld f30, 192(x18)
	fld f31, 200(x18)

randomx_riscv64_vector_program_main_loop:
	and x5, x15, x9		// x5 = spAddr0 & 64-byte aligned L3 mask
	add x5, x5, x12		// x5 = &scratchpad[spAddr0 & 64-byte aligned L3 mask]

	// read a 64-byte line from scratchpad (indexed by spAddr0) and XOR it with r0-r7
	ld x6, 0(x5)
	xor x20, x20, x6
	ld x6, 8(x5)
	xor x21, x21, x6
	ld x6, 16(x5)
	xor x22, x22, x6
	ld x6, 24(x5)
	xor x23, x23, x6
	ld x6, 32(x5)
	xor x24, x24, x6
	ld x6, 40(x5)
	xor x25, x25, x6
	ld x6, 48(x5)
	xor x26, x26, x6
	ld x6, 56(x5)
	xor x27, x27, x6

	srli x5, x15, 32	// x5 = spAddr1
	and x5, x5, x9		// x5 = spAddr1 & 64-byte aligned L3 mask
	add x5, x5, x12		// x5 = &scratchpad[spAddr1 & 64-byte aligned L3 mask]

	// read a 64-byte line from scratchpad (indexed by spAddr1) and initialize f0-f3, e0-e3 registers

	// Set vector registers to 2x32 bit
	vsetivli zero, 2, e32, m1, ta, ma

	// load f0
	vle32.v v16, (x5)
	vfwcvt.f.x.v v0, v16

	// load f1
	addi x6, x5, 8
	vle32.v v1, (x6)
	// Use v16 as an intermediary register because vfwcvt accepts only registers with even numbers here
	vfwcvt.f.x.v v16, v1
	vmv1r.v v1, v16

	// load f2
	addi x6, x5, 16
	vle32.v v16, (x6)
	vfwcvt.f.x.v v2, v16

	// load f3
	addi x6, x5, 24
	vle32.v v3, (x6)
	vfwcvt.f.x.v v16, v3
	vmv1r.v v3, v16

	// load e0
	addi x6, x5, 32
	vle32.v v16, (x6)
	vfwcvt.f.x.v v4, v16

	// load e1
	addi x6, x5, 40
	vle32.v v5, (x6)
	vfwcvt.f.x.v v16, v5
	vmv1r.v v5, v16

	// load e2
	addi x6, x5, 48
	vle32.v v16, (x6)
	vfwcvt.f.x.v v6, v16

	// load e3
	addi x6, x5, 56
	vle32.v v7, (x6)
	vfwcvt.f.x.v v16, v7
	vmv1r.v v7, v16

	// Set vector registers back to 2x64 bit
	vsetivli zero, 2, e64, m1, ta, ma

	// post-process e0-e3
	vand.vv v4, v4, v12
	vand.vv v5, v5, v12
	vand.vv v6, v6, v12
	vand.vv v7, v7, v12

	vor.vv v4, v4, v13
	vor.vv v5, v5, v13
	vor.vv v6, v6, v13
	vor.vv v7, v7, v13

DECL(randomx_riscv64_vector_program_main_loop_instructions):
	// Generated by JIT compiler
	// FDIV_M can generate up to 50 bytes of code (round it up to 52 - a multiple of 4)
	// +32 bytes for the scratchpad prefetch and the final jump instruction
	.fill RANDOMX_PROGRAM_MAX_SIZE * 52 + 32, 1, 0

DECL(randomx_riscv64_vector_program_main_loop_instructions_end):
	// Calculate dataset pointer for dataset read
	// Do it here to break false dependency from readReg2 and readReg3 (see below)
	srli x6, x14, 32	// x6 = ma & dataset mask

DECL(randomx_riscv64_vector_program_main_loop_mx_xor):
	xor x5, x24, x26	// x5 = readReg2 ^ readReg3 (JIT compiler will substitute the actual registers)

	and x5, x5, x19		// x5 = (readReg2 ^ readReg3) & dataset mask
	xor x14, x14, x5	// mx ^= (readReg2 ^ readReg3) & dataset mask

	and x5, x14, x19	// x5 = mx & dataset mask
	add x5, x5, x11		// x5 = &dataset[mx & dataset mask]

#ifdef __riscv_zicbop
	prefetch.r (x5)
#else
	ld x5, (x5)
#endif

	add x5, x6, x11		// x5 = &dataset[ma & dataset mask]

	// read a 64-byte line from dataset and XOR it with r0-r7
	ld x6, 0(x5)
	xor x20, x20, x6
	ld x6, 8(x5)
	xor x21, x21, x6
	ld x6, 16(x5)
	xor x22, x22, x6
	ld x6, 24(x5)
	xor x23, x23, x6
	ld x6, 32(x5)
	xor x24, x24, x6
	ld x6, 40(x5)
	xor x25, x25, x6
	ld x6, 48(x5)
	xor x26, x26, x6
	ld x6, 56(x5)
	xor x27, x27, x6

randomx_riscv64_vector_program_main_loop_swap_mx_ma:
	// swap mx <-> ma
#ifdef __riscv_zbb
	rori x14, x14, 32
#else
	srli x5, x14, 32
	slli x14, x14, 32
	or x14, x14, x5
#endif

	srli x5, x15, 32	// x5 = spAddr1
	and x5, x5, x9		// x5 = spAddr1 & 64-byte aligned L3 mask
	add x5, x5, x12		// x5 = &scratchpad[spAddr1 & 64-byte aligned L3 mask]

	// store registers r0-r7 to the scratchpad
	sd x20, 0(x5)
	sd x21, 8(x5)
	sd x22, 16(x5)
	sd x23, 24(x5)
	sd x24, 32(x5)
	sd x25, 40(x5)
	sd x26, 48(x5)
	sd x27, 56(x5)

	and x5, x15, x9		// x5 = spAddr0 & 64-byte aligned L3 mask
	add x5, x5, x12		// x5 = &scratchpad[spAddr0 & 64-byte aligned L3 mask]

DECL(randomx_riscv64_vector_program_main_loop_spaddr_xor):
	xor x15, x20, x22	// spAddr0-spAddr1 = readReg0 ^ readReg1 (JIT compiler will substitute the actual registers)

	// store registers f0-f3 to the scratchpad (f0-f3 are first combined with e0-e3)
	vxor.vv v0, v0, v4
	vxor.vv v1, v1, v5
	vxor.vv v2, v2, v6
	vxor.vv v3, v3, v7

	vse64.v v0, (x5)

	addi x6, x5, 16
	vse64.v v1, (x6)

	addi x6, x5, 32
	vse64.v v2, (x6)

	addi x6, x5, 48
	vse64.v v3, (x6)

	addi x13, x13, -1
	beqz x13, randomx_riscv64_vector_program_main_loop_end
	j randomx_riscv64_vector_program_main_loop

randomx_riscv64_vector_program_main_loop_end:
	// Restore x8 and x10
	addi x8, sp, 112
	ld x10, 88(sp)

	// Store integer registers
	sd x20, 0(x10)
	sd x21, 8(x10)
	sd x22, 16(x10)
	sd x23, 24(x10)
	sd x24, 32(x10)
	sd x25, 40(x10)
	sd x26, 48(x10)
	sd x27, 56(x10)

	// Store FP registers
	addi x5, x10, 64
	vse64.v v0, (x5)

	addi x5, x10, 80
	vse64.v v1, (x5)

	addi x5, x10, 96
	vse64.v v2, (x5)

	addi x5, x10, 112
	vse64.v v3, (x5)

	addi x5, x10, 128
	vse64.v v4, (x5)

	addi x5, x10, 144
	vse64.v v5, (x5)

	addi x5, x10, 160
	vse64.v v6, (x5)

	addi x5, x10, 176
	vse64.v v7, (x5)

	// Restore callee-saved registers
	ld x9, 0(sp)
	ld x18, 8(sp)
	ld x19, 16(sp)
	ld x20, 24(sp)
	ld x21, 32(sp)
	ld x22, 40(sp)
	ld x23, 48(sp)
	ld x24, 56(sp)
	ld x25, 64(sp)
	ld x26, 72(sp)
	ld x27, 80(sp)

	ld x8, 96(sp)	// old frame pointer
	ld x1, 104(sp)	// return address

	addi sp, sp, 112

	ret

DECL(randomx_riscv64_vector_program_main_loop_light_mode_data):
	// 1) Pointer to the scalar dataset init function
	// 2) Dataset offset
	.dword 0, 0

DECL(randomx_riscv64_vector_program_main_loop_instructions_end_light_mode):
	// Calculate dataset pointer for dataset read
	// Do it here to break false dependency from readReg2 and readReg3 (see below)
	srli x6, x14, 32	// x6 = ma & dataset mask

DECL(randomx_riscv64_vector_program_main_loop_mx_xor_light_mode):
	xor x5, x24, x26	// x5 = readReg2 ^ readReg3 (JIT compiler will substitute the actual registers)
	and x5, x5, x19		// x5 = (readReg2 ^ readReg3) & dataset mask
	xor x14, x14, x5	// mx ^= (readReg2 ^ readReg3) & dataset mask

	// Save all registers modified when calling dataset_init_scalar_func_ptr
	addi sp, sp, -192

	// bytes [0, 127] - saved registers
	// bytes [128, 191] - output buffer

	sd  x1,   0(sp)
	sd  x7,  16(sp)
	sd x10,  24(sp)
	sd x11,  32(sp)
	sd x12,  40(sp)
	sd x13,  48(sp)
	sd x14,  56(sp)
	sd x15,  64(sp)
	sd x16,  72(sp)
	sd x17,  80(sp)
	sd x28,  88(sp)
	sd x29,  96(sp)
	sd x30, 104(sp)
	sd x31, 112(sp)

	// setup randomx_riscv64_vector_sshash_dataset_init's parameters

	// x10 = pointer to pointer to cache memory
	// pointer to cache memory was saved in "sd x11, 32(sp)", so x10 = sp + 32
	addi x10, sp, 32

	// x11 = output buffer (64 bytes)
	addi x11, sp, 128

	// x12 = start block
	lla x5, randomx_riscv64_vector_program_main_loop_light_mode_data
	ld x12, 8(x5)
	add x12, x12, x6
	srli x12, x12, 6

	// x13 = end block
	addi x13, x12, 1

	ld x5, 0(x5)
	jalr x1, 0(x5)

	// restore registers
	ld  x1,   0(sp)
	ld  x7,  16(sp)
	ld x10,  24(sp)
	ld x11,  32(sp)
	ld x12,  40(sp)
	ld x13,  48(sp)
	ld x14,  56(sp)
	ld x15,  64(sp)
	ld x16,  72(sp)
	ld x17,  80(sp)
	ld x28,  88(sp)
	ld x29,  96(sp)
	ld x30, 104(sp)
	ld x31, 112(sp)

	// read a 64-byte line from dataset and XOR it with r0-r7
	ld x5, 128(sp)
	xor x20, x20, x5
	ld x5, 136(sp)
	xor x21, x21, x5
	ld x5, 144(sp)
	xor x22, x22, x5
	ld x5, 152(sp)
	xor x23, x23, x5
	ld x5, 160(sp)
	xor x24, x24, x5
	ld x5, 168(sp)
	xor x25, x25, x5
	ld x5, 176(sp)
	xor x26, x26, x5
	ld x5, 184(sp)
	xor x27, x27, x5

	addi sp, sp, 192

	j randomx_riscv64_vector_program_main_loop_swap_mx_ma

DECL(randomx_riscv64_vector_program_scratchpad_prefetch):
	xor x5, x20, x22	// spAddr0-spAddr1 = readReg0 ^ readReg1 (JIT compiler will substitute the actual registers)
	srli x6, x5, 32		// x6 = spAddr1

	and x5, x5, x9		// x5 = spAddr0 & 64-byte aligned L3 mask
	and x6, x6, x9		// x6 = spAddr1 & 64-byte aligned L3 mask

	c.add x5, x12		// x5 = &scratchpad[spAddr0 & 64-byte aligned L3 mask]
	c.add x6, x12		// x6 = &scratchpad[spAddr1 & 64-byte aligned L3 mask]

#ifdef __riscv_zicbop
	prefetch.r (x5)
	prefetch.r (x6)
#else
	ld x5, (x5)
	ld x6, (x6)
#endif

DECL(randomx_riscv64_vector_program_scratchpad_prefetch_end):

DECL(randomx_riscv64_vector_program_end):

DECL(randomx_riscv64_vector_code_end):
