# Copyright (c) 2018-2019, tevador <tevador@gmail.com>
# Copyright (c) 2019, SChernykh    <https://github.com/SChernykh>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 	* Redistributions of source code must retain the above copyright
# 	  notice, this list of conditions and the following disclaimer.
# 	* Redistributions in binary form must reproduce the above copyright
# 	  notice, this list of conditions and the following disclaimer in the
# 	  documentation and/or other materials provided with the distribution.
# 	* Neither the name of the copyright holder nor the
# 	  names of its contributors may be used to endorse or promote products
# 	  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#if defined(__APPLE__)
#define DECL(x) _##x
#else
#define DECL(x) x
#endif

	.arch armv8-a
	.text
	.global DECL(randomx_program_aarch64)
	.global DECL(randomx_program_aarch64_main_loop)
	.global DECL(randomx_program_aarch64_vm_instructions)
	.global DECL(randomx_program_aarch64_imul_rcp_literals_end)
	.global DECL(randomx_program_aarch64_vm_instructions_end)
	.global DECL(randomx_program_aarch64_cacheline_align_mask1)
	.global DECL(randomx_program_aarch64_cacheline_align_mask2)
	.global DECL(randomx_program_aarch64_update_spMix1)
	.global DECL(randomx_program_aarch64_vm_instructions_end_light)
	.global DECL(randomx_program_aarch64_light_cacheline_align_mask)
	.global DECL(randomx_program_aarch64_light_dataset_offset)
	.global DECL(randomx_init_dataset_aarch64)
	.global DECL(randomx_init_dataset_aarch64_end)
	.global DECL(randomx_calc_dataset_item_aarch64)
	.global DECL(randomx_calc_dataset_item_aarch64_prefetch)
	.global DECL(randomx_calc_dataset_item_aarch64_mix)
	.global DECL(randomx_calc_dataset_item_aarch64_store_result)
	.global DECL(randomx_calc_dataset_item_aarch64_end)

# Register allocation

# x0  -> pointer to reg buffer and then literal for IMUL_RCP
# x1  -> pointer to mem buffer and then to dataset
# x2  -> pointer to scratchpad
# x3  -> loop counter
# x4  -> "r0"
# x5  -> "r1"
# x6  -> "r2"
# x7  -> "r3"
# x8  -> fpcr (reversed bits)
# x9  -> mx, ma
# x10 -> spMix1
# x11 -> literal for IMUL_RCP
# x12 -> "r4"
# x13 -> "r5"
# x14 -> "r6"
# x15 -> "r7"
# x16 -> spAddr0
# x17 -> spAddr1
# x18 -> unused (platform register, don't touch it)
# x19 -> temporary
# x20 -> temporary
# x21 -> literal for IMUL_RCP
# x22 -> literal for IMUL_RCP
# x23 -> literal for IMUL_RCP
# x24 -> literal for IMUL_RCP
# x25 -> literal for IMUL_RCP
# x26 -> literal for IMUL_RCP
# x27 -> literal for IMUL_RCP
# x28 -> literal for IMUL_RCP
# x29 -> literal for IMUL_RCP
# x30 -> literal for IMUL_RCP

# v0-v15 -> store 32-bit literals
# v16 -> "f0"
# v17 -> "f1"
# v18 -> "f2"
# v19 -> "f3"
# v20 -> "e0"
# v21 -> "e1"
# v22 -> "e2"
# v23 -> "e3"
# v24 -> "a0"
# v25 -> "a1"
# v26 -> "a2"
# v27 -> "a3"
# v28 -> temporary
# v29 -> E 'and' mask = 0x00ffffffffffffff'00ffffffffffffff
# v30 -> E 'or' mask  = 0x3*00000000******'3*00000000******
# v31 -> scale mask   = 0x80f0000000000000'80f0000000000000

	.balign 4
DECL(randomx_program_aarch64):
	# Save callee-saved registers
	sub	sp, sp, 192
	stp	x16, x17, [sp]
	str	x19, [sp, 16]
	stp	x20, x21, [sp, 32]
	stp	x22, x23, [sp, 48]
	stp	x24, x25, [sp, 64]
	stp	x26, x27, [sp, 80]
	stp	x28, x29, [sp, 96]
	stp	x8, x30, [sp, 112]
	stp	d8, d9, [sp, 128]
	stp	d10, d11, [sp, 144]
	stp	d12, d13, [sp, 160]
	stp	d14, d15, [sp, 176]

	# Zero integer registers
	mov	x4, xzr
	mov	x5, xzr
	mov	x6, xzr
	mov	x7, xzr
	mov	x12, xzr
	mov	x13, xzr
	mov	x14, xzr
	mov	x15, xzr

	# Load ma, mx and dataset pointer
	ldp	x9, x1, [x1]

	# Load initial spMix value
	mov	x10, x9

	# Load group A registers
	ldp	q24, q25, [x0, 192]
	ldp	q26, q27, [x0, 224]

	# Load E 'and' mask
	movi	v29.2d, #0x00FFFFFFFFFFFFFF

	# Load E 'or' mask (stored in reg.f[0])
	ldr	q30, [x0, 64]

	# Load scale mask
	mov	x16, 0x80f0000000000000
	dup	v31.2d, x16

	# Read fpcr
	mrs	x8, fpcr
	rbit	x8, x8

	# Save x0
	str	x0, [sp, -16]!

	# Read literals
	adr	x30, literal_v0
	ldp	q0, q1, [x30]
	ldp	q2, q3, [x30, 32]
	ldp	q4, q5, [x30, 64]
	ldp	q6, q7, [x30, 96]
	ldp	q8, q9, [x30, 128]
	ldp	q10, q11, [x30, 160]
	ldp	q12, q13, [x30, 192]
	ldp	q14, q15, [x30, 224]

	ldp	x0, x11, [x30, -96]	// literal_x0
	ldp	x21, x22, [x30, -80]	// literal_x21
	ldp	x23, x24, [x30, -64]	// literal_x23
	ldp	x25, x26, [x30, -48]	// literal_x25
	ldp	x27, x28, [x30, -32]	// literal_x27
	ldp	x29, x30, [x30, -16]	// literal_x29

DECL(randomx_program_aarch64_main_loop):
	# spAddr0 = spMix1 & ScratchpadL3Mask64;
	# spAddr1 = (spMix1 >> 32) & ScratchpadL3Mask64;
	lsr	x20, x10, 32

	# Actual mask will be inserted by JIT compiler
	and	w16, w10, 1
	and	w17, w20, 1

	# x16 = scratchpad + spAddr0
	# x17 = scratchpad + spAddr1
	add	x16, x16, x2
	add	x17, x17, x2

	# xor integer registers with scratchpad data (spAddr0)
	ldp	x20, x19, [x16]
	eor	x4, x4, x20
	eor	x5, x5, x19
	ldp	x20, x19, [x16, 16]
	eor	x6, x6, x20
	eor	x7, x7, x19
	ldp	x20, x19, [x16, 32]
	eor	x12, x12, x20
	eor	x13, x13, x19
	ldp	x20, x19, [x16, 48]
	eor	x14, x14, x20
	eor	x15, x15, x19

	# Load group F registers (spAddr1)
	ldr	q17, [x17]
	sxtl	v16.2d, v17.2s
	scvtf	v16.2d, v16.2d
	sxtl2	v17.2d, v17.4s
	scvtf	v17.2d, v17.2d

	ldr q19, [x17, 16]
	sxtl	v18.2d, v19.2s
	scvtf	v18.2d, v18.2d
	sxtl2	v19.2d, v19.4s
	scvtf	v19.2d, v19.2d

	# Load group E registers (spAddr1)
	ldr	q21, [x17, 32]
	sxtl	v20.2d, v21.2s
	scvtf	v20.2d, v20.2d
	sxtl2	v21.2d, v21.4s
	scvtf	v21.2d, v21.2d

	ldr	q23, [x17, 48]
	sxtl	v22.2d, v23.2s
	scvtf	v22.2d, v22.2d
	sxtl2	v23.2d, v23.4s
	scvtf	v23.2d, v23.2d

	and	v20.16b, v20.16b, v29.16b
	and	v21.16b, v21.16b, v29.16b
	and	v22.16b, v22.16b, v29.16b
	and	v23.16b, v23.16b, v29.16b
	orr	v20.16b, v20.16b, v30.16b
	orr	v21.16b, v21.16b, v30.16b
	orr	v22.16b, v22.16b, v30.16b
	orr	v23.16b, v23.16b, v30.16b

	# Execute VM instructions
DECL(randomx_program_aarch64_vm_instructions):

	# 16 KB buffer for generated instructions
	.fill 4096,4,0

literal_x0:  .fill 1,8,0
literal_x11: .fill 1,8,0
literal_x21: .fill 1,8,0
literal_x22: .fill 1,8,0
literal_x23: .fill 1,8,0
literal_x24: .fill 1,8,0
literal_x25: .fill 1,8,0
literal_x26: .fill 1,8,0
literal_x27: .fill 1,8,0
literal_x28: .fill 1,8,0
literal_x29: .fill 1,8,0
literal_x30: .fill 1,8,0
DECL(randomx_program_aarch64_imul_rcp_literals_end):

literal_v0:  .fill 2,8,0
literal_v1:  .fill 2,8,0
literal_v2:  .fill 2,8,0
literal_v3:  .fill 2,8,0
literal_v4:  .fill 2,8,0
literal_v5:  .fill 2,8,0
literal_v6:  .fill 2,8,0
literal_v7:  .fill 2,8,0
literal_v8:  .fill 2,8,0
literal_v9:  .fill 2,8,0
literal_v10: .fill 2,8,0
literal_v11: .fill 2,8,0
literal_v12: .fill 2,8,0
literal_v13: .fill 2,8,0
literal_v14: .fill 2,8,0
literal_v15: .fill 2,8,0

DECL(randomx_program_aarch64_vm_instructions_end):
	# Calculate dataset pointer for dataset read
	# Do it here to break false dependency from readReg2 and readReg3 (see next line)
	lsr	x10, x9, 32

	# mx ^= r[readReg2] ^ r[readReg3];
	eor	x9, x9, x20

	# Calculate dataset pointer for dataset prefetch
DECL(randomx_program_aarch64_cacheline_align_mask1):
	# Actual mask will be inserted by JIT compiler
	and	x20, x9, 1
	add	x20, x20, x1

	# Prefetch dataset data
	prfm	pldl2strm, [x20]

	# mx <-> ma
	ror	x9, x9, 32

DECL(randomx_program_aarch64_cacheline_align_mask2):
	# Actual mask will be inserted by JIT compiler
	and	x10, x10, 1
	add	x10, x10, x1

DECL(randomx_program_aarch64_xor_with_dataset_line):
	# xor integer registers with dataset data
	ldp	x20, x19, [x10]
	eor	x4, x4, x20
	eor	x5, x5, x19
	ldp	x20, x19, [x10, 16]
	eor	x6, x6, x20
	eor	x7, x7, x19
	ldp	x20, x19, [x10, 32]
	eor	x12, x12, x20
	eor	x13, x13, x19
	ldp	x20, x19, [x10, 48]
	eor	x14, x14, x20
	eor	x15, x15, x19

DECL(randomx_program_aarch64_update_spMix1):
	# JIT compiler will replace it with "eor x10, config.readReg0, config.readReg1"
	eor	x10, x0, x0

	# Store integer registers to scratchpad (spAddr1)
	stp	x4, x5, [x17, 0]
	stp	x6, x7, [x17, 16]
	stp	x12, x13, [x17, 32]
	stp	x14, x15, [x17, 48]

	# xor group F and group E registers
	eor	v16.16b, v16.16b, v20.16b
	eor	v17.16b, v17.16b, v21.16b
	eor	v18.16b, v18.16b, v22.16b
	eor	v19.16b, v19.16b, v23.16b

	# Store FP registers to scratchpad (spAddr0)
	stp	q16, q17, [x16, 0]
	stp	q18, q19, [x16, 32]

	subs	x3, x3, 1
	bne	DECL(randomx_program_aarch64_main_loop)
	
	# Restore x0
	ldr	x0, [sp], 16

	# Store integer registers
	stp	x4, x5, [x0, 0]
	stp	x6, x7, [x0, 16]
	stp	x12, x13, [x0, 32]
	stp	x14, x15, [x0, 48]

	# Store FP registers
	stp	q16, q17, [x0, 64]
	stp	q18, q19, [x0, 96]
	stp	q20, q21, [x0, 128]
	stp	q22, q23, [x0, 160]

	# Restore callee-saved registers
	ldp	x16, x17, [sp]
	ldr	x19, [sp, 16]
	ldp	x20, x21, [sp, 32]
	ldp	x22, x23, [sp, 48]
	ldp	x24, x25, [sp, 64]
	ldp	x26, x27, [sp, 80]
	ldp	x28, x29, [sp, 96]
	ldp	x8, x30, [sp, 112]
	ldp	d8, d9, [sp, 128]
	ldp	d10, d11, [sp, 144]
	ldp	d12, d13, [sp, 160]
	ldp	d14, d15, [sp, 176]
	add	sp, sp, 192

	ret

DECL(randomx_program_aarch64_vm_instructions_end_light):
	sub	sp, sp, 96
	stp	x0, x1, [sp, 64]
	stp	x2, x30, [sp, 80]

	# mx ^= r[readReg2] ^ r[readReg3];
	eor	x9, x9, x20

	# mx <-> ma
	ror	x9, x9, 32

	# x0 -> pointer to cache memory
	mov	x0, x1

	# x1 -> pointer to output
	mov	x1, sp

DECL(randomx_program_aarch64_light_cacheline_align_mask):
	# Actual mask will be inserted by JIT compiler
	and	w2, w9, 1

	# x2 -> item number
	lsr	x2, x2, 6

DECL(randomx_program_aarch64_light_dataset_offset):
	# Apply dataset offset (filled in by JIT compiler)
	add	x2, x2, 0
	add	x2, x2, 0

	bl	DECL(randomx_calc_dataset_item_aarch64)

	mov	x10, sp
	ldp	x0, x1, [sp, 64]
	ldp	x2, x30, [sp, 80]
	add	sp, sp, 96

	b	DECL(randomx_program_aarch64_xor_with_dataset_line)



# Input parameters
#
# x0 -> pointer to cache
# x1 -> pointer to dataset memory at startItem
# x2 -> start item
# x3 -> end item

DECL(randomx_init_dataset_aarch64):
	# Save x20 (used as temporary, but must be saved to not break ABI) and x30 (return address)
	stp	x20, x30, [sp, -16]!

	# Load pointer to cache memory
	ldr	x0, [x0]

DECL(randomx_init_dataset_aarch64_main_loop):
	bl	DECL(randomx_calc_dataset_item_aarch64)
	add	x1, x1, 64
	add	x2, x2, 1
	cmp	x2, x3
	bne	DECL(randomx_init_dataset_aarch64_main_loop)

	# Restore x20 and x30
	ldp	x20, x30, [sp], 16

	ret

DECL(randomx_init_dataset_aarch64_end):

# Input parameters
#
# x0 -> pointer to cache memory
# x1 -> pointer to output
# x2 -> item number
#
# Register allocation
#
# x0-x7 -> output value (calculated dataset item)
# x8 -> pointer to cache memory
# x9 -> pointer to output
# x10 -> registerValue
# x11 -> mixBlock
# x12 -> temporary
# x13 -> temporary

DECL(randomx_calc_dataset_item_aarch64):
	sub	sp, sp, 112
	stp	x0, x1, [sp]
	stp	x2, x3, [sp, 16]
	stp	x4, x5, [sp, 32]
	stp	x6, x7, [sp, 48]
	stp	x8, x9, [sp, 64]
	stp	x10, x11, [sp, 80]
	stp	x12, x13, [sp, 96]

	adr	x7, superscalarMul0
	# superscalarMul0, superscalarAdd1
	ldp	x12, x13, [x7]

	ldp	x8, x9, [sp]
	mov	x10, x2

	# rl[0] = (itemNumber + 1) * superscalarMul0;
	madd	x0, x2, x12, x12

	# rl[1] = rl[0] ^ superscalarAdd1;
	eor	x1, x0, x13

	# rl[2] = rl[0] ^ superscalarAdd2;
	ldp	x12, x13, [x7, 16]
	eor	x2, x0, x12

	# rl[3] = rl[0] ^ superscalarAdd3;
	eor	x3, x0, x13

	# rl[4] = rl[0] ^ superscalarAdd4;
	ldp	x12, x13, [x7, 32]
	eor	x4, x0, x12

	# rl[5] = rl[0] ^ superscalarAdd5;
	eor	x5, x0, x13

	# rl[6] = rl[0] ^ superscalarAdd6;
	ldp	x12, x13, [x7, 48]
	eor	x6, x0, x12

	# rl[7] = rl[0] ^ superscalarAdd7;
	eor	x7, x0, x13

	b	DECL(randomx_calc_dataset_item_aarch64_prefetch)

superscalarMul0: .quad 6364136223846793005
superscalarAdd1: .quad 9298411001130361340
superscalarAdd2: .quad 12065312585734608966
superscalarAdd3: .quad 9306329213124626780
superscalarAdd4: .quad 5281919268842080866
superscalarAdd5: .quad 10536153434571861004
superscalarAdd6: .quad 3398623926847679864
superscalarAdd7: .quad 9549104520008361294

# Prefetch -> SuperScalar hash -> Mix will be repeated N times

DECL(randomx_calc_dataset_item_aarch64_prefetch):
	# Actual mask will be inserted by JIT compiler
	and	x11, x10, 1
	add	x11, x8, x11, lsl 6
	prfm	pldl2strm, [x11]

	# Generated SuperScalar hash program goes here

DECL(randomx_calc_dataset_item_aarch64_mix):
	ldp	x12, x13, [x11]
	eor	x0, x0, x12
	eor	x1, x1, x13
	ldp	x12, x13, [x11, 16]
	eor	x2, x2, x12
	eor	x3, x3, x13
	ldp	x12, x13, [x11, 32]
	eor	x4, x4, x12
	eor	x5, x5, x13
	ldp	x12, x13, [x11, 48]
	eor	x6, x6, x12
	eor	x7, x7, x13

DECL(randomx_calc_dataset_item_aarch64_store_result):
	stp	x0, x1, [x9]
	stp	x2, x3, [x9, 16]
	stp	x4, x5, [x9, 32]
	stp	x6, x7, [x9, 48]

	ldp	x0, x1, [sp]
	ldp	x2, x3, [sp, 16]
	ldp	x4, x5, [sp, 32]
	ldp	x6, x7, [sp, 48]
	ldp	x8, x9, [sp, 64]
	ldp	x10, x11, [sp, 80]
	ldp	x12, x13, [sp, 96]
	add	sp, sp, 112

	ret

DECL(randomx_calc_dataset_item_aarch64_end):
