/*
  Copyright (c) 2024, RESBI <https://github.com/RESBI>

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
  ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT HOLDER BE LIABLE FOR ANY DAMAGES ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

# Translated from jit_compiler_a64_static.S.
# ASCII ONLY: never write a non-ASCII byte in this file.
#
# $r4  -> pointer to the register file (never clobbered)
# $r5  -> pointer to the mem buffer, then to the dataset
# $r6  -> pointer to the scratchpad
# $r7  -> loop counter
# $r8 - $r15  -> "r0" - "r7"
# $r16 -> spAddr0
# $r17 -> spAddr1
# $r18 -> shadow of $fcsr0 (used by the CFROUND instructions)
# $r19 -> mx, ma
# $r20 -> spMix1
# $r21 -> temporary
# $r22 -> temporary, also the handoff register: the last instruction of the
#          generated program writes r[readReg2] ^ r[readReg3] here (A64: x20)
# $r23 -> temporary used by this file only
#
# $vr0-$vr3   -> "f0"-"f3"
# $vr4-$vr7   -> "e0"-"e3"
# $vr8-$vr11  -> "a0"-"a3" (loaded once, never clobbered)
# $vr12       -> temporary used by the generated instructions
# $vr13       -> E "and" mask 0x00ffffffffffffff
# $vr14       -> E "or" mask (stored in reg.f[0])
# $vr15       -> scale mask 0x80f0000000000000
# $vr16-$vr23 -> free for the fixed code of this file
# $vr24-$vr31 -> callee saved, never touched
#
# Every overwritable slot has a fixed length. The JIT compiler writes these
# slots in place and must not change their size.

	.text

	.global	randomx_program_la64
	.global	randomx_program_la64_main_loop
	.global	randomx_program_la64_main_loop_mask0
	.global	randomx_program_la64_main_loop_mask1
	.global	randomx_program_la64_vm_instructions
	.global	randomx_program_la64_vm_instructions_end
	.global	randomx_program_la64_cacheline_align_mask1
	.global	randomx_program_la64_cacheline_align_mask2
	.global	randomx_program_la64_xor_with_dataset_line
	.global	randomx_program_la64_update_spMix1
	.global	randomx_program_la64_v2_FE_mix
	.global	randomx_program_la64_v1_FE_mix
	.global	randomx_program_la64_v2_FE_mix_soft_aes
	.global	randomx_program_la64_aes_lut_pointers
	.global	randomx_program_la64_FE_store
	.global	randomx_program_la64_vm_instructions_end_light
	.global	randomx_program_la64_vm_instructions_end_light_tweak
	.global	randomx_program_la64_light_cacheline_align_mask
	.global	randomx_program_la64_light_dataset_offset
	.global	randomx_program_la64_vm_instructions_end_v1
	.global	randomx_program_la64_vm_instructions_end_v2
	.global	randomx_program_la64_vm_instructions_end_light_v1
	.global	randomx_program_la64_vm_instructions_end_light_v2
	.global	randomx_init_dataset_la64
	.global	randomx_init_dataset_la64_end
	.global	randomx_calc_dataset_item_la64
	.global	randomx_calc_dataset_item_la64_prefetch
	.global	randomx_calc_dataset_item_la64_mix
	.global	randomx_calc_dataset_item_la64_store_result
	.global	randomx_calc_dataset_item_la64_end

	# The two 64 bit masks the prologue reads with la.local are
	# declared BELOW, inside the copied block, right after the
	# AES LUT pointers: the range this file is copied in
	# [randomx_program_la64, randomx_init_dataset_la64_end) and
	# la.local is PC relative, so a table declared before
	# randomx_program_la64 would make the copy read
	# before the copy. The two quads below are
	# the only data that has to stay inside.

	.balign 4
randomx_program_la64:
	# Save the callee saved registers changed by this function.
	# $r22 is $fp and $r23 is $s0.  The soft AES
	# routines are called with "bl" below, and that
	# clobbers $r1, so $r1 is saved here as well.
	addi.d	$r3, $r3, -32
	st.d	$r1, $r3, 0
	st.d	$r22, $r3, 8
	st.d	$r23, $r3, 16

	# Zero integer registers
	move	$r8, $r0
	move	$r9, $r0
	move	$r10, $r0
	move	$r11, $r0
	move	$r12, $r0
	move	$r13, $r0
	move	$r14, $r0
	move	$r15, $r0

	# Load ma, mx and the dataset pointer
	ld.d	$r19, $r5, 0
	ld.d	$r5, $r5, 8

	# Load the initial spMix value
	move	$r20, $r19

	# Load group A registers
	vld	$vr8, $r4, 192
	vld	$vr9, $r4, 208
	vld	$vr10, $r4, 224
	vld	$vr11, $r4, 240

	# Load the E "and" mask
	la.local	$r21, la64_E_and_mask
	ld.d	$r21, $r21, 0
	vreplgr2vr.d	$vr13, $r21

	# Load the E "or" mask (stored in reg.f[0])
	vld	$vr14, $r4, 64

	# Load the scale mask
	la.local	$r21, la64_scale_mask
	ld.d	$r21, $r21, 0
	vreplgr2vr.d	$vr15, $r21

	# Read $fcsr0
	movfcsr2gr	$r18, $fcsr0

randomx_program_la64_main_loop:
	# spAddr0 = spMix1 & ScratchpadL3Mask64;
	# spAddr1 = (spMix1 >> 32) & ScratchpadL3Mask64;
	srli.d	$r21, $r20, 32

	# The two mask slots below are 12 bytes (3 words) each, they are
	# overwritten in place by the JIT compiler in this fixed shape:
	#   lu12i.w $r23, <hi> / ori $r23, $r23, <lo> / and $r16, $r20, $r23
	# $r23 is only ever the mask register here.
randomx_program_la64_main_loop_mask0:
	lu12i.w	$r23, 0
	ori	$r23, $r23, 0
	and	$r16, $r20, $r23

randomx_program_la64_main_loop_mask1:
	lu12i.w	$r23, 0
	ori	$r23, $r23, 0
	and	$r17, $r21, $r23

	# $r16 = scratchpad + spAddr0
	# $r17 = scratchpad + spAddr1
	add.d	$r16, $r16, $r6
	add.d	$r17, $r17, $r6

	# xor integer registers with the scratchpad data (spAddr0)
	ld.d	$r21, $r16, 0
	ld.d	$r22, $r16, 8
	xor	$r8, $r8, $r21
	xor	$r9, $r9, $r22
	ld.d	$r21, $r16, 16
	ld.d	$r22, $r16, 24
	xor	$r10, $r10, $r21
	xor	$r11, $r11, $r22
	ld.d	$r21, $r16, 32
	ld.d	$r22, $r16, 40
	xor	$r12, $r12, $r21
	xor	$r13, $r13, $r22
	ld.d	$r21, $r16, 48
	ld.d	$r22, $r16, 56
	xor	$r14, $r14, $r21
	xor	$r15, $r15, $r22

	# Load group F registers (spAddr1)
	vld	$vr0, $r17, 0
	vffintl.d.w	$vr0, $vr0
	vld	$vr1, $r17, 8
	vffintl.d.w	$vr1, $vr1
	vld	$vr2, $r17, 16
	vffintl.d.w	$vr2, $vr2
	vld	$vr3, $r17, 24
	vffintl.d.w	$vr3, $vr3

	# Load group E registers (spAddr1)
	vld	$vr4, $r17, 32
	vffintl.d.w	$vr4, $vr4
	vld	$vr5, $r17, 40
	vffintl.d.w	$vr5, $vr5
	vld	$vr6, $r17, 48
	vffintl.d.w	$vr6, $vr6
	vld	$vr7, $r17, 56
	vffintl.d.w	$vr7, $vr7

	# Apply the E masks
	vand.v	$vr4, $vr4, $vr13
	vand.v	$vr5, $vr5, $vr13
	vand.v	$vr6, $vr6, $vr13
	vand.v	$vr7, $vr7, $vr13
	vor.v	$vr4, $vr4, $vr14
	vor.v	$vr5, $vr5, $vr14
	vor.v	$vr6, $vr6, $vr14
	vor.v	$vr7, $vr7, $vr14

	# The JIT compiler writes the program here.
	# Unlike A64 no literal pool is needed here, so the whole
	# buffer belongs to the generated program: 32 KB = 8192 words.
randomx_program_la64_vm_instructions:
	.fill	8192,4,0

randomx_program_la64_vm_instructions_end:
	# Calculate the dataset pointer for the dataset read.
	# Do it here to break the false dependency on readReg2/readReg3.
	srli.d	$r21, $r19, 32

	# mx ^= r[readReg2] ^ r[readReg3];
	xor	$r19, $r19, $r22

	# Calculate the dataset pointer for the dataset prefetch.
	bstrpick.d	$r22, $r19, 31, 0

	# mx <-> ma
	# 16 bytes in total, the JIT compiler copies the v1 or the v2
	# template of exactly this length over the block above.
	rotri.d	$r19, $r19, 32

randomx_program_la64_vm_instructions_end_v1:
	srli.d	$r21, $r19, 32
	xor	$r19, $r19, $r22
	bstrpick.d	$r22, $r19, 31, 0
	rotri.d	$r19, $r19, 32

randomx_program_la64_vm_instructions_end_v2:
	srli.d	$r21, $r19, 32
	rotri.d	$r19, $r19, 32
	xor	$r19, $r19, $r22
	bstrpick.d	$r22, $r19, 31, 0

	# 12 bytes (3 words) each, overwritten in place by the
	# JIT compiler with the CacheLineAlignMask of the dataset.
randomx_program_la64_cacheline_align_mask1:
	lu12i.w	$r23, 0
	ori	$r23, $r23, 0
	and	$r22, $r22, $r23
	add.d	$r22, $r22, $r5

	# Prefetch the dataset cache line
	preld	0, $r22, 0

randomx_program_la64_cacheline_align_mask2:
	lu12i.w	$r23, 0
	ori	$r23, $r23, 0
	and	$r21, $r21, $r23
	add.d	$r21, $r21, $r5

	# xor integer registers with the dataset data (base = $r21)
randomx_program_la64_xor_with_dataset_line:
	ld.d	$r22, $r21, 0
	ld.d	$r23, $r21, 8
	xor	$r8, $r8, $r22
	xor	$r9, $r9, $r23
	ld.d	$r22, $r21, 16
	ld.d	$r23, $r21, 24
	xor	$r10, $r10, $r22
	xor	$r11, $r11, $r23
	ld.d	$r22, $r21, 32
	ld.d	$r23, $r21, 40
	xor	$r12, $r12, $r22
	xor	$r13, $r13, $r23
	ld.d	$r22, $r21, 48
	ld.d	$r23, $r21, 56
	xor	$r14, $r14, $r22
	xor	$r15, $r15, $r23

	# spMix1 = r[readReg0] ^ r[readReg1];
	# 4 bytes, overwritten in place by the JIT compiler.
randomx_program_la64_update_spMix1:
	xor	$r20, $r4, $r4

	# Store the integer registers to the scratchpad (spAddr1)
	st.d	$r8, $r17, 0
	st.d	$r9, $r17, 8
	st.d	$r10, $r17, 16
	st.d	$r11, $r17, 24
	st.d	$r12, $r17, 32
	st.d	$r13, $r17, 40
	st.d	$r14, $r17, 48
	st.d	$r15, $r17, 56

	# RandomX v2 AES tweak: mix group F and group E
	# 4 bytes. LA64 has no hardware AES, so the safe default is the
	# soft AES routine and the JIT compiler only needs to write
	# "b randomx_program_la64_v1_FE_mix" when it runs RandomX v1.
randomx_program_la64_v2_FE_mix:
	b	randomx_program_la64_v2_FE_mix_soft_aes

randomx_program_la64_v1_FE_mix:
	vxor.v	$vr0, $vr0, $vr4
	vxor.v	$vr1, $vr1, $vr5
	vxor.v	$vr2, $vr2, $vr6
	vxor.v	$vr3, $vr3, $vr7

randomx_program_la64_FE_store:
	# Store the FP registers to the scratchpad (spAddr0)
	vst	$vr0, $r16, 0
	vst	$vr1, $r16, 16
	vst	$vr2, $r16, 32
	vst	$vr3, $r16, 48

	# Run the main loop again while the iteration counter is not zero.
	addi.d	$r7, $r7, -1
	bne	$r7, $r0, randomx_program_la64_main_loop

	# Store the integer registers to the reg buffer
	st.d	$r8, $r4, 0
	st.d	$r9, $r4, 8
	st.d	$r10, $r4, 16
	st.d	$r11, $r4, 24
	st.d	$r12, $r4, 32
	st.d	$r13, $r4, 40
	st.d	$r14, $r4, 48
	st.d	$r15, $r4, 56

	# Store the FP registers to the reg buffer,
	# the layout is fixed: f0-f3 at 64, e0-e3 at 128, a0-a3 at 192
	vst	$vr0, $r4, 64
	vst	$vr1, $r4, 80
	vst	$vr2, $r4, 96
	vst	$vr3, $r4, 112
	vst	$vr4, $r4, 128
	vst	$vr5, $r4, 144
	vst	$vr6, $r4, 160
	vst	$vr7, $r4, 176
	vst	$vr8, $r4, 192
	vst	$vr9, $r4, 208
	vst	$vr10, $r4, 224
	vst	$vr11, $r4, 240

	# Restore the callee saved registers and return.
	# $r1 has to come back, the soft AES
	# routines above left their own "bl"
	# return address in it.
	ld.d	$r1, $r3, 0
	ld.d	$r22, $r3, 8
	ld.d	$r23, $r3, 16
	addi.d	$r3, $r3, 32
	ret

# The light program is reached through a branch written by the JIT
# compiler at the end of the generated program.
randomx_program_la64_vm_instructions_end_light:
	addi.d	$r3, $r3, -96
	st.d	$r4, $r3, 64
	st.d	$r5, $r3, 72
	st.d	$r6, $r3, 80
	st.d	$r1, $r3, 88

	# $r6 = ma
	srli.d	$r6, $r19, 32

	# 12 bytes (3 words), overwritten in place by the JIT compiler
randomx_program_la64_light_cacheline_align_mask:
	lu12i.w	$r21, 0
	ori	$r21, $r21, 0
	and	$r6, $r6, $r21

	# RandomX v1 AES tweak: mx ^= r[readReg2] ^ r[readReg3]; mx <-> ma
	# 8 bytes, the JIT compiler copies the v1 or the v2 template.
randomx_program_la64_vm_instructions_end_light_tweak:
	xor	$r19, $r19, $r22
	rotri.d	$r19, $r19, 32

	# $r4 -> pointer to the cache memory, $r5 -> pointer to the
	# output, $r6 -> item number
	move	$r4, $r5
	move	$r5, $r3
	srli.d	$r6, $r6, 6

	# 12 bytes (3 words), the dataset offsets are written here
	# by the JIT compiler in exactly this shape:
	#   lu12i.w $r21, <hi20> / ori $r21, $r21, <lo12> / add.d $r6, $r6, $r21
randomx_program_la64_light_dataset_offset:
	lu12i.w	$r21, 0
	ori	$r21, $r21, 0
	add.d	$r6, $r6, $r21

	bl	randomx_calc_dataset_item_la64

	# $r21 -> the freshly calculated dataset item
	move	$r21, $r3
	ld.d	$r4, $r3, 64
	ld.d	$r5, $r3, 72
	ld.d	$r6, $r3, 80
	ld.d	$r1, $r3, 88
	addi.d	$r3, $r3, 96

	b	randomx_program_la64_xor_with_dataset_line

# 8 bytes (2 words) each, copied over the tweak above.
randomx_program_la64_vm_instructions_end_light_v1:
	xor	$r19, $r19, $r22
	rotri.d	$r19, $r19, 32

randomx_program_la64_vm_instructions_end_light_v2:
	rotri.d	$r19, $r19, 32
	xor	$r19, $r19, $r22

	# RandomX v2 AES tweak with soft AES.
	# Interface:
	#   randomx_soft_aesenc_la64 / randomx_soft_aesdec_la64
	#     in:  $vr16 = state, $vr17 = round key, $r21 = LUT base
	#     out: $vr16 = result
	#     clobbers: $r22, $r23, $vr16, $vr17
randomx_program_la64_v2_FE_mix_soft_aes:
	addi.d	$r3, $r3, -32
	st.d	$r1, $r3, 0

	# Read the two AES LUT pointers written by the JIT compiler
	la.local	$r21, randomx_program_la64_aes_lut_pointers
	ld.d	$r22, $r21, 0
	st.d	$r22, $r3, 8
	ld.d	$r23, $r21, 8
	st.d	$r23, $r3, 16

	# f0 = aesenc(f0, e0..e3)
	vor.v	$vr16, $vr0, $vr0
	vor.v	$vr17, $vr4, $vr4
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr17, $vr5, $vr5
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr17, $vr6, $vr6
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr17, $vr7, $vr7
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr0, $vr16, $vr16

	# f1 = aesdec(f1, e0..e3)
	vor.v	$vr16, $vr1, $vr1
	vor.v	$vr17, $vr4, $vr4
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr17, $vr5, $vr5
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr17, $vr6, $vr6
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr17, $vr7, $vr7
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr1, $vr16, $vr16

	# f2 = aesenc(f2, e0..e3)
	vor.v	$vr16, $vr2, $vr2
	vor.v	$vr17, $vr4, $vr4
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr17, $vr5, $vr5
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr17, $vr6, $vr6
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr17, $vr7, $vr7
	ld.d	$r21, $r3, 8
	bl	randomx_soft_aesenc_la64
	vor.v	$vr2, $vr16, $vr16

	# f3 = aesdec(f3, e0..e3)
	vor.v	$vr16, $vr3, $vr3
	vor.v	$vr17, $vr4, $vr4
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr17, $vr5, $vr5
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr17, $vr6, $vr6
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr17, $vr7, $vr7
	ld.d	$r21, $r3, 16
	bl	randomx_soft_aesdec_la64
	vor.v	$vr3, $vr16, $vr16

	ld.d	$r1, $r3, 0
	addi.d	$r3, $r3, 32

	b	randomx_program_la64_FE_store

	# The JIT compiler writes the two addresses here:
	# +0 = &lutEnc[0][0], +8 = &lutDec[0][0]. 16 bytes in total.
randomx_program_la64_aes_lut_pointers:
	.fill	2,8,0

	# The two 64 bit masks the prologue reads with la.local.
	# A64 gets them for free: "movi v29.2d, #0x00FFFFFFFFFFFFFF" and
	# "mov x16, 0x80f0000000000000 / dup v31.2d, x16".
	# LSX has no vector immediate move, so the two values
	# live here, inside the copied block and outside
	# every executed path.
	.balign 8
la64_E_and_mask:
	.quad	0x00ffffffffffffff

la64_scale_mask:
	.quad	0x80f0000000000000

	# Software AES encryption step.
	# in:  $vr16 = state, $vr17 = round key, $r21 = &lutEnc[0][0]
	# out: $vr16 = result. $r22 and $r23 are clobbered.
randomx_soft_aesenc_la64:
	addi.d	$r3, $r3, -48
	st.d	$r1, $r3, 0
	vst	$vr16, $r3, 16

	# result word 0: lut[0][b0] ^ lut[1][b5] ^ lut[2][b10] ^ lut[3][b15]
	ld.bu	$r22, $r3, 16
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 21
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 26
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 31
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 32

	# result word 1: lut[0][b4] ^ lut[1][b9] ^ lut[2][b14] ^ lut[3][b3]
	ld.bu	$r22, $r3, 20
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 25
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 30
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 19
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 36

	# result word 2: lut[0][b8] ^ lut[1][b13] ^ lut[2][b2] ^ lut[3][b7]
	ld.bu	$r22, $r3, 24
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 29
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 18
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 23
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 40

	# result word 3: lut[0][b12] ^ lut[1][b1] ^ lut[2][b6] ^ lut[3][b11]
	ld.bu	$r22, $r3, 28
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 17
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 22
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 27
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 44

	vld	$vr16, $r3, 32
	vxor.v	$vr16, $vr16, $vr17

	ld.d	$r1, $r3, 0
	addi.d	$r3, $r3, 48
	ret

	# Software AES decryption step.
	# in:  $vr16 = state, $vr17 = round key, $r21 = &lutDec[0][0]
	# out: $vr16 = result. $r22 and $r23 are clobbered.
randomx_soft_aesdec_la64:
	addi.d	$r3, $r3, -48
	st.d	$r1, $r3, 0
	vst	$vr16, $r3, 16

	# result word 0: lut[0][b0] ^ lut[1][b13] ^ lut[2][b10] ^ lut[3][b7]
	ld.bu	$r22, $r3, 16
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 29
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 26
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 23
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 32

	# result word 1: lut[0][b4] ^ lut[1][b1] ^ lut[2][b14] ^ lut[3][b11]
	ld.bu	$r22, $r3, 20
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 17
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 30
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 27
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 36

	# result word 2: lut[0][b8] ^ lut[1][b5] ^ lut[2][b2] ^ lut[3][b15]
	ld.bu	$r22, $r3, 24
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 21
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 18
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 31
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 40

	# result word 3: lut[0][b12] ^ lut[1][b9] ^ lut[2][b6] ^ lut[3][b3]
	ld.bu	$r22, $r3, 28
	slli.d	$r22, $r22, 2
	ldx.w	$r23, $r22, $r21
	ld.bu	$r22, $r3, 25
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 22
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	ld.bu	$r22, $r3, 19
	slli.d	$r22, $r22, 2
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	addi.d	$r22, $r22, 1024
	ldx.w	$r22, $r22, $r21
	xor	$r23, $r23, $r22
	st.w	$r23, $r3, 44

	vld	$vr16, $r3, 32
	vxor.v	$vr16, $vr16, $vr17

	ld.d	$r1, $r3, 0
	addi.d	$r3, $r3, 48
	ret

# Input parameters
#
# $r4 -> pointer to the cache pointer
# $r5 -> pointer to the dataset memory at startItem
# $r6 -> start item
# $r7 -> end item
#
# randomx_calc_dataset_item_la64 saves $r1 and $r4-$r15, so these four
# arguments survive every call.
randomx_init_dataset_la64:
	# The "bl" below writes the return address into $r1, so the
	# return address of this function has to be kept on the stack.
	addi.d	$r3, $r3, -16
	st.d	$r1, $r3, 0

	# Load the pointer to the cache memory
	ld.d	$r4, $r4, 0

randomx_init_dataset_la64_main_loop:
	bl	randomx_calc_dataset_item_la64
	addi.d	$r5, $r5, 64
	addi.d	$r6, $r6, 1
	bne	$r6, $r7, randomx_init_dataset_la64_main_loop

	ld.d	$r1, $r3, 0
	addi.d	$r3, $r3, 16

	ret

randomx_init_dataset_la64_end:

# Input parameters
#
# $r4 -> pointer to the cache memory
# $r5 -> pointer to the output
# $r6 -> item number
#
# Register allocation
#
# $r4-$r11  -> output value (the calculated dataset item)
# $r12      -> pointer to the cache memory
# $r13      -> pointer to the output
# $r14      -> registerValue, refreshed by the JIT compiler after every mix
# $r15      -> mix block pointer
# $r21-$r23 -> temporaries
#
# The whole register file is saved, otherwise the caller could not keep
# $r4-$r7 alive across the call.
#
# $r21-$r23 are temporaries of the AArch64 original (x12, x13 and the
# table pointer) and have to be saved as well: $r22 is $fp and $r23 is
# $s0, so both are callee saved on LoongArch, and the translated
# program keeps its own values in them. The AArch64 original saved x20
# inside its init loop for the same reason.
randomx_calc_dataset_item_la64:
	addi.d	$r3, $r3, -136
	st.d	$r1, $r3, 0
	st.d	$r4, $r3, 8
	st.d	$r5, $r3, 16
	st.d	$r6, $r3, 24
	st.d	$r7, $r3, 32
	st.d	$r8, $r3, 40
	st.d	$r9, $r3, 48
	st.d	$r10, $r3, 56
	st.d	$r11, $r3, 64
	st.d	$r12, $r3, 72
	st.d	$r13, $r3, 80
	st.d	$r14, $r3, 88
	st.d	$r15, $r3, 96
	st.d	$r21, $r3, 104
	st.d	$r22, $r3, 112
	st.d	$r23, $r3, 120

	la.local	$r22, la64_superscalar_mul0
	ld.d	$r21, $r22, 0

	# rl[0] = (itemNumber + 1) * superscalarMul0
	ld.d	$r14, $r3, 24
	addi.d	$r23, $r14, 1
	mul.d	$r4, $r23, $r21

	ld.d	$r12, $r3, 8
	ld.d	$r13, $r3, 16

	# rl[1] - rl[7] = rl[0] ^ superscalarAdd1 - superscalarAdd7
	ld.d	$r23, $r22, 8
	xor	$r5, $r4, $r23
	ld.d	$r23, $r22, 16
	xor	$r6, $r4, $r23
	ld.d	$r23, $r22, 24
	xor	$r7, $r4, $r23
	ld.d	$r23, $r22, 32
	xor	$r8, $r4, $r23
	ld.d	$r23, $r22, 40
	xor	$r9, $r4, $r23
	ld.d	$r23, $r22, 48
	xor	$r10, $r4, $r23
	ld.d	$r23, $r22, 56
	xor	$r11, $r4, $r23

	b	randomx_calc_dataset_item_la64_prefetch

	# The A64 table (jit_compiler_a64_static.S lines 830-837) is placed
	# inside the copied block on purpose: "la.local" is PC relative, so the
	# copy keeps pointing at its own table copy.
	.balign 8
la64_superscalar_mul0:
	.quad	6364136223846793005
la64_superscalar_add1:
	.quad	9298411001130361340
la64_superscalar_add2:
	.quad	12065312585734608966
la64_superscalar_add3:
	.quad	9306329213124626780
la64_superscalar_add4:
	.quad	5281919268842080866
la64_superscalar_add5:
	.quad	10536153434571861004
la64_superscalar_add6:
	.quad	3398623926847679864
la64_superscalar_add7:
	.quad	9549104520008361294

# Prefetch -> superscalar hash -> mix, repeated CacheAccesses times.
#
# The JIT compiler builds the whole body like this:
#   1. copy [randomx_calc_dataset_item_la64, _prefetch)
#   2. overwrite the FIRST word below with the cache line mask:
#        bstrpick.d $r15, $r14, Log2_CacheSize - 1, 0
#   3. copy [_prefetch + 4, _mix), that is the three words below the slot
#   4. append the generated superscalar program, which must leave the new
#      registerValue in $r14
#   5. copy [_mix, _store_result)
#   6. append the new registerValue, that is "move $r14, <addressRegister>"
# The word count of every label boundary is part of this contract:
#   [calc, _prefetch) = prologue, [_prefetch, _prefetch + 4) = the mask word,
#   [_prefetch + 4, _mix) = prefetch, [_mix, _store_result) = mix.
randomx_calc_dataset_item_la64_prefetch:
	# One overwritable word, never more.
	bstrpick.d	$r15, $r14, 0, 0
	slli.d	$r15, $r15, 6
	add.d	$r15, $r15, $r12
	preld	0, $r15, 0

	# The generated superscalar hash program is inserted HERE.

randomx_calc_dataset_item_la64_mix:
	ld.d	$r22, $r15, 0
	ld.d	$r23, $r15, 8
	xor	$r4, $r4, $r22
	xor	$r5, $r5, $r23
	ld.d	$r22, $r15, 16
	ld.d	$r23, $r15, 24
	xor	$r6, $r6, $r22
	xor	$r7, $r7, $r23
	ld.d	$r22, $r15, 32
	ld.d	$r23, $r15, 40
	xor	$r8, $r8, $r22
	xor	$r9, $r9, $r23
	ld.d	$r22, $r15, 48
	ld.d	$r23, $r15, 56
	xor	$r10, $r10, $r22
	xor	$r11, $r11, $r23

randomx_calc_dataset_item_la64_store_result:
	st.d	$r4, $r13, 0
	st.d	$r5, $r13, 8
	st.d	$r6, $r13, 16
	st.d	$r7, $r13, 24
	st.d	$r8, $r13, 32
	st.d	$r9, $r13, 40
	st.d	$r10, $r13, 48
	st.d	$r11, $r13, 56

	ld.d	$r1, $r3, 0
	ld.d	$r4, $r3, 8
	ld.d	$r5, $r3, 16
	ld.d	$r6, $r3, 24
	ld.d	$r7, $r3, 32
	ld.d	$r8, $r3, 40
	ld.d	$r9, $r3, 48
	ld.d	$r10, $r3, 56
	ld.d	$r11, $r3, 64
	ld.d	$r12, $r3, 72
	ld.d	$r13, $r3, 80
	ld.d	$r14, $r3, 88
	ld.d	$r15, $r3, 96
	ld.d	$r21, $r3, 104
	ld.d	$r22, $r3, 112
	ld.d	$r23, $r3, 120
	addi.d	$r3, $r3, 136
	ret

randomx_calc_dataset_item_la64_end:

	# End of the translated A64 code. jit_compiler_la64.cpp fills the
	# overwritable slots above at the offsets declared here.
