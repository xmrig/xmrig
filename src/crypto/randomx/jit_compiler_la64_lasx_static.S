/*
  Copyright (c) 2024, RESBI <https://github.com/RESBI>

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above
	  copyright notice, this list of conditions and the following
	  disclaimer.
	* Redistributions in binary form must reproduce the above
	  copyright notice, this list of conditions and the following
	  disclaimer in the documentation and/or other materials provided
	  with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this source without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, DAMAGES, OR OTHER LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

# The 256 bit ( LASX ) dataset item function of the LoongArch build.
#
# This file is compiled ONLY when the build carries XMRIG_FEATURE_LASX,
# see cmake/randomx.cmake.  A machine that only carries LSX never
# executes one word of it, and a build with -DWITH_LASX=OFF does not
# even carry it in its binary.
#
# The 128 bit function in jit_compiler_la64_static.S calculates ONE
# item at a time.  $xr carries four 64 bit lanes, so the very same
# program calculates FOUR items at once here, one item per lane:
#
#	lane k holds item ( startItem + k )
#
#	$xr0 - $xr7   -> rl[0] - rl[7], lane k = rl[n] of item k
#	$xr8          -> registerValue, lane k = item k
#	$xr9          -> the cache line index mask, four equal lanes
#	$xr10         -> the cache memory, four equal lanes
#	$xr11 - $xr18 -> the collided cache line, two 256 bit
#	                 registers per item
#	$xr19, $xr20  -> the transposition scratch
#	$xr21 - $xr23  -> the scratch of the generated program
#
# The JIT compiler builds the body exactly like the 128 bit one:
#	1. copy [ randomx_calc_dataset_item_la64_lasx, _prefetch )
#	2. emit " lu12i.w $r21, <the mask> ",
#	        " ori $r21, $r21, <the mask> " and
#	        " xvreplgr2vr.d $xr9, $r21 " -- $xr9 is the
#	        one value the JIT compiler owns; a table
#	        would not do, see the note below
#	3. copy [ _prefetch, _mix ) once per cache access
#	4. append the generated superscalar program
#	5. copy [ _mix, _store_result )
#	6. append " xvor.v $xr8, $xr<addressRegister>, $xr<addressRegister> "
#	7. copy [ _store_result, _end ) once
#
# " la.local " is PC relative, but the two words it turns into
# carry a *relocation*: the linker of the shared object, of the
# xmrig binary, fills in the address inside the final image,
# so the copy the JIT compiler makes keeps pointing at the
# tables of the image, NOT at its own bytes.  Therefore no
# table entry below may ever be written at run time: the
# whole mask of the JIT compiler travels as the
# words of step 2 above, and the copy
# reads the read only tables.
#

	.text

	.global	randomx_init_dataset_la64_lasx
	.global	randomx_init_dataset_la64_lasx_end
	.global	randomx_calc_dataset_item_la64_lasx
	.global	randomx_calc_dataset_item_la64_lasx_prefetch
	.global	randomx_calc_dataset_item_la64_lasx_mix
	.global	randomx_calc_dataset_item_la64_lasx_store_result
	.global	randomx_calc_dataset_item_la64_lasx_end
	.global	randomx_calc_dataset_item_la64_lasx_constants
	.global	randomx_calc_dataset_item_la64_lasx_lanes

	.balign	4

# Input parameters
#
# $r4 -> pointer to the cache pointer
# $r5 -> pointer to the dataset memory at startItem
# $r6 -> start item
# $r7 -> end item
#
# The item function below saves $r1 and $r4 - $r6, so those
# arguments survive every call.
randomx_init_dataset_la64_lasx:
	# The " bl " below writes the return address into $r1.
	addi.d	$r3, $r3, -16
	st.d	$r1, $r3, 0

	# The memory of the cache.
	ld.d	$r4, $r4, 0

randomx_init_dataset_la64_lasx_main_loop:
	# This word is rewritten by the JIT compiler: the item
	# function moves into the same copied block.
randomx_init_dataset_la64_lasx_bl:
	bl	randomx_calc_dataset_item_la64_lasx
	addi.d	$r5, $r5, 256
	addi.d	$r6, $r6, 4
	blt	$r6, $r7, randomx_init_dataset_la64_lasx_main_loop

	ld.d	$r1, $r3, 0
	addi.d	$r3, $r3, 16

	ret

randomx_init_dataset_la64_lasx_end:

# Input parameters
#
# $r4 -> pointer to the cache memory
# $r5 -> pointer to the output, FOUR items
# $r6 -> the first item of the four
#
# The whole register file is saved, otherwise the caller
# could not keep $r4 - $r7 alive across the call.
randomx_calc_dataset_item_la64_lasx:
	addi.d		$r3, $r3, -80
	st.d		$r1, $r3, 0
	st.d		$r4, $r3, 8
	st.d		$r5, $r3, 16
	st.d		$r6, $r3, 24
	st.d		$r12, $r3, 32
	st.d		$r13, $r3, 40
	# The generated superscalar program
	# materialises its immediates
	# in $r21 - $r23 ( the
	# 128 bit body uses
	# $r4 - $r23 ).
	st.d		$r21, $r3, 48
	st.d		$r22, $r3, 56
	st.d		$r23, $r3, 64

	la.local	$r12, randomx_calc_dataset_item_la64_lasx_constants

	# $xr9, the broadcast of the cache line index
	# mask, is emitted right behind this
	# prologue by the JIT.

	# registerValue starts as the item number itself
	# ( dataset.cpp : uint64_t registerValue = itemNumber ),
	# one item per lane, so lane k starts at
	# item + k and the { 0, 1, 2, 3 }
	# table below carries the k.
	xvreplgr2vr.d	$xr8, $r6
	la.local	$r11, randomx_calc_dataset_item_la64_lasx_lanes
	xvld		$xr11, $r11, 0
	xvadd.d		$xr8, $xr8, $xr11

	# The cache memory, four equal lanes.
	xvreplgr2vr.d	$xr10, $r4

	# rl[0] = ( itemNumber + 1 ) * superscalarMul0,
	# one item per lane.
	xvld		$xr0, $r12, 0
	xvadd.d		$xr0, $xr0, $xr8
	ld.d		$r13, $r12, 32
	xvreplgr2vr.d	$xr1, $r13
	xvmul.d		$xr0, $xr0, $xr1

	# rl[1] - rl[7] = rl[0] ^ superscalarAdd1 - superscalarAdd7
	ld.d		$r13, $r12, 40
	xvreplgr2vr.d	$xr1, $r13
	xvxor.v		$xr1, $xr0, $xr1
	ld.d		$r13, $r12, 48
	xvreplgr2vr.d	$xr2, $r13
	xvxor.v		$xr2, $xr0, $xr2
	ld.d		$r13, $r12, 56
	xvreplgr2vr.d	$xr3, $r13
	xvxor.v		$xr3, $xr0, $xr3
	ld.d		$r13, $r12, 64
	xvreplgr2vr.d	$xr4, $r13
	xvxor.v		$xr4, $xr0, $xr4
	ld.d		$r13, $r12, 72
	xvreplgr2vr.d	$xr5, $r13
	xvxor.v		$xr5, $xr0, $xr5
	ld.d		$r13, $r12, 80
	xvreplgr2vr.d	$xr6, $r13
	xvxor.v		$xr6, $xr0, $xr6
	ld.d		$r13, $r12, 88
	xvreplgr2vr.d	$xr7, $r13
	xvxor.v		$xr7, $xr0, $xr7

	b	randomx_calc_dataset_item_la64_lasx_prefetch

	# The constants the prologue reads.  They live inside the
	# copied block because " la.local " is PC relative.
	#
	#	+0  - +24 : the four rl[0] offsets, { 1, 1, 1, 1 }
	#	+32      : superscalarMul0
	#	+40 - +88: superscalarAdd1 - superscalarAdd7
	#	+96      : the cache line index mask, filled in by the
	#	           JIT compiler before the block runs.
	.balign	8
randomx_calc_dataset_item_la64_lasx_constants:
	.quad	1
	.quad	1
	.quad	1
	.quad	1
	.quad	6364136223846793005
	.quad	9298411001130361340
	.quad	12065312585734608966
	.quad	9306329213124626780
	.quad	5281919268842080866
	.quad	10536153434571861004
	.quad	3398623926847679864
	.quad	9549104520008361294

	# The lane offsets, { 0, 1, 2, 3 } : the lane k
	# of the item function starts at item + k.
	.balign	32
randomx_calc_dataset_item_la64_lasx_lanes:
	.quad	0
	.quad	1
	.quad	2
	.quad	3

randomx_calc_dataset_item_la64_lasx_prefetch:
	xvand.v		$xr8, $xr8, $xr9
	xvslli.d	$xr8, $xr8, 6
	xvadd.d		$xr8, $xr8, $xr10
	xvpickve2gr.du	$r12, $xr8, 0
	preld		0, $r12, 0
	xvpickve2gr.du	$r13, $xr8, 1
	preld		0, $r13, 0
	xvpickve2gr.du	$r12, $xr8, 2
	preld		0, $r12, 0
	xvpickve2gr.du	$r13, $xr8, 3
	preld		0, $r13, 0

	# THE GENERATED SUPERSCALAR PROGRAM IS INSERTED HERE.
	#
	# Its register file is $xr0 - $xr7, it may use
	# $xr21 - $xr23 as its scratch, and it must leave
	# $xr8 ( registerValue ) and $xr9 / $xr10 alone.

randomx_calc_dataset_item_la64_lasx_mix:
	xvpickve2gr.du	$r12, $xr8, 0
	xvld		$xr11, $r12, 0
	xvld		$xr12, $r12, 32
	xvpickve2gr.du	$r13, $xr8, 1
	xvld		$xr13, $r13, 0
	xvld		$xr14, $r13, 32
	xvpickve2gr.du	$r12, $xr8, 2
	xvld		$xr15, $r12, 0
	xvld		$xr16, $r12, 32
	xvpickve2gr.du	$r13, $xr8, 3
	xvld		$xr17, $r13, 0
	xvld		$xr18, $r13, 32

	# rl[q] ^= the qword q of the four items.
	#
	# The four registers of one qword index have to meet
	# each other in one register, that is the five word
	# recipe below.
	xvilvl.d	$xr19, $xr11, $xr13
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvl.d	$xr20, $xr15, $xr17
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr0, $xr0, $xr19

	xvilvh.d	$xr19, $xr11, $xr13
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvh.d	$xr20, $xr15, $xr17
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr1, $xr1, $xr19

	xvilvl.d	$xr19, $xr11, $xr13
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvl.d	$xr20, $xr15, $xr17
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr2, $xr2, $xr19

	xvilvh.d	$xr19, $xr11, $xr13
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvh.d	$xr20, $xr15, $xr17
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr3, $xr3, $xr19

	xvilvl.d	$xr19, $xr12, $xr14
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvl.d	$xr20, $xr16, $xr18
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr4, $xr4, $xr19

	xvilvh.d	$xr19, $xr12, $xr14
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvh.d	$xr20, $xr16, $xr18
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr5, $xr5, $xr19

	xvilvl.d	$xr19, $xr12, $xr14
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvl.d	$xr20, $xr16, $xr18
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr6, $xr6, $xr19

	xvilvh.d	$xr19, $xr12, $xr14
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvh.d	$xr20, $xr16, $xr18
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvxor.v		$xr7, $xr7, $xr19

randomx_calc_dataset_item_la64_lasx_store_result:
	# item 0, rl[0] - rl[3]
	xvilvl.d	$xr19, $xr0, $xr1
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvl.d	$xr20, $xr2, $xr3
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 0

	# item 0, rl[4] - rl[7]
	xvilvl.d	$xr19, $xr4, $xr5
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvl.d	$xr20, $xr6, $xr7
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 32

	# item 1
	xvilvh.d	$xr19, $xr0, $xr1
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvh.d	$xr20, $xr2, $xr3
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 64

	xvilvh.d	$xr19, $xr4, $xr5
	xvpermi.d	$xr19, $xr19, 0xb1
	xvilvh.d	$xr20, $xr6, $xr7
	xvpermi.d	$xr20, $xr20, 0xb1
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 96

	# item 2
	xvilvl.d	$xr19, $xr0, $xr1
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvl.d	$xr20, $xr2, $xr3
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 128

	xvilvl.d	$xr19, $xr4, $xr5
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvl.d	$xr20, $xr6, $xr7
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 160

	# item 3
	xvilvh.d	$xr19, $xr0, $xr1
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvh.d	$xr20, $xr2, $xr3
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 192

	xvilvh.d	$xr19, $xr4, $xr5
	xvpermi.d	$xr19, $xr19, 0x1b
	xvilvh.d	$xr20, $xr6, $xr7
	xvpermi.d	$xr20, $xr20, 0x1b
	xvpermi.q	$xr19, $xr20, 0x02
	xvst		$xr19, $r5, 224

	# Every store above writes ONE WHOLE ITEM, that is 64
	# bytes: rl[0] - rl[7] of the item in question.  The
	# JIT compiler appends nothing here, the epilogue is
	# the fixed tail of the function.

	ld.d		$r1, $r3, 0
	ld.d		$r4, $r3, 8
	ld.d		$r5, $r3, 16
	ld.d		$r6, $r3, 24
	ld.d		$r12, $r3, 32
	ld.d		$r13, $r3, 40
	ld.d		$r21, $r3, 48
	ld.d		$r22, $r3, 56
	ld.d		$r23, $r3, 64
	addi.d		$r3, $r3, 80
	ret

randomx_calc_dataset_item_la64_lasx_end:

