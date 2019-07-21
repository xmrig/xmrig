# Copyright (c) 2018-2019, tevador <tevador@gmail.com>
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

.intel_syntax noprefix
#if defined(__APPLE__)
.text
#define DECL(x) _##x
#else
.section .text
#define DECL(x) x
#endif

#if defined(__WIN32__) || defined(__CYGWIN__)
#define WINABI
#endif

.global DECL(randomx_program_prologue)
.global DECL(randomx_program_loop_begin)
.global DECL(randomx_program_loop_load)
.global DECL(randomx_program_start)
.global DECL(randomx_program_read_dataset)
.global DECL(randomx_program_read_dataset_sshash_init)
.global DECL(randomx_program_read_dataset_sshash_fin)
.global DECL(randomx_program_loop_store)
.global DECL(randomx_program_loop_end)
.global DECL(randomx_dataset_init)
.global DECL(randomx_program_epilogue)
.global DECL(randomx_sshash_load)
.global DECL(randomx_sshash_prefetch)
.global DECL(randomx_sshash_end)
.global DECL(randomx_sshash_init)
.global DECL(randomx_program_end)
.global DECL(randomx_reciprocal_fast)

#define RANDOMX_SCRATCHPAD_MASK      2097088
#define RANDOMX_DATASET_BASE_MASK    2147483584
#define RANDOMX_CACHE_MASK           4194303

#define db .byte

.balign 64
DECL(randomx_program_prologue):
#if defined(WINABI)
	#include "asm/program_prologue_win64.inc"
#else
	#include "asm/program_prologue_linux.inc"
#endif
	movapd xmm13, xmmword ptr mantissaMask[rip]
	movapd xmm14, xmmword ptr exp240[rip]
	movapd xmm15, xmmword ptr scaleMask[rip]
	jmp DECL(randomx_program_loop_begin)

.balign 64
	#include "asm/program_xmm_constants.inc"

.balign 64
DECL(randomx_program_loop_begin):
	nop

DECL(randomx_program_loop_load):
	#include "asm/program_loop_load.inc"

DECL(randomx_program_start):
	nop

DECL(randomx_program_read_dataset):
	#include "asm/program_read_dataset.inc"

DECL(randomx_program_read_dataset_sshash_init):
	#include "asm/program_read_dataset_sshash_init.inc"

DECL(randomx_program_read_dataset_sshash_fin):
	#include "asm/program_read_dataset_sshash_fin.inc"

DECL(randomx_program_loop_store):
	#include "asm/program_loop_store.inc"

DECL(randomx_program_loop_end):
	nop

.balign 64
DECL(randomx_dataset_init):
	push rbx
	push rbp
	push r12
	push r13
	push r14
	push r15
#if defined(WINABI)
	push rdi
	push rsi
	mov rdi, qword ptr [rcx] ;# cache->memory
	mov rsi, rdx ;# dataset
	mov rbp, r8  ;# block index
	push r9      ;# max. block index
#else
	mov rdi, qword ptr [rdi] ;# cache->memory
	;# dataset in rsi
	mov rbp, rdx  ;# block index
	push rcx      ;# max. block index
#endif
init_block_loop:
	prefetchw byte ptr [rsi]
	mov rbx, rbp
	.byte 232 ;# 0xE8 = call
	;# .set CALL_LOC, 
	.int 32768 - (call_offset - DECL(randomx_dataset_init))
call_offset:
	mov qword ptr [rsi+0], r8
	mov qword ptr [rsi+8], r9
	mov qword ptr [rsi+16], r10
	mov qword ptr [rsi+24], r11
	mov qword ptr [rsi+32], r12
	mov qword ptr [rsi+40], r13
	mov qword ptr [rsi+48], r14
	mov qword ptr [rsi+56], r15
	add rbp, 1
	add rsi, 64
	cmp rbp, qword ptr [rsp]
	jb init_block_loop
	pop rax
#if defined(WINABI)
	pop rsi
	pop rdi
#endif
	pop r15
	pop r14
	pop r13
	pop r12
	pop rbp
	pop rbx
	ret

.balign 64
DECL(randomx_program_epilogue):
	#include "asm/program_epilogue_store.inc"
#if defined(WINABI)
	#include "asm/program_epilogue_win64.inc"
#else
	#include "asm/program_epilogue_linux.inc"
#endif

.balign 64
DECL(randomx_sshash_load):
	#include "asm/program_sshash_load.inc"

DECL(randomx_sshash_prefetch):
	#include "asm/program_sshash_prefetch.inc"

DECL(randomx_sshash_end):
	nop

.balign 64
DECL(randomx_sshash_init):
	lea r8, [rbx+1]
	#include "asm/program_sshash_prefetch.inc"
	imul r8, qword ptr r0_mul[rip]
	mov r9, qword ptr r1_add[rip]
	xor r9, r8
	mov r10, qword ptr r2_add[rip]
	xor r10, r8
	mov r11, qword ptr r3_add[rip]
	xor r11, r8
	mov r12, qword ptr r4_add[rip]
	xor r12, r8
	mov r13, qword ptr r5_add[rip]
	xor r13, r8
	mov r14, qword ptr r6_add[rip]
	xor r14, r8
	mov r15, qword ptr r7_add[rip]
	xor r15, r8
	jmp DECL(randomx_program_end)

.balign 64
	#include "asm/program_sshash_constants.inc"
	
.balign 64
DECL(randomx_program_end):
	nop

DECL(randomx_reciprocal_fast):
#if !defined(WINABI)
	mov rcx, rdi
#endif
	#include "asm/randomx_reciprocal.inc"
