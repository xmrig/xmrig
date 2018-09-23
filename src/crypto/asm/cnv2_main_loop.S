#define ALIGN .align
.intel_syntax noprefix
.section .text
.global cnv2_mainloop_ivybridge_asm
.global cnv2_mainloop_ryzen_asm

ALIGN 64
cnv2_mainloop_ivybridge_asm:
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_ivybridge.inc"
	add rsp, 48
	ret 0

ALIGN 64
cnv2_mainloop_ryzen_asm:
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_ryzen.inc"
	add rsp, 48
	ret 0
