#ifdef __APPLE__
#   define ALIGN(x) .align 6
#else
#   define ALIGN(x) .align 64
#endif
.intel_syntax noprefix
#ifdef __APPLE__
#   define FN_PREFIX(fn) _ ## fn
.text
#else
#   define FN_PREFIX(fn) fn
.section .text
#endif
.global FN_PREFIX(cnv2_mainloop_ivybridge_asm)
.global FN_PREFIX(cnv2_mainloop_ryzen_asm)
.global FN_PREFIX(cnv2_mainloop_bulldozer_asm)
.global FN_PREFIX(cnv2_double_mainloop_sandybridge_asm)

ALIGN(64)
FN_PREFIX(cnv2_mainloop_ivybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_main_loop_ivybridge.inc"
	add rsp, 48
	ret 0
	nop
	nop
	nop
	nop

ALIGN(64)
FN_PREFIX(cnv2_mainloop_ryzen_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_main_loop_ryzen.inc"
	add rsp, 48
	ret 0
	nop
	nop
	nop
	nop

ALIGN(64)
FN_PREFIX(cnv2_mainloop_bulldozer_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_main_loop_bulldozer.inc"
	add rsp, 48
	ret 0
	nop
	nop
	nop
	nop

ALIGN(64)
FN_PREFIX(cnv2_double_mainloop_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	mov rdx, rsi
	#include "cn2/cnv2_double_main_loop_sandybridge.inc"
	add rsp, 48
	ret 0
	nop
	nop
	nop
	nop
