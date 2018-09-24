#define ALIGN .align
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

ALIGN 16
FN_PREFIX(cnv2_mainloop_ivybridge_asm):
#ifndef XMRIG_WINDOWS
	sub rsp, 48
	mov rcx, rdi
#endif
	#include "cnv2_main_loop_ivybridge.inc"
#ifndef XMRIG_WINDOWS
	add rsp, 48
#endif
	ret 0

ALIGN 16
FN_PREFIX(cnv2_mainloop_ryzen_asm):
#ifndef XMRIG_WINDOWS
	sub rsp, 48
	mov rcx, rdi
#endif
	#include "cnv2_main_loop_ryzen.inc"
#ifndef XMRIG_WINDOWS
	add rsp, 48
#endif
	ret 0
