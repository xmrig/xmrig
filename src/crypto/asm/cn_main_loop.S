#define ALIGN .align
.intel_syntax noprefix
#ifdef __APPLE__
#   define FN_PREFIX(fn) _ ## fn
.text
#else
#   define FN_PREFIX(fn) fn
.section .text
#endif
.global FN_PREFIX(cnv1_mainloop_sandybridge_asm)
.global FN_PREFIX(cnv2_mainloop_ivybridge_asm)
.global FN_PREFIX(cnv2_mainloop_ryzen_asm)
.global FN_PREFIX(cnv2_double_mainloop_sandybridge_asm)

.global FN_PREFIX(cnv1_mainloop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cnv2_mainloop_soft_aes_sandybridge_asm)

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_mainloop_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv1_mainloop_sandybridge.inc"
	add rsp, 48
	ret 0
#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_mainloop_ivybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_ivybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_mainloop_ryzen_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_ryzen.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_double_mainloop_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	mov rdx, rsi
	#include "cnv2_double_main_loop_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_mainloop_soft_aes_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv1_mainloop_soft_aes_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_mainloop_soft_aes_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_mainloop_soft_aes_sandybridge.inc"
	add rsp, 48
	ret 0
