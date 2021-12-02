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
.global FN_PREFIX(cnv1_single_mainloop_asm)
.global FN_PREFIX(cnv1_double_mainloop_asm)
.global FN_PREFIX(cnv1_quad_mainloop_asm)
.global FN_PREFIX(cnv2_mainloop_ivybridge_asm)
.global FN_PREFIX(cnv2_mainloop_ryzen_asm)
.global FN_PREFIX(cnv2_mainloop_bulldozer_asm)
.global FN_PREFIX(cnv2_double_mainloop_sandybridge_asm)
.global FN_PREFIX(cnv2_rwz_mainloop_asm)
.global FN_PREFIX(cnv2_rwz_double_mainloop_asm)
.global FN_PREFIX(cnv2_upx_double_mainloop_zen3_asm)

ALIGN(64)
FN_PREFIX(cnv1_single_mainloop_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn1/cnv1_single_main_loop.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv1_double_mainloop_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn1/cnv1_double_main_loop.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv1_quad_mainloop_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn1/cnv1_quad_main_loop.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv2_mainloop_ivybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_main_loop_ivybridge.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv2_mainloop_ryzen_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_main_loop_ryzen.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv2_mainloop_bulldozer_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_main_loop_bulldozer.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv2_double_mainloop_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_double_main_loop_sandybridge.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv2_rwz_mainloop_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_rwz_main_loop.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv2_rwz_double_mainloop_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_rwz_double_main_loop.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cnv2_upx_double_mainloop_zen3_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cn2/cnv2_upx_double_mainloop_zen3.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif
