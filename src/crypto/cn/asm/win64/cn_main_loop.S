#define ALIGN(x) .align 64
.intel_syntax noprefix
.section .text
.global cnv2_mainloop_ivybridge_asm
.global cnv2_mainloop_ryzen_asm
.global cnv2_mainloop_bulldozer_asm
.global cnv2_double_mainloop_sandybridge_asm
.global cnv2_rwz_mainloop_asm
.global cnv2_rwz_double_mainloop_asm

ALIGN(64)
cnv2_mainloop_ivybridge_asm:
	#include "../cn2/cnv2_main_loop_ivybridge.inc"
	ret 0
	mov eax, 3735929054

ALIGN(64)
cnv2_mainloop_ryzen_asm:
	#include "../cn2/cnv2_main_loop_ryzen.inc"
	ret 0
	mov eax, 3735929054

ALIGN(64)
cnv2_mainloop_bulldozer_asm:
	#include "../cn2/cnv2_main_loop_bulldozer.inc"
	ret 0
	mov eax, 3735929054

ALIGN(64)
cnv2_double_mainloop_sandybridge_asm:
	#include "../cn2/cnv2_double_main_loop_sandybridge.inc"
	ret 0
	mov eax, 3735929054

ALIGN(64)
cnv2_rwz_mainloop_asm:
	#include "cn2/cnv2_rwz_main_loop.inc"
	ret 0
	mov eax, 3735929054

ALIGN(64)
cnv2_rwz_double_mainloop_asm:
	#include "cn2/cnv2_rwz_double_main_loop.inc"
	ret 0
	mov eax, 3735929054
