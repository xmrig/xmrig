#define ALIGN .align
.intel_syntax noprefix
.section .text
.global cnv2_mainloop_ivybridge_asm
.global cnv2_mainloop_ryzen_asm
.global cnv2_double_mainloop_sandybridge_asm

ALIGN 16
cnv2_mainloop_ivybridge_asm:
	#include "../cnv2_main_loop_ivybridge.inc"
	ret 0

ALIGN 16
cnv2_mainloop_ryzen_asm:
	#include "../cnv2_main_loop_ryzen.inc"
	ret 0

ALIGN 16
cnv2_double_mainloop_sandybridge_asm:
	#include "../cnv2_double_main_loop_sandybridge.inc"
	ret 0
