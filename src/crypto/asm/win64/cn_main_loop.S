#define ALIGN(x) .align 64
.intel_syntax noprefix
.section .text
.global cnv2_mainloop_ivybridge_asm
.global cnv2_mainloop_ryzen_asm
.global cnv2_mainloop_bulldozer_asm
.global cnv2_double_mainloop_sandybridge_asm

ALIGN(64)
cnv2_mainloop_ivybridge_asm:
	#include "../cn2/cnv2_main_loop_ivybridge.inc"
	ret 0
	nop
	nop
	nop
	nop

ALIGN(64)
cnv2_mainloop_ryzen_asm:
	#include "../cn2/cnv2_main_loop_ryzen.inc"
	ret 0
	nop
	nop
	nop
	nop

ALIGN(64)
cnv2_mainloop_bulldozer_asm:
	#include "../cn2/cnv2_main_loop_bulldozer.inc"
	ret 0
	nop
	nop
	nop
	nop

ALIGN(64)
cnv2_double_mainloop_sandybridge_asm:
	#include "../cn2/cnv2_double_main_loop_sandybridge.inc"
	ret 0
	nop
	nop
	nop
	nop
