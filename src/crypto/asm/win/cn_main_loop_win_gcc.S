#define ALIGN .align
.intel_syntax noprefix
#   define FN_PREFIX(fn) fn
.section .text

.global FN_PREFIX(cnv1_mainloop_sandybridge_asm)
.global FN_PREFIX(cn_litev1_mainloop_sandybridge_asm)
.global FN_PREFIX(cn_fast_mainloop_sandybridge_asm)
.global FN_PREFIX(cnv2_mainloop_ivybridge_asm)
.global FN_PREFIX(cnv2_mainloop_ryzen_asm)
.global FN_PREFIX(cnv2_mainloop_bulldozer_asm)
.global FN_PREFIX(cnv2_double_mainloop_sandybridge_asm)
.global FN_PREFIX(cn_fast2_mainloop_ivybridge_asm)
.global FN_PREFIX(cn_fast2_mainloop_ryzen_asm)
.global FN_PREFIX(cn_fast2_mainloop_bulldozer_asm)
.global FN_PREFIX(cn_fast2_double_mainloop_sandybridge_asm)
.global FN_PREFIX(cn_liteupx_mainloop_sandybridge_asm)

.global FN_PREFIX(cnv1_mainloop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cn_litev1_mainloop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cn_fast_mainloop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cnv2_mainloop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cn_fast2_mainloop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cn_liteupx_mainloop_soft_aes_sandybridge_asm)

ALIGN 64
FN_PREFIX(cnv1_mainloop_sandybridge_asm):
	#include "../cnv1_mainloop_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_litev1_mainloop_sandybridge_asm):
	#include "../cn_litev1_mainloop_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_fast_mainloop_sandybridge_asm):
	#include "../cn_fast_mainloop_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cnv2_mainloop_ivybridge_asm):
	#include "../cnv2_main_loop_ivybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cnv2_mainloop_ryzen_asm):
	#include "../cnv2_main_loop_ryzen.inc"
	ret 0

ALIGN 64
FN_PREFIX(cnv2_mainloop_bulldozer_asm):
	#include "../cnv2_main_loop_bulldozer.inc"
	ret 0

ALIGN 64
FN_PREFIX(cnv2_double_mainloop_sandybridge_asm):
	#include "../cnv2_double_main_loop_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_fast2_mainloop_ivybridge_asm):
	#include "../cn_fast2_main_loop_ivybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_fast2_mainloop_ryzen_asm):
	#include "../cn_fast2_main_loop_ryzen.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_fast2_mainloop_bulldozer_asm):
	#include "../cn_fast2_main_loop_bulldozer.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_fast2_double_mainloop_sandybridge_asm):
	#include "../cn_fast2_double_main_loop_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_liteupx_mainloop_sandybridge_asm):
	#include "../cn_liteupx_mainloop_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cnv1_mainloop_soft_aes_sandybridge_asm):
	#include "../cnv1_mainloop_soft_aes_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_litev1_mainloop_soft_aes_sandybridge_asm):
	#include "../cn_litev1_mainloop_soft_aes_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_fast_mainloop_soft_aes_sandybridge_asm):
	#include "../cn_fast_mainloop_soft_aes_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cnv2_mainloop_soft_aes_sandybridge_asm):
	#include "../cnv2_mainloop_soft_aes_sandybridge.inc"
	ret 0

ALIGN 64
FN_PREFIX(cn_fast2_mainloop_soft_aes_sandybridge_asm):
	#include "../cn_fast2_mainloop_soft_aes_sandybridge.inc"
	ret 0


ALIGN 64
FN_PREFIX(cn_liteupx_mainloop_soft_aes_sandybridge_asm):
	#include "../cn_liteupx_mainloop_soft_aes_sandybridge.inc"
	ret 0