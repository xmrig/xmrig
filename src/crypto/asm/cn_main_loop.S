#define ALIGN .align
.intel_syntax noprefix
#ifdef __APPLE__
#   define FN_PREFIX(fn) _ ## fn
.text
#else
#   define FN_PREFIX(fn) fn
.section .text
#endif
.global FN_PREFIX(cnv1_main_loop_sandybridge_asm)
.global FN_PREFIX(cnv1_main_loop_lite_sandybridge_asm)
.global FN_PREFIX(cnv1_main_loop_fast_sandybridge_asm)
.global FN_PREFIX(cnv1_main_loop_upx_sandybridge_asm)

.global FN_PREFIX(cnv2_main_loop_ivybridge_asm)
.global FN_PREFIX(cnv2_main_loop_ryzen_asm)
.global FN_PREFIX(cnv2_main_loop_bulldozer_asm)
.global FN_PREFIX(cnv2_double_main_loop_sandybridge_asm)

.global FN_PREFIX(cnv2_main_loop_fastv2_ivybridge_asm)
.global FN_PREFIX(cnv2_main_loop_fastv2_ryzen_asm)
.global FN_PREFIX(cnv2_main_loop_fastv2_bulldozer_asm)
.global FN_PREFIX(cnv2_double_main_loop_fastv2_sandybridge_asm)

.global FN_PREFIX(cnv2_main_loop_ultralite_ivybridge_asm)
.global FN_PREFIX(cnv2_main_loop_ultralite_ryzen_asm)
.global FN_PREFIX(cnv2_main_loop_ultralite_bulldozer_asm)
.global FN_PREFIX(cnv2_double_main_loop_ultralite_sandybridge_asm)

.global FN_PREFIX(cnv1_main_loop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cnv1_main_loop_lite_soft_aes_sandybridge_asm)
.global FN_PREFIX(cnv1_main_loop_fast_soft_aes_sandybridge_asm)
.global FN_PREFIX(cnv1_main_loop_upx_soft_aes_sandybridge_asm)

.global FN_PREFIX(cnv2_main_loop_soft_aes_sandybridge_asm)
.global FN_PREFIX(cnv2_main_loop_fastv2_soft_aes_sandybridge_asm)
.global FN_PREFIX(cnv2_main_loop_ultralite_soft_aes_sandybridge_asm)

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv1_main_loop_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_lite_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv1_main_loop_lite_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_fast_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv1_main_loop_fast_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_upx_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv1_main_loop_upx_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_ivybridge_asm):
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
FN_PREFIX(cnv2_main_loop_ryzen_asm):
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
FN_PREFIX(cnv2_main_loop_bulldozer_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_bulldozer.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_double_main_loop_sandybridge_asm):
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
FN_PREFIX(cnv2_main_loop_fastv2_ivybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_fastv2_ivybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_fastv2_ryzen_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_fastv2_ryzen.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_fastv2_bulldozer_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_fastv2_bulldozer.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_double_main_loop_fastv2_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	mov rdx, rsi
	#include "cnv2_double_main_loop_fastv2_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_ultralite_ivybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_ultralite_ivybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_ultralite_ryzen_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_ultralite_ryzen.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_ultralite_bulldozer_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_ultralite_bulldozer.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_double_main_loop_ultralite_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	mov rdx, rsi
	#include "cnv2_double_main_loop_ultralite_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_soft_aes_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv1_main_loop_soft_aes_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_lite_soft_aes_sandybridge_asm):
    sub rsp, 48
    mov rcx, rdi
    #include "cnv1_main_loop_lite_soft_aes_sandybridge.inc"
    add rsp, 48
    ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_fast_soft_aes_sandybridge_asm):
    sub rsp, 48
    mov rcx, rdi
    #include "cnv1_main_loop_fast_soft_aes_sandybridge.inc"
    add rsp, 48
    ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv1_main_loop_upx_soft_aes_sandybridge_asm):
    sub rsp, 48
    mov rcx, rdi
    #include "cnv1_main_loop_upx_soft_aes_sandybridge.inc"
    add rsp, 48
    ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_soft_aes_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_soft_aes_sandybridge.inc"
	add rsp, 48
	ret 0

#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_fastv2_soft_aes_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cnv2_main_loop_fastv2_soft_aes_sandybridge.inc"
	add rsp, 48
	ret 0


#ifdef __APPLE__
ALIGN 16
#else
ALIGN 64
#endif
FN_PREFIX(cnv2_main_loop_ultralite_soft_aes_sandybridge_asm):
    sub rsp, 48
    mov rcx, rdi
    #include "cnv2_main_loop_ultralite_soft_aes_sandybridge.inc"
    add rsp, 48
    ret 0